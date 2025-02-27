import pickle
from typing import Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import wandb
from chex import assert_trees_all_equal

import algorithms.common.types as tp
from targets.base_target import Target
from utils.path_utils import project_path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

Array = tp.Array
MNIST_IMAGE_SHAPE = tp.MNIST_IMAGE_SHAPE
Batch = tp.VaeBatch
RandomKey = tp.RandomKey
OptState = tp.OptState
UpdateFn = tp.UpdateFn
VAEResult = tp.VAEResult


def kl_divergence_standard_gaussian(mean, std) -> Array:
    """KL divergence from diagonal Gaussian with mean std to standard normal.

  Independence means the KL is a sum of KL divergences for each dimension.
  expectation_{q(x)}log prod_i q_i(x_i)/p_i(x_i)
  = sum_{i=1}^{N} expectation_{q_i(x_i)} log q_i(x_i) / p_i(x_i)
  So we have a sum of KL divergence between univariate Gaussians where
  p_i(x_i) is a standar normal.
  So each term is 0.5 * ((std)^2 + (mean)^2 - 1 - 2 ln (std) )
  Args:
    mean: Array of length (ndim,)
    std: Array of length (ndim,)
  Returns:
    KL-divergence Array of shape ().
  """
    chex.assert_rank([mean, std], [1, 1])
    terms = 0.5 * (jnp.square(std) + jnp.square(mean) - 1. - 2. * jnp.log(std))
    return jnp.sum(terms)


def batch_kl_divergence_standard_gaussian(mean, std) -> Array:
    """Mean KL divergence diagonal Gaussian with mean std to standard normal.

  Works for batches of mean/std.
  Independence means the KL is a sum of KL divergences for each dimension.
  expectation_{q(x)}log prod_i q_i(x_i)/p_i(x_i)
  = sum_{i=1}^{N} expectation_{q_i(x_i)} log q_i(x_i) / p_i(x_i)
  So we have a sum of KL divergence between univariate Gaussians where
  p_i(x_i) is a standar normal.
  So each term is 0.5 * ((std)^2 + (mean)^2 - 1 - 2 ln (std) )
  Args:
    mean: Array of length (batch,ndim)
    std: Array of length (batch,ndim)
  Returns:
    KL-divergence Array of shape ().
  """
    chex.assert_rank([mean, std], [2, 2])
    chex.assert_equal_shape([mean, std])
    batch_kls = jax.vmap(kl_divergence_standard_gaussian)(mean, std)
    return jnp.mean(batch_kls)


def generate_binarized_images(key: RandomKey, logits: Array) -> Array:
    return jax.random.bernoulli(key, jax.nn.sigmoid(logits))


def load_dataset(split: str, batch_size: int):
    """Load the dataset."""
    read_config = tfds.ReadConfig(shuffle_seed=1)
    ds = tfds.load(
        'binarized_mnist',
        split=split,
        shuffle_files=True,
        read_config=read_config)
    ds = ds.shuffle(buffer_size=10 * batch_size, seed=1)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=5)
    ds = ds.repeat()
    return iter(tfds.as_numpy(ds))


class ConvEncoder(hk.Module):
    """A residual network encoder with mean stdev outputs."""

    def __init__(self,
                 num_latents: int = 20):
        super().__init__()
        self._num_latents = num_latents

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        conv_a = hk.Conv2D(kernel_shape=(4, 4),
                           stride=(2, 2),
                           output_channels=16,
                           padding='valid')
        conv_b = hk.Conv2D(kernel_shape=(4, 4),
                           stride=(2, 2),
                           output_channels=32,
                           padding='valid')
        flatten = hk.Flatten()
        sequential = hk.Sequential([conv_a,
                                    jax.nn.relu,
                                    conv_b,
                                    jax.nn.relu,
                                    flatten])
        progress = sequential(x)

        def get_output_params(progress_in, name=None):
            flat_output = hk.Linear(self._num_latents, name=name)(progress_in)
            flat_output = hk.LayerNorm(create_scale=True,
                                       create_offset=True,
                                       axis=1)(flat_output)
            return flat_output

        latent_mean = get_output_params(progress)
        unconst_std_dev = get_output_params(progress)
        latent_std = jax.nn.softplus(unconst_std_dev)

        return latent_mean, latent_std


class ConvDecoder(hk.Module):
    """A residual network decoder with logit outputs."""

    def __init__(self, image_shape: Tuple[int, int, int] = MNIST_IMAGE_SHAPE):
        super().__init__()
        self._image_shape = image_shape

    def __call__(self,
                 z: Array) -> Tuple[Array, Array, Array]:
        linear_features = 7 * 7 * 32
        linear = hk.Linear(linear_features)
        progress = linear(z)
        hk.LayerNorm(create_scale=True,
                     create_offset=True,
                     axis=1)(progress)
        progress = jnp.reshape(progress, (-1, 7, 7, 32))
        deconv_a = hk.Conv2DTranspose(
            kernel_shape=(3, 3), stride=(2, 2), output_channels=64)
        deconv_b = hk.Conv2DTranspose(
            kernel_shape=(3, 3), stride=(2, 2), output_channels=32)
        deconv_c = hk.Conv2DTranspose(
            kernel_shape=(3, 3), stride=(1, 1), output_channels=1)
        sequential = hk.Sequential([deconv_a,
                                    jax.nn.relu,
                                    deconv_b,
                                    jax.nn.relu,
                                    deconv_c])
        progress = sequential(progress)
        return progress


class ConvVAE(hk.Module):
    """A VAE with residual nets, diagonal normal q and logistic mixture output."""

    def __init__(self, num_latents: int = 30,
                 output_shape: Tuple[int, int, int] = MNIST_IMAGE_SHAPE):
        super().__init__()
        self._num_latents = num_latents
        self._output_shape = output_shape
        self.encoder = ConvEncoder(self._num_latents)
        self.decoder = ConvDecoder()

    def __call__(self, x: Array) -> VAEResult:
        x = x.astype(jnp.float32)
        latent_mean, latent_std = self.encoder(x)
        latent = latent_mean + latent_std * jax.random.normal(
            hk.next_rng_key(), latent_mean.shape)
        free_latent = jax.random.normal(hk.next_rng_key(), latent_mean.shape)
        logits = self.decoder(latent)
        free_logits = self.decoder(free_latent)
        reconst_sample = jax.nn.sigmoid(logits)
        sample_image = jax.nn.sigmoid(free_logits)
        return VAEResult(sample_image, reconst_sample, latent_mean, latent_std,
                         logits)


def binary_cross_entropy_from_logits(logits: Array, labels: Array) -> Array:
    """Numerically stable implementation of binary cross entropy with logits.

  For an individual term we follow a standard manipulation of the loss:
  H = -label * log sigmoid(logit) - (1-label) * log (1-sigmoid(logit))
  = logit - label * logit + log(1+exp(-logit))
  or for logit < 0 we take a different version for numerical stability.
  = - label * logit + log(1+exp(logit))
  combining to avoid a conditional.
  = max(logit, 0) - label * logit + log(1+exp(-abs(logit)))

  Args:
    logits: (batch, sample_shape) containing logits of class probs.
    labels: (batch, sample_shape) containing {0, 1} class labels.
  Returns:
    sum of loss over all shape indices then mean of loss over batch index.
  """
    chex.assert_equal_shape([logits, labels])
    max_logits_zero = jax.nn.relu(logits)
    negative_abs_logits = -jnp.abs(logits)
    terms = max_logits_zero - logits * labels + jax.nn.softplus(negative_abs_logits)
    return jnp.sum(jnp.mean(terms, axis=0))


def vae_loss(target: Array, logits: Array, latent_mean: Array,
             latent_std: Array) -> Array:
    log_loss = binary_cross_entropy_from_logits(logits, target)
    kl_term = batch_kl_divergence_standard_gaussian(latent_mean, latent_std)
    free_energy = log_loss + kl_term
    return free_energy


class AutoEncoderLikelihood(Target):
    """Generative decoder log p(x,z| theta) as a function of latents z.

  This evaluates log p(x,z| theta) = log p(x, z| theta ) + log p(z) for a VAE.
  Here x is an binarized MNIST Image, z are real valued latents, theta denotes
  the generator neural network parameters.

  Since x is fixed and z is a random variable this is the log of an unnormalized
  z density p(x, z | theta)
  The normalizing constant is a marginal p(x | theta) = int p(x, z | theta) dz.
  The normalized target density is the posterior over latents p(z|x, theta).

  The likelihood uses a pretrained generator neural network.
  It is contained in a pickle file specifed by config.params_filesname

  A script producing such a pickle file can be found in train_vae.py

  The resulting pretrained network used in the AFT paper
  can be found at data/vae.pickle

  The binarized MNIST test set image used is specfied by config.image_index

  """
    def __init__(self, image_index, dim=30, log_Z=None, can_sample=False, sample_bounds=None) -> None:
        super().__init__(dim=dim, log_Z=log_Z, can_sample=can_sample)

        self._num_dim = dim
        self._vae_params = self._get_vae_params(project_path('targets/data/vae.pickle'))
        test_batch_size = 1
        test_ds = load_dataset(tfds.Split.TEST, test_batch_size)
        for unused_index in range(image_index):
            unused_batch = next(test_ds)
        self._test_image = next(test_ds)["image"]
        assert self._test_image.shape[0] == 1  # Batch size needs to be 1.
        assert self._test_image.shape[1:] == MNIST_IMAGE_SHAPE
        self.entropy_eval = hk.transform(self.cross_entropy_eval_func)
        self.logit_eval = hk.transform(self.get_logits)
        # print(f'ln Z = {self.estimate_log_z(5000)}')

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        return None

    def _check_constructor_inputs(self, config: tp.ConfigDict,
                                  sample_shape: tp.SampleShape):
        assert_trees_all_equal(sample_shape, (30,))
        num_mnist_test = 10000
        in_range = config.image_index >= 0 and config.image_index < num_mnist_test
        if not in_range:
            msg = "VAE image_index must be greater than or equal to zero "
            msg += "and strictly less than " + str(num_mnist_test) + "."
            raise ValueError(msg)

    def _get_vae_params(self, ckpt_filename):
        with open(ckpt_filename, "rb") as f:
            vae_params = pickle.load(f)
        return vae_params

    def cross_entropy_eval_func(self, data: Array, latent: Array) -> Array:
        """Evaluate the binary cross entropy for given latent and data.

    Needs to be called within a Haiku transform.

    Args:
      data: Array of shape (1, image_shape)
      latent: Array of shape (num_latent_dim,)

    Returns:
      Array, value of binary cross entropy for single data point in question.
    """
        chex.assert_rank(latent, 1)
        chex.assert_rank(data, 4)  # Shape should be (1, 28, 28, 1) hence rank 4.
        vae = ConvVAE()
        # New axis here required for batch size = 1 for VAE compatibility.
        batch_latent = latent[None, :]
        logits = vae.decoder(batch_latent)
        chex.assert_equal_shape([logits, data])
        return binary_cross_entropy_from_logits(logits, data)

    def get_logits(self, data: Array, latent: Array) -> Array:
        """Evaluate the binary cross entropy for given latent and data.

    Needs to be called within a Haiku transform.

    Args:
      data: Array of shape (1, image_shape)
      latent: Array of shape (num_latent_dim,)

    Returns:
      Array, value of binary cross entropy for single data point in question.
    """
        chex.assert_rank(latent, 1)
        chex.assert_rank(data, 4)  # Shape should be (1, 28, 28, 1) hence rank 4.
        vae = ConvVAE()
        # New axis here required for batch size = 1 for VAE compatibility.
        batch_latent = latent[None, :]
        logits = vae.decoder(batch_latent)
        return logits

    def log_prior(self, latent: Array) -> Array:
        """Latent shape (num_dim,) -> standard multivariate log density."""
        chex.assert_rank(latent, 1)
        log_norm_gaussian = -0.5 * self._num_dim * jnp.log(2. * jnp.pi)
        data_term = - 0.5 * jnp.sum(jnp.square(latent))
        return data_term + log_norm_gaussian

    def total_log_probability(self, latent: Array) -> Array:
        chex.assert_rank(latent, 1)
        log_prior = self.log_prior(latent)
        dummy_rng_key = 0
        # Data point log likelihood is negative of loss for batch size of 1.
        log_likelihood = -1. * self.entropy_eval.apply(
            self._vae_params, dummy_rng_key, self._test_image, latent)
        total_log_probability = log_prior + log_likelihood
        return total_log_probability

    def evaluate_log_density(self, x: Array) -> Array:
        return jax.vmap(self.total_log_probability)(x)

    def log_prob(self, x: chex.Array) -> chex.Array:
        batched = x.ndim == 2
        if not batched:
            log_prob = self.total_log_probability(x)
        else:
            log_prob = self.evaluate_log_density(x)
        return log_prob

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        plt.close()
        dummy_rng_key = 0
        # Data point log likelihood is negative of loss for batch size of 1.
        batched = samples.ndim == 2

        n = 25
        n_rows = int(np.sqrt(n))
        fig, ax = plt.subplots(n_rows, n_rows, figsize=(28, 28))
        x = samples[:n]

        logit_fn = lambda x: self.logit_eval.apply(self._vae_params, dummy_rng_key, self._test_image, x)
        logits = jax.vmap(logit_fn)(x)

        # Plot each image
        for i in range(n_rows):
            for j in range(n_rows):
                ax[i, j].imshow(logits[i * n_rows + j][0, :, :, 0], cmap='gray')
                ax[i, j].axis('off')

        fig2, ax2 = plt.subplots()
        ax2.imshow(self._test_image[0, :, :, 0])

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()

        return wb
