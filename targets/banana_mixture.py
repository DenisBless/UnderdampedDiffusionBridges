import chex
import jax
import jax.numpy as jnp
import jax.random
from matplotlib import pyplot as plt
from targets.base_target import Target
from scipy.stats import special_ortho_group
from jax.scipy.special import logsumexp
import numpyro.distributions as dist
import numpy as np
import wandb


class BananaDistribution:
    def __init__(self, seed, curvature, mean, var, translation):
        self.dim = len(mean)
        self.curvature = curvature
        self.base_dist = dist.MultivariateNormal(mean, var)
        self.translation = translation
        np.random.seed(seed)
        self.rotation = special_ortho_group.rvs(self.dim)

    def sample(self, seed, sample_shape):
        gaus_samples = self.base_dist.sample(seed, sample_shape)
        x = jnp.zeros_like(gaus_samples)

        # transform to banana shaped distribution
        x = x.at[:, 0].set(gaus_samples[:, 0])
        x = x.at[:, 1:].set(gaus_samples[:, 1:] +
                            self.curvature * jnp.square(gaus_samples[:, 0].reshape(-1, 1)) - 1 * self.curvature)
        # rotate samples
        x = jnp.dot(x, self.rotation)

        # translate samples
        x = x + self.translation
        return x

    def log_prob(self, samples):
        gaus_samples = jnp.zeros_like(samples)

        # translate back
        samples = samples - self.translation
        # rotate back
        samples = jnp.dot(samples, self.rotation.T)
        # transform back

        gaus_samples = gaus_samples.at[:, 0].set(samples[:, 0])
        gaus_samples = gaus_samples.at[:, 1:].set(samples[:, 1:] -
                                                  self.curvature * jnp.square(
            gaus_samples[:, 0].reshape(-1, 1)) + 1 * self.curvature)

        log_probs = self.base_dist.log_prob(gaus_samples)

        return log_probs

    def visualize(self):
        x, y = jnp.meshgrid(jnp.linspace(-10, 10, 100), jnp.linspace(-10, 10, 100))
        grid = jnp.c_[x.ravel(), y.ravel()]
        pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
        pdf_values = jnp.reshape(pdf_values, x.shape)
        plt.contourf(x, y, pdf_values, levels=20, cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.show()


class BananaMixtureModel(Target):
    def __init__(self, num_components, dim, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        # parameters
        super().__init__(dim, log_Z, can_sample)
        min_cov_val = .1
        max_cov_val = 1
        min_translation_val = -10
        max_translation_val = 10
        min_val_mixture_weight = 0.3
        max_val_mixture_weight = 0.7
        curvature_factor = 3

        key = jax.random.PRNGKey(1)

        self.num_components = num_components
        self.ndim = dim

        # set component distributions
        self.means = jnp.zeros((self.num_components, self.ndim))
        key, subkey = jax.random.split(key)
        self.covariances = jnp.array([jnp.eye(self.ndim) * 0.4 for _ in range(self.num_components)])\
            #                * jax.random.uniform(
            # key=subkey, minval=min_cov_val, maxval=max_cov_val, shape=(num_components, dim, dim))
        key, subkey = jax.random.split(key)
        self.translations = jax.random.uniform(subkey, minval=min_translation_val, maxval=max_translation_val,
                                               shape=(self.num_components, self.ndim))
        self.curvatures = jnp.ones(self.num_components) * curvature_factor

        key, subkey = jax.random.split(key)
        # set mixture weights

        uniform_mws = True
        if uniform_mws:
            self.mixture_weights = jnp.ones(num_components) / num_components
        else:
            self.mixture_weights = dist.Uniform(low=min_val_mixture_weight,
                                                high=max_val_mixture_weight).sample(subkey,
                                                                                    sample_shape=(num_components,))

        self.bananas = []
        for i in range(self.num_components):
            key, subkey = jax.random.split(key)
            self.bananas.append(BananaDistribution(subkey, self.curvatures[i], self.means[i], self.covariances[i],
                                                   self.translations[i]))

    def log_prob(self, samples: chex.Array) -> chex.Array:
        batched = samples.ndim == 2

        if not batched:
            samples = samples[None,]

        log_mixture_weights = jnp.log(self.mixture_weights)
        banana_log_probs = jnp.stack([banana.log_prob(samples) for banana in self.bananas])
        likelihoods = log_mixture_weights[:, jnp.newaxis] + banana_log_probs

        # log sum exp trick for numerical stability
        result = logsumexp(likelihoods, axis=0)

        if not batched:
            result = jnp.squeeze(result, axis=0)
        return result

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:
        categorical = dist.Categorical(probs=self.mixture_weights)
        component_assignments = categorical.sample(seed, sample_shape)

        # Sample from the selected components
        component_counts = jnp.bincount(component_assignments, length=self.num_components)
        non_zero_components = jnp.where(component_counts > 0)[0] # todo problems with jit

        samples = []
        for idx in non_zero_components:
            component_samples = self.bananas[idx].sample(seed, (component_counts[idx],))
            samples.append(component_samples)

        samples = jnp.concatenate(samples, 0)

        # shuffle so it is not biased
        indices = jnp.arange(len(component_assignments))
        shuffled_indices = jax.random.permutation(seed, indices)
        return samples[shuffled_indices]

    def entropy(self, samples: chex.Array = None):
        idx = jnp.argmax(jnp.stack([banana.log_prob(samples) for banana in self.bananas]).T, 1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_components)))
        return entropy

    def visualise(self, samples: chex.Array = None, axes=None, show=False, prefix='') -> dict:
        plt.close()
        seed = jax.random.PRNGKey(0)
        boarder = [-15, 15]

        if self.ndim > 2:
            return {}

        if self.ndim == 2:

            fig = plt.figure()
            ax = fig.add_subplot()
            x, y = jnp.meshgrid(jnp.linspace(boarder[0], boarder[1], 300),
                                jnp.linspace(boarder[0], boarder[1], 300))
            grid = jnp.c_[x.ravel(), y.ravel()]
            pdf_values = jax.vmap(jnp.exp)(self.log_prob(grid))
            pdf_values = jnp.reshape(pdf_values, x.shape)
            ax.contourf(x, y, pdf_values, levels=20, cmap='viridis')
            # ax.contour(x, y, jnp.log(pdf_values), levels=50)
            if samples is not None:
                plt.scatter(samples[:300, 0], samples[:300, 1], c='r', alpha=0.5, marker='x')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            plt.xticks([])
            plt.yticks([])
            wb = {"figures/vis": [wandb.Image(fig)]}
            if show:
                plt.show()

            return wb
        else:
            return {}



if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    # single banana in 2d
    # mean = jnp.array([0.0, 0.0])
    # covariance = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    # curvatures = jnp.array([2])
    # translation = jnp.array([0.0, 0.0])
    # banana = BananaDistribution(seed=key, curvature=curvatures, translation=translation, mean=mean, var=covariance)
    # banana.visualize()
    # banana.sample(key, (2,))

    bmm = BananaMixtureModel(dim=2, num_components=15)
    samples = bmm.sample(key, (1000,))
    print(bmm.entropy(samples))
    bmm.visualise(show=True)

    print(bmm.log_prob(samples))
    log_prob_grad = jax.vmap(jax.grad(bmm.log_prob))(samples)
    print(log_prob_grad)
