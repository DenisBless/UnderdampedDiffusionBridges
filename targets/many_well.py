from typing import List

import chex
import jax
import jax.numpy as jnp
from jax import random
import distrax
import math
#
import matplotlib.pyplot as plt

import wandb
from targets.base_target import Target
from utils.plot import plot_contours_2D, plot_marginal_pair


class ManyWellEnergy(Target):
    def __init__(
            self,
            dim: int = 5,
            m: float = 5,
            delta: float = 4,
            can_sample: bool = True,
            sample_bounds=None,
    ):
        self.d = dim
        self.m = m
        self.delta = jnp.array(delta)

        self._plot_bound = 3.0

        super().__init__(dim=dim, log_Z=self.log_Z, can_sample=can_sample)

    def log_prob(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]
        assert x.shape[1] == self.d, "Dimension mismatch"
        m = self.m
        d = self.d
        delta = self.delta

        prefix = x[:, :m]
        k = ((prefix ** 2 - delta) ** 2).sum(axis=1)

        suffix = x[:, m:]
        k2 = 0.5 * (suffix ** 2).sum(axis=1)

        log_probs = -k - k2
        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def log_prob_2D(self, x):
        batched = x.ndim == 2

        if not batched:
            x = x[None,]

        m = self.m
        d = self.d
        delta = self.delta

        prefix = x[:, :2]
        k = ((prefix ** 2 - delta) ** 2).sum(axis=1)

        # suffix = x[:, m:]
        # k2 = 0.5 * (suffix**2).sum(axis=1)
        k2 = 0

        log_probs = -k - k2
        if not batched:
            log_probs = jnp.squeeze(log_probs, axis=0)

        return log_probs

    def visualise(
            self,
            samples: chex.Array,
            axes: List[plt.Axes] = None,
            show: bool = False,
            savefig: bool = False,
    ) -> None:
        """Visualise samples from the model."""
        plt.close()
        fig, ax = plt.subplots()

        plot_contours_2D(self.log_prob_2D, ax, bound=self._plot_bound, levels=20)
        plot_marginal_pair(samples, ax, bounds=(-self._plot_bound, self._plot_bound))

        wb = {"figures/vis": [wandb.Image(fig)]}
        if show:
            plt.show()
        if savefig:
            plt.savefig("vis.png")
        return wb

    @property
    def log_Z(self):
        # numerical integration
        l, r = -100, 100
        s = 100000000
        key = jax.random.PRNGKey(0)

        pt = jax.random.uniform(key, (s,), minval=l, maxval=r)
        fst = jnp.log(jnp.sum(jnp.exp(-((pt ** 2 - self.delta) ** 2)) * ((r - l) / s)))

        self.logZ_1d = fst

        # well the below works but there's analytic solution this is Gaussian lmao - junhua
        pt = jax.random.uniform(key, (s,), minval=l, maxval=r)
        snd = jnp.log(jnp.sum(jnp.exp(-0.5 * pt ** 2) * ((r - l) / s)))

        return fst * self.m + snd * (self.d - self.m)

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape) -> chex.Array:

        REJECTION_SCALE = 6

        def doubleWell1dLogDensity(xs, shift, separation):
            return -((xs - shift) ** 2 - separation) ** 2 - self.logZ_1d

        def rejection_sampling(seed, shape, proposal, target_pdf, scaling):
            new_key, subkey1, subkey2 = random.split(seed, num=3)
            n_samples = math.prod(shape)
            samples = proposal.sample(seed=subkey1, sample_shape=(n_samples * math.ceil(scaling) * 10,))
            unif = random.uniform(subkey2, (samples.shape[0],))
            unif *= scaling * jnp.exp(proposal.log_prob(samples))
            accept = unif < target_pdf(samples).squeeze(1)
            samples = samples[accept]
            if samples.shape[0] >= n_samples:
                return jnp.reshape(samples[:n_samples], shape)
            else:
                new_shape = (n_samples - samples.shape[0],)
                new_samples = rejection_sampling(new_key, new_shape, proposal, target_pdf, scaling)
                return jnp.concat([samples.reshape(*shape, -1), new_samples])

        def GetProposal(shift, separation):
            # proposal distribution for 1D doubleWell rejection sampling
            loc = shift + jnp.sqrt(separation) * jnp.array([[-1.0], [1.0]])
            scale = 1 / jnp.sqrt(separation) * jnp.array([[1.0], [1.0]])
            ps = jnp.array([0.5, 0.5])
            components = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            gmm = distrax.MixtureSameFamily(
                mixture_distribution=distrax.Categorical(probs=ps),
                components_distribution=components
            )
            return gmm

        def Sample1DDoubleWell(seed, shape, shift, separation):
            proposal = GetProposal(shift, separation)
            target_pdf = lambda xs: jnp.exp(doubleWell1dLogDensity(xs, shift, separation))
            return rejection_sampling(seed, shape, proposal, target_pdf, REJECTION_SCALE)

        new_key, subkey1, subkey2 = random.split(seed, num=3)

        n_dw, n_gauss = self.m, self.d - self.m
        dw_samples = Sample1DDoubleWell(subkey1, sample_shape + (n_dw,), 0, self.delta)

        gauss_samples = random.normal(subkey2, sample_shape + (n_gauss,))

        return jnp.concat([dw_samples, gauss_samples], axis=-1)


def test_manywell(d=5, m=5, delta=4):
    MW554 = ManyWellEnergy(d, m, delta)
    print(MW554.log_Z)

    key = random.PRNGKey(0)
    samples = MW554.sample(key, (1000,))
    MW554.visualise(samples, savefig=True)

    MW114 = ManyWellEnergy(1, 1, 4)
    samples = jnp.squeeze(MW114.sample(key, (1000,)))  # reuse key but okay

    fig, ax = plt.subplots()
    ax.hist(samples, bins=50, density=True)

    X = jnp.linspace(-5, 5, 400)
    Y = MW114.log_prob(X[:, None])

    ax.plot(X, jnp.exp(Y) / jnp.exp(MW114.log_Z))
    plt.show()
    # plt.savefig("vis2.png")


if __name__ == "__main__":
    test_manywell(2, 2, 4)
