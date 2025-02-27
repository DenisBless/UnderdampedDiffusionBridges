import os
from datetime import datetime

import jax.numpy as jnp
import distrax
import chex
import jax.random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from targets.base_target import Target
from utils.path_utils import project_path


class GMM1D(Target):
    def __init__(self, dim=1, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        super().__init__(dim, log_Z, can_sample)

        self.num_comp = 4
        logits = jnp.ones(self.num_comp)
        mean = jnp.array([-3.0, 0.5, 1.5, 2.5]).reshape((-1, 1))
        scale = jnp.array([0.5, 0.6, 0.3, 0.4]).reshape((-1, 1))

        # self.num_comp = 2
        # logits = jnp.ones(self.num_comp)
        # mean = jnp.array([2.5, 3.5]).reshape((-1, 1))
        # scale = jnp.array([0.6, 0.3]).reshape((-1, 1))

        # self.num_comp = 1
        # logits = jnp.ones(self.num_comp)
        # mean = jnp.array([0.]).reshape((-1, 1))
        # scale = jnp.array([1.]).reshape((-1, 1))

        self.mixture_dist = distrax.Categorical(logits=logits)
        self.components_dist = distrax.Independent(
            distrax.Normal(loc=mean, scale=scale), reinterpreted_batch_ndims=1
        )
        self.distribution = distrax.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            components_distribution=self.components_dist,
        )

        self._plot_bound = 5

    def log_prob(self, x: chex.Array) -> chex.Array:

        assert x.shape[-1] == 1, f"The last dimension of x should be 1, but got {x.shape[-1]}"

        batched = x.ndim == 2
        if not batched:
            x = x[None,]

        log_prob = self.distribution.log_prob(x)

        if not batched:
            log_prob = jnp.squeeze(log_prob, axis=0)

        return log_prob

    def sample(self, seed: chex.PRNGKey, sample_shape: chex.Shape = ()):
        return self.distribution.sample(seed=seed, sample_shape=sample_shape)

    def sample_with_component_indice(self, seed, sample_shape):
        key, keygen = jax.random.split(seed)
        component_indices = self.mixture_dist.sample(seed=key, sample_shape=sample_shape)
        key, keygen = jax.random.split(keygen)
        # Sample from components based on component indices
        components_samples = self.components_dist.sample(seed=key, sample_shape=sample_shape)

        # Gather samples according to the component indices
        samples = jnp.take_along_axis(components_samples, component_indices[:, None, None], axis=1)
        samples = jnp.squeeze(samples, axis=1)
        # now lets consider last 3 components as one component
        modified_component_indices = jnp.where((component_indices == 2) | (component_indices == 3), 1,
                                               component_indices)

        return samples, modified_component_indices

    def entropy(self, samples: chex.Array = None):
        expanded = jnp.expand_dims(samples, axis=-2)
        # Compute `log_prob` in every component.
        idx = jnp.argmax(self.distribution.components_distribution.log_prob(expanded), 1)
        unique_elements, counts = jnp.unique(idx, return_counts=True)
        mode_dist = counts / samples.shape[0]
        entropy = -jnp.sum(mode_dist * (jnp.log(mode_dist) / jnp.log(self.num_comp)))
        return entropy

    def simple_forward_visualization(self, samples, show, wb, step):
        x_range = (-7, 7)
        resolution = 100

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        marg_dens, _ = np.histogram(samples, bins=y_grid, density=True)
        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])

        y_range = (y_grid[0], y_grid[-1])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), )

        ax.set_xlim(*y_range)
        ax.set_xlabel('$x$')

        ax.hist(y_grid[:-1], weights=marg_dens, range=y_range, bins=y_grid, color=dark_gray[0],
                orientation='vertical', edgecolor='white', linewidth=0.75, density=True)

        x_values = np.linspace(*x_range, 1000)
        x_values = jnp.expand_dims(x_values, 1)
        log_probs = jnp.exp(self.log_prob(x_values))

        ax.plot(x_values, log_probs, label='$g(x)$', color='black')
        ax.set_ylim(0, np.max(log_probs) * 1.1)
        fig.suptitle(f'1D GMM Approximation Iteration: ' + str(step))
        ax.set_title('$\\pi(x)$')

        wb["figures/vis"] = wandb.Image(fig)
        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def simple_forward_visualization_presentation(self, samples, show, wb, step, samples_component_indices,
                                                  component_wise=True):
        x_range = (-7, 7)
        resolution = 100

        weights = jnp.bincount(samples_component_indices) / samples_component_indices.shape[0]
        if not component_wise:
            samples = jnp.expand_dims(samples, axis=0)
        else:
            unique_components = np.unique(samples_component_indices)
            samples = [samples[samples_component_indices == comp] for comp in unique_components]

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        #        marg_dens = np.histogram(samples, bins=y_grid, density=True)[0]
        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, 2))

        y_range = (y_grid[0], y_grid[-1])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), )

        ax.set_xlim(*y_range)
        ax.set_xlabel('$x$')
        ax.axis('off')

        ###########################
        x_0 = [np.histogram(comp[:, 0], bins=y_grid, density=True)[0] for comp in samples]
        y_0_all = np.broadcast_to(y_grid[:-1], (len(x_0), y_grid.size - 1)).T
        prior_counts = np.vstack(x_0).T * np.broadcast_to(weights, (y_grid.size - 1, len(x_0)))
        ############################
        ax.hist(y_0_all, weights=prior_counts, range=y_range, bins=y_grid, color=comp_colors,
                orientation='vertical', edgecolor='white', linewidth=0.75, stacked=True, histtype='bar')

        x_values = np.linspace(*x_range, 1000)
        x_values = jnp.expand_dims(x_values, 1)
        log_probs = jnp.exp(self.log_prob(x_values))

        ax.plot(x_values, log_probs, label='$g(x)$', color='black')
        ax.set_ylim(0, np.max(log_probs) * 1.1)
        # fig.suptitle(f'1D GMM Approximation Iteration: ' + str(step))
        # ax.set_title('$\\pi(x)$')

        wb["figures/vis"] = wandb.Image(fig)
        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def forward_diffusion_visualization(self, prior_samples, prior_log_prob, x_t_prior_to_target, show, wb, step,
                                        params=None):
        trajectory = jnp.concatenate((jnp.reshape(prior_samples, (-1, 1, 1)), jnp.expand_dims(x_t_prior_to_target, -1)),
                                     axis=1)
        trajectory = jnp.expand_dims(trajectory, axis=0)

        # diffusion traj
        x_range = (-7, 7)
        resolution = 100
        num_trajectories = 10

        # model_sample_list ist array of (component, n_batch, n_time, dim)
        model_samples = np.vstack(trajectory)
        # model_samples is (n_batch, n_time, dim)

        time_line = model_samples.shape[1]
        model_samples = model_samples.squeeze()  # get rid of target dim (1)

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        x_grid = np.arange(time_line)

        target_log_probs = np.exp(self.log_prob(y_grid.reshape((-1, 1))).flatten())

        # first count frequencies then plot
        marg_dens = np.zeros((resolution, time_line))

        for t in range(time_line):
            p_t, _ = np.histogram(model_samples[:, t], bins=y_grid, density=True)
            marg_dens[:, t] = p_t

        trajectories = [comp[np.random.choice(comp.shape[0], num_trajectories, replace=False), :] for comp in
                        trajectory]
        # at last step we want to distinguish contributions of different components again
        x_T = [np.histogram(comp[:, -1], bins=y_grid, density=True)[0] for comp in trajectory]

        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        num_components = len(trajectories)
        if num_components == 1:
            comp_colors = dark_gray
        else:
            comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, num_components))

        y_range = (y_grid[0], y_grid[-1])
        # plot
        fig, ax = plt.subplots(1, 3, figsize=(6, 4), gridspec_kw={'width_ratios': [1, 4, 1]})
        # fig.suptitle(f'Diffusion Plot Iteration: ' + str(step))

        ax[0].set_ylim(*y_range)
        ax[0].set_xlim(0, np.max(target_log_probs) * 1.1)

        # ax[0].plot(np.exp(np.array(norm.logpdf(y_grid, 0.0, model_init_std))), y_grid,
        #            color='black', linewidth=0.75)

        ax[1].set_ylim(*y_range)

        # grids must be one larger that dimensions of marg_dens
        pcm = ax[1].pcolormesh(np.insert(x_grid, -1, x_grid[-1] + 1), y_grid, marg_dens)
        # add trajectories
        for j, component in enumerate(trajectories):
            # disturb color by uniform noise to give each trajectory unique color
            cols = comp_colors[j]
            for j in range(component.shape[0]):  # num_trajecories
                ax[1].plot(x_grid, component[j, :], color='w', linewidth=0.75)

        ax[2].set_ylim(*y_range)
        if x_T is None:
            x_T = [marg_dens[:, -1]]
        y_all = np.broadcast_to(y_grid[:-1], (len(x_T), y_grid.size - 1)).T
        target_counts = np.vstack(x_T).T
        # plot for each component target samples separately and stack bars

        ax[2].set_xlim(0, np.max(target_log_probs) * 1.1)

        """ Use this for paper plot"""
        # high_res_y = np.linspace(*x_range, 1000)
        # self.plot_colored_line(ax[2],  np.exp(self.log_prob(high_res_y.reshape((-1, 1))).flatten()), high_res_y, cmap=pcm.get_cmap())
        # self.plot_colored_line(ax[0],  np.exp(prior_log_prob(high_res_y.reshape((-1, 1))).flatten()), high_res_y, cmap=pcm.get_cmap())

        ax[1].set_title('$X_t$')
        ax[1].set_xlabel('$t$')
        ax[2].set_title('$X_T$')
        ax[0].set_title('$X_0$')
        ax[2].plot(target_log_probs, y_grid, color='black', linewidth=0.75)
        ax[0].plot(np.exp(prior_log_prob(y_grid.reshape((-1, 1)), params).flatten()), y_grid, color='black',
                   linewidth=0.75)

        ax[2].hist(y_all, weights=target_counts, range=y_range, bins=y_grid, color=comp_colors,
                   orientation='horizontal', edgecolor='white', linewidth=0.75, histtype='bar', stacked=True)
        ax[0].hist(y_grid[:-1], weights=marg_dens[:, 0], range=y_range, bins=y_grid, color=dark_gray[0],
                   orientation='horizontal', edgecolor='white', linewidth=0.75)

        plt.subplots_adjust(wspace=0)  # Set wspace to 0 for no horizontal space

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)

        plt.setp(ax[0].get_yticklabels(), visible=False)
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[2].get_xticklabels(), visible=False)
        plt.setp(ax[0].set_xticks([]))
        plt.setp(ax[1].set_xticks([]))
        plt.setp(ax[2].set_xticks([]))
        plt.setp(ax[0].set_yticks([]))
        plt.setp(ax[1].set_yticks([]))
        plt.setp(ax[2].set_yticks([]))

        wb["figures/diffusion_vis"] = [wandb.Image(fig)]

        path = project_path() + '/figures/diff_plot'
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.pdf'), bbox_inches='tight', pad_inches=0.1, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def plot_colored_line(self, axes, x, y, cmap):
        # Sort data for continuous line
        # sorted_indices = np.argsort(x)
        # x_sorted = x[sorted_indices]
        # y_sorted = y[sorted_indices]

        # Create line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize x for color mapping
        norm = Normalize(x.min(), x.max())
        # colors = plt.cm.viridis(norm(x_sorted))

        # Create a LineCollection
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(x)
        lc.set_linewidth(2)

        # Plot
        axes.add_collection(lc)

    def forward_position_visualization(self, ax, prior_samples, prior_log_prob, x_t_prior_to_target, show, wb, step):
        trajectory = jnp.concatenate((jnp.reshape(prior_samples, (-1, 1, 1)), jnp.expand_dims(x_t_prior_to_target, -1)),
                                     axis=1)
        trajectory = jnp.expand_dims(trajectory, axis=0)

        # diffusion traj
        x_range = (-7, 7)
        resolution = 100
        num_trajectories = 10

        # model_sample_list ist array of (component, n_batch, n_time, dim)
        model_samples = np.vstack(trajectory)
        # model_samples is (n_batch, n_time, dim)

        time_line = model_samples.shape[1]
        model_samples = model_samples.squeeze()  # get rid of target dim (1)

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        x_grid = np.arange(time_line)

        target_log_probs = np.exp(self.log_prob(y_grid.reshape((-1, 1))).flatten())

        # first count frequencies then plot
        marg_dens = np.zeros((resolution, time_line))

        for t in range(time_line):
            p_t, _ = np.histogram(model_samples[:, t], bins=y_grid, density=True)
            marg_dens[:, t] = p_t

        trajectories = [comp[np.random.choice(comp.shape[0], num_trajectories, replace=False), :] for comp in
                        trajectory]
        # at last step we want to distinguish contributions of different components again
        x_T = [np.histogram(comp[:, -1], bins=y_grid, density=True)[0] for comp in trajectory]

        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        num_components = len(trajectories)
        if num_components == 1:
            comp_colors = dark_gray
        else:
            comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, num_components))

        y_range = (y_grid[0], y_grid[-1])
        # plot
        # fig, ax = plt.subplots(1, 3, figsize=(6, 4), gridspec_kw={'width_ratios': [1, 4, 1]})
        # fig.suptitle(f'Diffusion Plot Iteration: ' + str(step))

        ax[0].set_ylim(*y_range)
        ax[0].set_xlim(0, np.max(target_log_probs) * 1.1)
        ax[0].set_title('$X_0$')
        ax[0].hist(y_grid[:-1], weights=marg_dens[:, 0], range=y_range, bins=y_grid, color=dark_gray[0],
                   orientation='horizontal', edgecolor='white', linewidth=0.75)
        # ax[0].plot(np.exp(np.array(norm.logpdf(y_grid, 0.0, model_init_std))), y_grid,
        #            color='black', linewidth=0.75)

        ax[1].set_ylim(*y_range)
        ax[1].set_title('$X_t$')
        # ax[1].set_xlabel('$t$')
        # grids must be one larger that dimensions of marg_dens
        pcm = ax[1].pcolormesh(np.insert(x_grid, -1, x_grid[-1] + 1), y_grid, marg_dens)
        # add trajectories
        for j, component in enumerate(trajectories):
            # disturb color by uniform noise to give each trajectory unique color
            cols = comp_colors[j]
            for j in range(component.shape[0]):  # num_trajecories
                ax[1].plot(x_grid, component[j, :], color='w', linewidth=0.75)

        ax[2].set_ylim(*y_range)
        ax[2].set_title('$X_T$')
        if x_T is None:
            x_T = [marg_dens[:, -1]]
        y_all = np.broadcast_to(y_grid[:-1], (len(x_T), y_grid.size - 1)).T
        target_counts = np.vstack(x_T).T
        # plot for each component target samples separately and stack bars
        ax[2].hist(y_all, weights=target_counts, range=y_range, bins=y_grid, color=comp_colors,
                   orientation='horizontal', edgecolor='white', linewidth=0.75, histtype='bar', stacked=True)
        ax[2].plot(target_log_probs, y_grid, color='black', linewidth=0.75)
        # ax[0].plot(jnp.exp(prior_log_prob(y_grid.reshape(-1, 1))), y_grid, color='black', linewidth=0.75)

        ax[2].set_xlim(0, np.max(target_log_probs) * 1.1)

        # high_res_y = np.linspace(*x_range, 1000)
        # self.plot_colored_line(ax[2],  np.exp(self.log_prob(high_res_y.reshape((-1, 1))).flatten()), high_res_y, cmap=pcm.get_cmap())
        # self.plot_colored_line(ax[0],  np.exp(prior_log_prob(high_res_y.reshape((-1, 1))).flatten()), high_res_y, cmap=pcm.get_cmap())

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)

        plt.setp(ax[0].get_yticklabels(), visible=False)
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[2].get_xticklabels(), visible=False)
        plt.setp(ax[0].set_xticks([]))
        plt.setp(ax[1].set_xticks([]))
        plt.setp(ax[2].set_xticks([]))
        plt.setp(ax[0].set_yticks([]))
        plt.setp(ax[1].set_yticks([]))
        plt.setp(ax[2].set_yticks([]))

        return ax

    def forward_position_visualization_paper(self, ax, prior_samples, prior_log_prob, x_t_prior_to_target, show, wb,
                                             step):
        trajectory = jnp.concatenate((jnp.reshape(prior_samples, (-1, 1, 1)), jnp.expand_dims(x_t_prior_to_target, -1)),
                                     axis=1)
        trajectory = jnp.expand_dims(trajectory, axis=0)

        # diffusion traj
        x_range = (-7, 7)
        resolution = 100
        num_trajectories = 5

        # model_sample_list ist array of (component, n_batch, n_time, dim)
        model_samples = np.vstack(trajectory)
        # model_samples is (n_batch, n_time, dim)

        time_line = model_samples.shape[1]
        model_samples = model_samples.squeeze()  # get rid of target dim (1)

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        x_grid = np.arange(time_line)

        target_log_probs = np.exp(self.log_prob(y_grid.reshape((-1, 1))).flatten())

        # first count frequencies then plot
        marg_dens = np.zeros((resolution, time_line))

        for t in range(time_line):
            p_t, _ = np.histogram(model_samples[:, t], bins=y_grid, density=True)
            marg_dens[:, t] = p_t

        trajectories = [comp[np.random.choice(comp.shape[0], num_trajectories, replace=False), :] for comp in
                        trajectory]
        # at last step we want to distinguish contributions of different components again
        x_T = [np.histogram(comp[:, -1], bins=y_grid, density=True)[0] for comp in trajectory]

        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        light_gray = np.array([[90 / 255, 90 / 255, 90 / 255, .3]])
        num_components = len(trajectories)
        if num_components == 1:
            comp_colors = dark_gray
        else:
            comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, num_components))

        y_range = (y_grid[0], y_grid[-1])

        # Create colormap for ax[0] and ax[2]
        # cmap = plt.get_cmap('magma')
        cmap = plt.get_cmap('viridis')

        # ax[0] histogram
        ax[0].set_ylim(*y_range)
        ax[0].set_xlim(0, np.max(target_log_probs) * 1.1)
        color_ax0 = cmap(np.mean(marg_dens[:, 0]) / np.max(marg_dens))
        ax[0].hist(y_grid[:-1], weights=marg_dens[:, 0], range=y_range, bins=y_grid, color=light_gray,
                   orientation='horizontal')

        # ax[1] pcolormesh with proper alignment
        ax[1].set_ylim(*y_range)

        # Create proper coordinate grids for pcolormesh
        X, Y = np.meshgrid(np.arange(time_line + 1), y_grid)

        # Use pcolormesh with the correct dimensions
        pcm = ax[1].pcolormesh(X, Y, marg_dens, cmap=cmap, shading='flat', edgecolor='face')

        # add trajectories
        for j, component in enumerate(trajectories):
            cols = comp_colors[j]
            for j in range(component.shape[0]):  # num_trajecories
                ax[1].plot(x_grid, component[j, :], color='w', linewidth=1.2)

        # ax[2] histogram
        ax[2].set_ylim(*y_range)
        if x_T is None:
            x_T = [marg_dens[:, -1]]
        y_all = np.broadcast_to(y_grid[:-1], (len(x_T), y_grid.size - 1)).T
        target_counts = np.vstack(x_T).T
        color_ax2 = cmap(np.mean(marg_dens[:, -1]) / np.max(marg_dens))
        ax[2].hist(y_all, weights=target_counts, range=y_range, bins=y_grid,
                   orientation='horizontal', color=light_gray)

        high_res_y = np.linspace(*x_range, 1000)
        self.plot_colored_line(ax[2], np.exp(self.log_prob(high_res_y.reshape((-1, 1))).flatten()), high_res_y,
                               cmap=pcm.get_cmap())
        self.plot_colored_line(ax[0], np.exp(prior_log_prob(high_res_y.reshape((-1, 1)), ).flatten()), high_res_y,
                               cmap=pcm.get_cmap())

        ax[2].set_xlim(0, np.max(target_log_probs) * 1.1)

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)

        plt.setp(ax[0].get_yticklabels(), visible=False)
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[2].get_xticklabels(), visible=False)
        plt.setp(ax[0].set_xticks([]))
        plt.setp(ax[1].set_xticks([]))
        plt.setp(ax[2].set_xticks([]))
        plt.setp(ax[0].set_yticks([]))
        plt.setp(ax[1].set_yticks([]))
        plt.setp(ax[2].set_yticks([]))

        return ax

    def forward_velocity_visualization_paper(self, ax, prior_samples, prior_log_prob, x_t_prior_to_target, show, wb,
                                             step):
        trajectory = jnp.concatenate((jnp.reshape(prior_samples, (-1, 1, 1)), jnp.expand_dims(x_t_prior_to_target, -1)),
                                     axis=1)
        trajectory = jnp.expand_dims(trajectory, axis=0)

        # diffusion traj
        x_range = (-7, 7)
        resolution = 100
        num_trajectories = 5

        # model_sample_list ist array of (component, n_batch, n_time, dim)
        model_samples = np.vstack(trajectory)
        # model_samples is (n_batch, n_time, dim)

        time_line = model_samples.shape[1]
        model_samples = model_samples.squeeze()  # get rid of target dim (1)

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        x_grid = np.arange(time_line)

        target_log_probs = np.exp(self.log_prob(y_grid.reshape((-1, 1))).flatten())

        # first count frequencies then plot
        marg_dens = np.zeros((resolution, time_line))

        for t in range(time_line):
            p_t, _ = np.histogram(model_samples[:, t], bins=y_grid, density=True)
            marg_dens[:, t] = p_t

        trajectories = [comp[np.random.choice(comp.shape[0], num_trajectories, replace=False), :] for comp in
                        trajectory]
        # at last step we want to distinguish contributions of different components again
        x_T = [np.histogram(comp[:, -1], bins=y_grid, density=True)[0] for comp in trajectory]

        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        light_gray = np.array([[90 / 255, 90 / 255, 90 / 255, .3]])
        num_components = len(trajectories)
        if num_components == 1:
            comp_colors = dark_gray
        else:
            comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, num_components))

        y_range = (y_grid[0], y_grid[-1])

        # Create colormap for ax[0] and ax[2]
        cmap = plt.get_cmap('magma')
        # cmap = plt.get_cmap('viridis')

        # ax[0] histogram
        ax[0].set_ylim(*y_range)
        ax[0].set_xlim(0, np.max(target_log_probs) * 1.1)
        color_ax0 = cmap(np.mean(marg_dens[:, 0]) / np.max(marg_dens))
        ax[0].hist(y_grid[:-1], weights=marg_dens[:, 0], range=y_range, bins=y_grid, color=light_gray,
                   orientation='horizontal')

        # ax[1] pcolormesh with proper alignment
        ax[1].set_ylim(*y_range)

        # Create proper coordinate grids for pcolormesh
        X, Y = np.meshgrid(np.arange(time_line + 1), y_grid)

        # Use pcolormesh with the correct dimensions
        pcm = ax[1].pcolormesh(X, Y, marg_dens, cmap=cmap, shading='flat', edgecolor='face')

        # add trajectories
        for j, component in enumerate(trajectories):
            cols = comp_colors[j]
            for j in range(component.shape[0]):  # num_trajecories
                ax[1].plot(x_grid, component[j, :], color='w', linewidth=1.2)

        # ax[2] histogram
        ax[2].set_ylim(*y_range)
        if x_T is None:
            x_T = [marg_dens[:, -1]]
        y_all = np.broadcast_to(y_grid[:-1], (len(x_T), y_grid.size - 1)).T
        target_counts = np.vstack(x_T).T
        color_ax2 = cmap(np.mean(marg_dens[:, -1]) / np.max(marg_dens))
        ax[2].hist(y_all, weights=target_counts, range=y_range, bins=y_grid,
                   orientation='horizontal', color=light_gray)

        high_res_y = np.linspace(*x_range, 1000)
        self.plot_colored_line(ax[2], np.exp(prior_log_prob(high_res_y.reshape((-1, 1))).flatten()), high_res_y,
                               cmap=pcm.get_cmap())
        self.plot_colored_line(ax[0], np.exp(prior_log_prob(high_res_y.reshape((-1, 1)), ).flatten()), high_res_y,
                               cmap=pcm.get_cmap())

        ax[2].set_xlim(0, np.max(target_log_probs) * 1.1)

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)

        plt.setp(ax[0].get_yticklabels(), visible=False)
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[2].get_xticklabels(), visible=False)
        plt.setp(ax[0].set_xticks([]))
        plt.setp(ax[1].set_xticks([]))
        plt.setp(ax[2].set_xticks([]))
        plt.setp(ax[0].set_yticks([]))
        plt.setp(ax[1].set_yticks([]))
        plt.setp(ax[2].set_yticks([]))

        return ax

    def visualize_pos_vel(self, prior_samples, prior_log_prob, x_t_prior_to_target, vel_t, show, wb, step):
        trajectory = jnp.concatenate((jnp.reshape(prior_samples, (-1, 1, 1)), jnp.expand_dims(x_t_prior_to_target, -1)),
                                     axis=1)
        trajectory = jnp.expand_dims(trajectory, axis=0)
        fig, ax = plt.subplots(2, 3, figsize=(6, 8), gridspec_kw={'width_ratios': [1, 4, 1]})
        # fig.suptitle(f'Diffusion Plot Iteration: ' + str(step))

        self.forward_position_visualization(ax[0], prior_samples, prior_log_prob, x_t_prior_to_target, show, wb, step)
        self.forward_velocity_visualization(ax[1], vel_t[:, 0], prior_log_prob, vel_t[:, 1:], show, wb, step)

        wb["figures/diffusion_vis"] = [wandb.Image(fig)]

        path = project_path() + '/figures/diff_plot'
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.pdf'), bbox_inches='tight', pad_inches=0.1, dpi=300)

        if show:
            fig.show()
        else:
            fig.close()

        return wb

    def paper_vis(self, prior_samples, prior_log_prob, x_t_prior_to_target, vel_t, show, wb, step):
        trajectory = jnp.concatenate((jnp.reshape(prior_samples, (-1, 1, 1)), jnp.expand_dims(x_t_prior_to_target, -1)),
                                     axis=1)
        trajectory = jnp.expand_dims(trajectory, axis=0)
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(6, 3),
                               gridspec_kw={
                                   'width_ratios': [1, 3, 1],
                                   'wspace': 0.,
                                   # 'hspace': 0.4
                               })  # fig.suptitle(f'Diffusion Plot Iteration: ' + str(step))

        self.forward_position_visualization_paper(ax, prior_samples, prior_log_prob, x_t_prior_to_target, show, wb,
                                                  step)
        # self.forward_velocity_visualization_paper(ax, vel_t[:, 0], prior_log_prob, vel_t[:, 1:], show, wb, step)

        wb["figures/diffusion_vis"] = [wandb.Image(fig)]

        path = project_path() + '/figures/diff_plot'
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.pdf'), bbox_inches='tight', pad_inches=0.1, dpi=300)

        if show:
            fig.show()
        else:
            fig.close()

        return wb

    def forward_diffusion_visualization_presentation(self, prior_samples, x_t_prior_to_target, show, wb, step,
                                                     prior_sampled_components, component_wise=False):

        trajectory = jnp.concatenate((jnp.reshape(prior_samples, (-1, 1, 1)), x_t_prior_to_target), axis=1)
        weights = jnp.bincount(prior_sampled_components) / prior_sampled_components.shape[0]
        if not component_wise:
            trajectory = jnp.expand_dims(trajectory, axis=0)
        else:
            unique_components = np.unique(prior_sampled_components)
            trajectory = [trajectory[prior_sampled_components == comp] for comp in unique_components]

        # diffusion traj
        x_range = (-5, 5)
        resolution = 100
        num_trajectories_per_comp = 5

        # model_sample_list ist array of (component, n_batch, n_time, dim)
        model_samples = np.vstack(trajectory)
        # model_samples is (n_batch, n_time, dim)

        time_line = model_samples.shape[1]
        model_samples = model_samples.squeeze()  # get rid of target dim (1)

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        x_grid = np.arange(time_line)

        target_log_probs = np.exp(self.log_prob(y_grid.reshape((-1, 1))).flatten())

        # first count frequencies then plot
        marg_dens = np.zeros((resolution, time_line))

        for t in range(time_line):
            p_t, _ = np.histogram(model_samples[:, t], bins=y_grid, density=True)
            marg_dens[:, t] = p_t

        trajectories = [comp[np.random.choice(comp.shape[0], num_trajectories_per_comp), :] for comp in trajectory]
        # at last step we want to distinguish contributions of different components again
        x_T = [np.histogram(comp[:, -1], bins=y_grid, density=True)[0] for comp in trajectory]

        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        dark_blue = np.array([[0.12156863, 0.46666667, 0.70588235, 1.]])
        light_blue = np.array([[0.09019608, 0.74509804, 0.81176471, 1.]])
        num_components = len(trajectories)
        if num_components == 1:
            comp_colors = dark_gray
            comp_colors = light_blue
        else:
            comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, num_components))

        y_range = (y_grid[0], y_grid[-1])
        # plot
        fig, ax = plt.subplots(1, 3, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 4, 1]})
        fig.suptitle(f'Iteration: ' + str(step))
        x_0 = [np.histogram(comp[:, 0], bins=y_grid, density=True)[0] for comp in trajectory]
        y_0_all = np.broadcast_to(y_grid[:-1], (len(x_0), y_grid.size - 1)).T
        prior_counts = np.vstack(x_0).T * np.broadcast_to(weights, (y_grid.size - 1, len(x_0)))

        ax[0].set_ylim(*y_range)
        # ax[0].set_ylabel('$x$')
        ax[0].set_xlim(0, np.max(target_log_probs) * 1.1)
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)
        ax[0].axis('off')

        ax[0].set_title('Prior')
        ax[0].hist(y_0_all, weights=prior_counts, range=y_range, bins=y_grid, color=comp_colors,
                   orientation='horizontal', edgecolor='white', linewidth=0.75, histtype='bar', stacked=True)
        # ax[0].plot(np.exp(np.array(norm.logpdf(y_grid, 0.0, model_init_std))), y_grid,
        #            color='black', linewidth=0.75)

        ax[1].set_ylim(*y_range)
        # ax[1].set_xlabel('$t$')
        # grids must be one larger that dimensions of marg_dens
        ax[1].pcolormesh(np.insert(x_grid, -1, x_grid[-1] + 1), y_grid, marg_dens)
        # add trajectories
        # for j, component in enumerate(trajectories):
        #     cols = comp_colors[j]
        #     for i in range(component.shape[0]):  # num_trajecories
        #         ax[1].plot(x_grid, component[i, :], color=cols, linewidth=0.75)
        ax[1].xaxis.set_visible(False)
        ax[1].yaxis.set_visible(False)

        ax[2].set_ylim(*y_range)
        ax[2].set_title('Target')
        if x_T is None:
            x_T = [marg_dens[:, -1]]
        y_all = np.broadcast_to(y_grid[:-1], (len(x_T), y_grid.size - 1)).T
        target_counts = np.vstack(x_T).T * np.broadcast_to(weights, (y_grid.size - 1, len(x_T)))
        # target_counts = np.vstack(x_T).T
        # plot for each component target samples separately and stack bars
        ax[2].hist(y_all, weights=target_counts, range=y_range, bins=y_grid, color=comp_colors,
                   orientation='horizontal', edgecolor='white', linewidth=0.75, histtype='bar', stacked=True)
        ax[2].plot(target_log_probs, y_grid, color='black', linewidth=0.75)
        ax[2].set_xlim(0, np.max(target_log_probs) * 1.1)
        ax[2].xaxis.set_visible(False)
        ax[2].yaxis.set_visible(False)
        ax[2].axis('off')

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)

        wb["figures/diffusion_vis"] = [wandb.Image(fig)]

        path = project_path() + '/figures/diff_plot'
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def backward_visualization_var_explo(self, samples, step):
        x_range = (-7, 7)
        samples = samples.flatten()
        bw = 0.06 + 0.00045 * step
        # bw = 0.06
        kde = gaussian_kde(samples, bw_method=bw)

        x = np.linspace(x_range[0], x_range[1], 1000, )
        y = kde.evaluate(x)

        # Plot the KDE

        plt.plot(x, y)
        plt.fill_between(x, y, alpha=0.5)
        plt.title("Variance Exploding Diffusion: " + str(step))
        plt.ylim(0, 0.4)
        plt.xlim(x_range[0], x_range[1])
        plt.axis("off")

        path = './figures/var_explo'
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()
        plt.close()

    def backward_visualization(self, target_samples, x_t_target_to_prior, show, step):
        wb = {}

        trajectory = jnp.concatenate((jnp.reshape(target_samples, (-1, 1, 1)), x_t_target_to_prior), axis=1)
        trajectory = jnp.expand_dims(trajectory, 0)
        x_range = (-7, 7)
        resolution = 100
        num_trajectories = 10

        # model_sample_list ist array of (component, n_batch, n_time, dim)
        model_samples = np.vstack(trajectory)
        # model_samples is (n_batch, n_time, dim)

        time_line = model_samples.shape[1]
        model_samples = model_samples.squeeze()  # get rid of target dim (1)

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        x_grid = np.arange(time_line)

        # first count frequencies then plot
        marg_dens = np.zeros((resolution, time_line))

        for t in range(time_line):
            p_t, _ = np.histogram(model_samples[:, t], bins=y_grid, density=True)
            marg_dens[:, t] = p_t

        # select randomly num_trajectories from each component
        trajectories = [comp[np.random.choice(comp.shape[0], num_trajectories, replace=False), :] for comp in
                        trajectory]
        # at last step we want to distinguish contributions of different components again
        trajectory = [np.histogram(comp[:, -1], bins=y_grid, density=True)[0] for comp in trajectory]

        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        num_components = len(trajectories)
        if num_components == 1:
            comp_colors = dark_gray
        else:
            comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, num_components))

        y_range = (y_grid[0], y_grid[-1])
        # plot
        fig, ax = plt.subplots(1, 3, figsize=(6, 4), gridspec_kw={'width_ratios': [1, 4, 1]})
        fig.suptitle(f'Diffusion Plot Iteration: ' + str(step))
        ax[0].set_ylim(*y_range)

        ax[0].set_ylabel('$x$')
        ax[0].set_title('$\pi(x_T)$')
        ax[0].hist(y_grid[:-1], weights=marg_dens[:, 0], range=y_range, bins=y_grid, color=dark_gray[0],
                   orientation='horizontal', edgecolor='white', linewidth=0.75)
        # ax[0].plot(np.exp(np.array(norm.logpdf(y_grid, 0.0, model_init_std))), y_grid,
        #            color='black', linewidth=0.75)
        log_probs = np.exp(self.log_prob(y_grid.reshape((-1, 1))).flatten())
        ax[0].plot(log_probs, y_grid, color='black', linewidth=0.75)
        ax[0].set_xlim(0, np.max(log_probs) * 1.1)

        ax[1].set_ylim(*y_range)
        ax[1].set_title('$p(x_t)$')
        ax[1].set_xlabel('$t$')
        # grids must be one larger that dimensions of marg_dens
        ax[1].pcolormesh(np.insert(x_grid, -1, x_grid[-1] + 1), y_grid, marg_dens)
        # add trajectories
        for j, component in enumerate(trajectories):
            # disturb color by uniform noise to give each trajectory unique color
            cols = comp_colors[j]
            for j in range(component.shape[0]):  # num_trajecories
                ax[1].plot(x_grid, component[j, :], color=cols, linewidth=0.75)

        ax[2].set_ylim(*y_range)
        ax[2].set_title('$p(x_0)$')
        if trajectory is None:
            trajectory = [marg_dens[:, -1]]
        y_all = np.broadcast_to(y_grid[:-1], (len(trajectory), y_grid.size - 1)).T
        target_counts = np.vstack(trajectory).T
        # plot for each component target samples separately and stack bars
        ax[2].hist(y_all, weights=target_counts, range=y_range, bins=y_grid, color=comp_colors,
                   orientation='horizontal', edgecolor='white', linewidth=0.75, histtype='bar', stacked=True)
        ax[2].set_xlim(0, np.max(log_probs) * 1.1)

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)
        wb["figures/diffusion_vis"] = [wandb.Image(fig)]

        path = './figures/diff_plot'
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def backward_visualization_presentation(self, target_samples, x_t_target_to_prior, show, step,
                                            target_samples_indices, component_wise=False):
        wb = {}

        trajectory = jnp.concatenate((jnp.reshape(target_samples, (-1, 1, 1)), x_t_target_to_prior), axis=1)

        weights = jnp.bincount(target_samples_indices) / target_samples_indices.shape[0]
        if not component_wise:
            trajectory = jnp.expand_dims(trajectory, axis=0)
        else:
            unique_components = np.unique(target_samples_indices)
            trajectory = [trajectory[target_samples_indices == comp] for comp in unique_components]

        x_range = (-5, 5)
        resolution = 100
        num_trajectories = 10

        # model_sample_list ist array of (component, n_batch, n_time, dim)
        model_samples = np.vstack(trajectory)
        # model_samples is (n_batch, n_time, dim)

        time_line = model_samples.shape[1]
        model_samples = model_samples.squeeze()  # get rid of target dim (1)

        y_grid = np.linspace(x_range[0], x_range[1], resolution + 1)
        x_grid = np.arange(time_line)

        # first count frequencies then plot
        marg_dens = np.zeros((resolution, time_line))

        for t in range(time_line):
            p_t, _ = np.histogram(model_samples[:, t], bins=y_grid, density=True)
            marg_dens[:, t] = p_t

        # select randomly num_trajectories from each component
        trajectories = [comp[np.random.choice(comp.shape[0], num_trajectories, replace=False), :] for comp in
                        trajectory]
        # at last step we want to distinguish contributions of different components again
        x_0 = [np.histogram(comp[:, -1], bins=y_grid, density=True)[0] for comp in trajectory]

        dark_gray = np.array([[90 / 255, 90 / 255, 90 / 255, 1.0]])
        num_components = len(trajectories)
        if num_components == 1:
            comp_colors = dark_gray
        else:
            comp_colors = matplotlib.colormaps['tab10'](np.linspace(1.0, 0.0, num_components))

        y_range = (y_grid[0], y_grid[-1])
        # plot
        fig, ax = plt.subplots(1, 3, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 4, 1]})
        # fig.suptitle("Backward Process")
        ax[0].set_ylim(*y_range)

        ax[0].set_ylabel('$x$')
        # ax[0].set_title('$\pi(x_T)$')
        x_T = [np.histogram(comp[:, 0], bins=y_grid, density=True)[0] for comp in trajectory]
        y_0_all = np.broadcast_to(y_grid[:-1], (len(x_T), y_grid.size - 1)).T
        prior_counts = np.vstack(x_T).T * np.broadcast_to(weights, (y_grid.size - 1, len(x_T)))
        ax[0].hist(y_0_all, weights=prior_counts, range=y_range, bins=y_grid, color=comp_colors,
                   orientation='horizontal', edgecolor='white', linewidth=0.75, histtype='bar', stacked=True)
        # ax[0].plot(np.exp(np.array(norm.logpdf(y_grid, 0.0, model_init_std))), y_grid,
        #            color='black', linewidth=0.75)
        log_probs = np.exp(self.log_prob(y_grid.reshape((-1, 1))).flatten())
        ax[0].plot(log_probs, y_grid, color='black', linewidth=0.75)
        ax[0].set_xlim(0, np.max(log_probs) * 1.1)
        ax[0].axis('off')

        ax[1].set_ylim(*y_range)
        # ax[1].set_title('$p(x_t)$')
        ax[1].set_xlabel('$t$')
        # grids must be one larger that dimensions of marg_dens
        ax[1].pcolormesh(np.insert(x_grid, -1, x_grid[-1] + 1), y_grid, marg_dens)
        # add trajectories
        for j, component in enumerate(trajectories):
            # disturb color by uniform noise to give each trajectory unique color
            cols = comp_colors[j]
            for j in range(component.shape[0]):  # num_trajecories
                ax[1].plot(x_grid, component[j, :], color=cols, linewidth=0.75)
        ax[1].axis('off')

        ax[2].set_ylim(*y_range)
        # ax[2].set_title('$p(x_0)$')
        if x_0 is None:
            x_0 = [marg_dens[:, -1]]
        y_all = np.broadcast_to(y_grid[:-1], (len(x_0), y_grid.size - 1)).T
        target_counts = np.vstack(x_0).T * np.broadcast_to(weights, (y_grid.size - 1, len(x_0)))
        # plot for each component target samples separately and stack bars
        ax[2].hist(y_all, weights=target_counts, range=y_range, bins=y_grid, color=comp_colors,
                   orientation='horizontal', edgecolor='white', linewidth=0.75, histtype='bar', stacked=True)
        ax[2].set_xlim(0, np.max(log_probs) * 1.1)
        ax[2].axis('off')

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)
        wb["figures/diffusion_vis"] = [wandb.Image(fig)]

        path = './figures/diff_plot'
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_diff_traj'
        plt.savefig(os.path.join(path, name + '.png'), bbox_inches='tight', pad_inches=0.1, dpi=300)

        if show:
            plt.show()
        else:
            plt.close()

        return wb

    def visualise(self, x_T=None, x_0=None, x_t_prior_to_target=None, x_t_target_to_prior=None, vel_t=None,
                  prior_log_prob=None, show: bool = False, suffix: str = '', x_0_components=None,
                  ground_truth_target_samples=None, params=None) -> None:

        wb = {}

        wb = self.paper_vis(x_0, prior_log_prob, x_t_prior_to_target, vel_t, show, wb, suffix)

        """
        if vel_t is not None:
            wb = self.visualize_pos_vel(x_0, prior_log_prob, x_t_prior_to_target, vel_t, show, wb, suffix)
            return wb
        if x_t_target_to_prior is not None and x_t_prior_to_target.shape[1] != 0:
            self.backward_visualization(ground_truth_target_samples, x_t_target_to_prior, show, suffix)
            # self.backward_visualization_presentation(x_T, x_t_target_to_prior, show, suffix)
        # wb = self.simple_forward_visualization(x_T, show, wb, suffix)
        if x_t_prior_to_target is not None and x_t_prior_to_target.shape[1] != 0:
            wb = self.forward_diffusion_visualization(x_0, prior_log_prob, x_t_prior_to_target, show, wb, suffix, params=params)
       """

        # wb = self.simple_forward_visualization_presentation(x_T, show, wb, suffix, x_0_components)
        # wb = self.forward_diffusion_visualization_presentation(x_0, x_t_prior_to_target, show, wb, suffix, x_0_components, component_wise=True)

        return wb


if __name__ == '__main__':
    gmm = GMM1D()
    # one component, 40 bathc, 60 time
    samples = gmm.sample(jax.random.PRNGKey(0), (1, 400, 6))
    gmm.log_prob(samples)
    gmm.entropy(samples)
    # gmm.visualise( show=True)
    gmm.visualise(samples, show=True)
