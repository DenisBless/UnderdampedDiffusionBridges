import distrax
import jax
import jax.numpy as jnp

from algorithms.common.eval_methods.utils import moving_averages, save_samples, compute_reverse_ess
from algorithms.common.ipm_eval import discrepancies
from algorithms.common.utils import plot_annealing, plot_timesteps


def get_eval_fn(
        rnd,
        diffusion_model,
        target,
        target_samples,
        cfg):
    rnd = jax.jit(rnd)

    logger = {
        'params/dt': [],
        'params/friction': [],
        'params/mass_std': [],
        'KL/elbo': [],
        'logZ/delta_reverse': [],
        'logZ/reverse': [],
        'ESS/reverse': [],
        'discrepancies/sd': [],
        'other/target_log_prob': [],
        'other/EMC': [],
        "stats/step": [],
        "stats/wallclock": [],
        "stats/nfe": [],
        "loss/mean": [jnp.inf],
        "loss/std": [jnp.inf],
    }

    def short_eval(model_state, key):
        if isinstance(model_state, tuple):
            model_state1, model_state2 = model_state
            params = (model_state1.params, model_state2.params)
        else:
            params = (model_state.params,)
        samples, running_costs, stochastic_costs, terminal_costs, x_t, vel_t = rnd(key, model_state, *params)
        # print(f'Running costs: {running_costs.mean()}; Stochastic costs {stochastic_costs.mean()}; Terminal Costs {terminal_costs.mean()}')

        logger['params/friction'].append(jax.nn.softplus(model_state.params['params']['friction']))
        logger['params/mass_std'].append(jax.nn.softplus(model_state.params['params']['mass_std']))

        log_is_weights = -(running_costs + stochastic_costs + terminal_costs)
        ln_z = jax.scipy.special.logsumexp(log_is_weights) - jnp.log(cfg.eval_samples)
        elbo = -jnp.mean(running_costs + terminal_costs)

        if target.log_Z is not None:
            logger['logZ/delta_reverse'].append(jnp.abs(ln_z - target.log_Z))

        logger['logZ/reverse'].append(ln_z)
        logger['KL/elbo'].append(elbo)
        logger['ESS/reverse'].append(compute_reverse_ess(log_is_weights, cfg.eval_samples))
        logger['other/target_log_prob'].append(jnp.mean(target.log_prob(samples)))

        if cfg.target.name == 'gaussian_mixture1d':
            logger.update(target.visualise(x_0=x_t[:, 0],
                                           x_T=samples,
                                           vel_t=None if vel_t is None else jnp.squeeze(vel_t[:, 1:], -1),
                                           ground_truth_target_samples=target_samples,
                                           x_t_prior_to_target=jnp.squeeze(x_t[:, 1:], -1),
                                           x_t_target_to_prior=None,
                                           show=cfg.visualize_samples,
                                           suffix='',
                                           prior_log_prob=lambda x: diffusion_model.prior_log_prob(x, model_state.params),
                                           x_0_components=None,
                                           params=params,
                                           ))
        else:
            logger.update(target.visualise(samples=samples, show=cfg.visualize_samples))

        # logger.update(plot_annealing(model_state, cfg))
        # logger.update(plot_timesteps(diffusion_model, model_state, cfg))

        if cfg.compute_emc and cfg.target.has_entropy:
            logger['other/EMC'].append(target.entropy(samples))

        for d in cfg.discrepancies:
            logger[f'discrepancies/{d}'].append(getattr(discrepancies, f'compute_{d}')(target_samples, samples,
                                                                                       cfg) if target_samples is not None else jnp.inf)

        if cfg.moving_average.use_ma:
            logger.update(moving_averages(logger, window_size=cfg.moving_average.window_size))

        if cfg.save_samples:
            save_samples(cfg, logger, samples)

        return logger

    return short_eval, logger
