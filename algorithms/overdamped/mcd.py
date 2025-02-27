import jax
import jax.numpy as jnp

from algorithms.common.init_diffusion_model import init_underdamped, init_langevin, init_model
from algorithms.common.diffusion_models import DiffusionModel
from algorithms.common.utils import init_dt
from utils.helper import inverse_softplus


def init_mcd(key, cfg, target):
    alg_cfg = cfg.algorithm
    dim = target.dim

    params = {'params': {'betas': jnp.ones((alg_cfg.num_steps,)),
                         'prior_mean': jnp.zeros((dim,)),
                         'prior_std': jnp.ones((dim,)) * inverse_softplus(alg_cfg.init_std),
                         'mass_std': jnp.ones(1) * inverse_softplus(1.),
                         'dt': init_dt(cfg),
                         'friction': jnp.ones(dim) * inverse_softplus(alg_cfg.friction) if alg_cfg.per_dim_friction else jnp.ones(1) * inverse_softplus(alg_cfg.friction),
                         }}

    prior_log_prob, prior_sampler, delta_t_fn, friction_fn, mass_fn = init_underdamped(cfg, dim)
    langevin_fn = init_langevin(cfg, prior_log_prob, target.log_prob)

    def forward_model(step, x, model_state, params, aux):
        return jnp.zeros_like(x)

    def backward_model(step, x, model_state, params, aux):
        langevin_vals = aux
        model_params = jax.lax.stop_gradient(params['params']['bwd_params']) if alg_cfg.use_path_gradient else params['params']['bwd_params']
        return model_state.apply_fn[1](model_params, x, step,
                                       jax.lax.stop_gradient(langevin_vals))

    def drift_fn(step, x, params):
        return langevin_fn(step, x, params)

    key, key_gen = jax.random.split(key)
    model_state = init_model(key, params, cfg, dim, learn_forward=False, learn_backward=True)

    return DiffusionModel(num_steps=alg_cfg.num_steps,
                          forward_model=forward_model,
                          backward_model=backward_model,
                          drift_fn=drift_fn,
                          delta_t_fn=delta_t_fn,
                          friction_fn=friction_fn,
                          mass_fn=mass_fn,
                          prior_sampler=prior_sampler,
                          prior_log_prob=prior_log_prob,
                          target_log_prob=target.log_prob,
                          ), model_state
