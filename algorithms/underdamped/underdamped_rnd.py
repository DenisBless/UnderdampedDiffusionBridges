import jax
import jax.numpy as jnp

from algorithms.common.utils import log_prob_kernel


def per_sample_rnd(seed, model_state, params, integrator, diffusion_model, stop_grad=False, eval=False):
    mass_std = diffusion_model.mass_fn(params)

    key, key_gen = jax.random.split(seed)

    init_x = diffusion_model.prior_sampler(params, key, 1)
    key, key_gen = jax.random.split(key_gen)
    init_x = jnp.squeeze(init_x, 0)
    init_vel = mass_std * jax.random.normal(key, shape=init_x.shape)  # (dim,)
    if stop_grad:
        init_x = jax.lax.stop_gradient(init_x)
        init_vel = jax.lax.stop_gradient(init_vel)
    key, key_gen = jax.random.split(key_gen)
    aux = (init_x, init_vel, jnp.zeros(1), key)
    integrate = integrator(model_state, params, stop_grad, eval)
    aux, per_step_output = jax.lax.scan(integrate, aux, jnp.arange(0, diffusion_model.num_steps))
    final_x, final_vel, log_ratio, _ = aux
    sample_terminal_cost = diffusion_model.prior_log_prob(init_x, params) - diffusion_model.target_log_prob(final_x)
    momentum_terminal_cost = log_prob_kernel(init_vel, jnp.zeros(init_x.shape[0]), mass_std) \
                             - log_prob_kernel(final_vel, jnp.zeros(init_x.shape[0]), mass_std) # todo detach mass for LV

    terminal_cost = sample_terminal_cost + momentum_terminal_cost

    running_cost = -log_ratio

    if eval:
        x_t, vel_t = per_step_output
        vel_t = jax.lax.stop_gradient(jnp.concatenate((jnp.expand_dims(init_vel, 0), vel_t), 0))
        x_t = jax.lax.stop_gradient(jnp.concatenate([jnp.expand_dims(init_x, 0), x_t], 0))
    else:
        x_t, vel_t = None, None
    stochastic_costs = jnp.zeros_like(running_cost)
    return final_x, running_cost, stochastic_costs, terminal_cost.reshape(running_cost.shape), x_t, vel_t


def rnd(key, model_state, params, integrator, diffusion_model, batch_size, stop_grad=False, eval=False):
    keys = jax.random.split(key, num=batch_size)
    in_tuple = (keys, model_state, params, integrator, diffusion_model, stop_grad, eval)
    in_axes = (0, None, None, None, None, None, None)
    rnd_result = jax.vmap(per_sample_rnd, in_axes=in_axes)(*in_tuple)
    x_0, running_costs, stochastic_costs, terminal_costs, x_t, vel_t = rnd_result

    return x_0, running_costs, stochastic_costs, terminal_costs, x_t, vel_t


def neg_elbo(key, model_state, params, integrator, diffusion_model, batch_size):
    rnd_result = rnd(key, model_state, params, integrator, diffusion_model, batch_size, stop_grad=False, eval=False)
    samples, running_costs, stochastic_costs, terminal_costs, x_t, vel_t = rnd_result
    neg_elbo = running_costs + terminal_costs
    return jnp.mean(neg_elbo)


def log_var(key, model_state, params, integrator, diffusion_model, batch_size):
    rnd_result = rnd(key, model_state, params, integrator, diffusion_model, batch_size, stop_grad=True, eval=False)
    samples, running_costs, stochastic_costs, terminal_costs, x_t, vel_t = rnd_result
    rnds = running_costs + terminal_costs + stochastic_costs
    return 0.5 * rnds.var()
