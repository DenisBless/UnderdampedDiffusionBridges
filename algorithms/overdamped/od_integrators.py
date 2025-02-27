import jax
import jax.numpy as jnp

from algorithms.common.utils import sample_kernel, log_prob_kernel, check_stop_grad


def get_integrator(cfg, diffusion_model):
    def integrator(model_state, params, stop_grad=False, eval=False):

        def integrate_EM(state, per_step_input):
            x, log_w, key_gen = state
            step = per_step_input

            step = step.astype(jnp.float32)

            # Compute SDE components
            dt = diffusion_model.delta_t_fn(step, params)
            sigma_square = 1. / diffusion_model.friction_fn(step, params)
            eta = dt * sigma_square
            eta = check_stop_grad(eta, stop_grad)
            scale = jnp.sqrt(2 * eta)
            scale = check_stop_grad(scale, stop_grad)

            # Forward kernel
            drift, aux = diffusion_model.drift_fn(step, x, params)
            ctrl = eta * diffusion_model.forward_model(step, x, model_state, params, aux)
            fwd_mean = eta * drift + ctrl
            key, key_gen = jax.random.split(key_gen)
            x_new = sample_kernel(key, x + fwd_mean, scale)

            # Backward kernel
            drift_new, aux_new = diffusion_model.drift_fn(step + 1, x_new, params)
            bwd_mean = eta * (drift_new + diffusion_model.backward_model(step + 1, x_new, model_state, params, aux_new))

            # Evaluate kernels
            delta_x = check_stop_grad(x_new - x, stop_grad)
            fwd_log_prob = log_prob_kernel(delta_x, fwd_mean, scale)
            bwd_log_prob = log_prob_kernel(-delta_x, bwd_mean, scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (check_stop_grad(x_new, stop_grad), log_w, key_gen)

            if eval:
                per_step_output = x_new
            else:
                per_step_output = None

            return next_state, per_step_output

        if cfg.algorithm.integrator == 'EM':
            integrate = integrate_EM
        else:
            raise ValueError(f'No integrator named {cfg.algorithm.integrator}.')

        return integrate

    return integrator
