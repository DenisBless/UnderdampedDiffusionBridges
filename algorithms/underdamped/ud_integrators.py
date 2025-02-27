import jax
import jax.numpy as jnp

from algorithms.common.utils import sample_kernel, log_prob_kernel, check_stop_grad


def get_integrator(cfg, diffusion_model):
    def integrator(model_state, params, stop_grad=False, eval=False):
        mass_std = diffusion_model.mass_fn(params)

        def integrate_EM(state, per_step_input):
            x, vel, log_w, key_gen = state
            step = per_step_input

            step = step.astype(jnp.float32)

            # Compute SDE components
            dt = diffusion_model.delta_t_fn(step, params)
            friction = diffusion_model.friction_fn(step, params)
            eta = dt * friction
            scale = jnp.sqrt(2 * eta) * mass_std

            drift, aux = diffusion_model.drift_fn(step, x, params)

            # Forward kernel
            fwd_mean = vel * (1 - eta) + 2 * eta * diffusion_model.forward_model(step, x, vel, model_state, params,
                                                                                 aux) + dt * drift
            key, key_gen = jax.random.split(key_gen)
            vel_new = sample_kernel(key, jax.lax.stop_gradient(fwd_mean) if stop_grad else fwd_mean, scale)

            # Leapfrog integration
            x_new = x + dt * vel_new / (mass_std ** 2)
            drift_new, aux_new = diffusion_model.drift_fn(step + 1, x_new, params)

            # Backward kernel
            bwd_vel_mean = vel_new * (1 - eta) + 2 * eta * diffusion_model.backward_model(step + 1, x_new, vel_new,
                                                                                          model_state,
                                                                                          params,
                                                                                          aux_new) + dt * drift_new

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(vel_new, fwd_mean, scale)
            bwd_log_prob = log_prob_kernel(vel, bwd_vel_mean, scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (x_new, vel_new, log_w, key_gen)

            if eval:
                per_step_output = (x_new, vel_new)
            else:
                per_step_output = (None, None)

            return next_state, per_step_output

        def integrate_OBAB(state, per_step_input):
            x, vel, log_w, key_gen = state
            step = per_step_input

            step = step.astype(jnp.float32)

            # Compute SDE components
            dt = diffusion_model.delta_t_fn(step, params)
            friction = diffusion_model.friction_fn(step, params)
            eta = dt * friction
            scale = jnp.sqrt(2 * eta) * mass_std

            drift, aux = diffusion_model.drift_fn(step, x, params)

            # Forward kernel
            fwd_mean = vel * (1 - eta) + 2 * eta * diffusion_model.forward_model(step, x, vel, model_state, params, aux)
            key, key_gen = jax.random.split(key_gen)
            vel_prime = check_stop_grad(sample_kernel(key,  fwd_mean, scale), stop_grad)

            # Leapfrog integration
            vel_prime_prime = vel_prime + 0.5 * dt * drift
            x_new = check_stop_grad(x + dt * vel_prime_prime / (mass_std ** 2), stop_grad)
            drift_new, aux_new = diffusion_model.drift_fn(step, x_new, params)
            vel_new = check_stop_grad(vel_prime_prime + 0.5 * dt * drift_new, stop_grad)

            # Backward kernel
            bwd_vel_mean = vel_prime * (1 - eta) + 2 * eta * diffusion_model.backward_model(step, x, vel_prime,
                                                                                            model_state,
                                                                                            params, aux_new)

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(vel_prime, fwd_mean, scale)
            bwd_log_prob = log_prob_kernel(vel, bwd_vel_mean, scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (x_new, vel_new, log_w, key_gen)

            if eval:
                per_step_output = (x_new, vel_new)
            else:
                per_step_output = (None, None)

            return next_state, per_step_output

        def integrate_OBABO(state, per_step_input):
            x, vel, log_w, key_gen = state
            step = per_step_input

            step = step.astype(jnp.float32)

            # Compute SDE components
            dt = diffusion_model.delta_t_fn(step, params)
            friction = diffusion_model.friction_fn(step, params)
            eta = 0.5 * dt * friction  # Using * 0.5 for half steps
            scale = jnp.sqrt(2 * eta) * mass_std

            drift, aux = diffusion_model.drift_fn(step, x, params)

            # O
            fwd_mean = vel * (1 - eta) + 2 * eta * diffusion_model.forward_model(step, x, vel, model_state, params, aux)
            key, key_gen = jax.random.split(key_gen)
            vel_prime = check_stop_grad(sample_kernel(key,  fwd_mean, scale), stop_grad)
            # BAB - Leapfrog integration
            vel_prime_prime = check_stop_grad(vel_prime + 0.5 * dt * drift, stop_grad)
            x_new = check_stop_grad(x + dt * vel_prime_prime / (mass_std ** 2), stop_grad)
            drift_new, aux_new = diffusion_model.drift_fn(step, x_new, params)
            vel_prime_prime_prime = check_stop_grad(vel_prime_prime + 0.5 * dt * drift_new, stop_grad)

            # O
            fwd_mean_two = vel_prime_prime_prime * (1 - eta) + 2 * eta * diffusion_model.forward_model(step + 0.5,
                                                                                                       x_new,
                                                                                                       vel_prime_prime_prime,
                                                                                                       model_state,
                                                                                                       params,
                                                                                                       aux)
            key, key_gen = jax.random.split(key_gen)
            vel_new = check_stop_grad(sample_kernel(key, fwd_mean_two, scale), stop_grad)

            # Backward kernel
            bwd_mean_two = vel_new * (1 - eta) + 2 * eta * diffusion_model.backward_model(step + 1, x_new, vel_new,
                                                                                          model_state,
                                                                                          params, aux_new)
            bwd_mean = vel_prime * (1 - eta) + 2 * eta * diffusion_model.backward_model(step + 0.5, x, vel_prime,
                                                                                        model_state,
                                                                                        params, aux_new)

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(vel_prime, fwd_mean, scale) + log_prob_kernel(vel_new, fwd_mean_two, scale)
            bwd_log_prob = log_prob_kernel(vel_prime_prime_prime, bwd_mean_two, scale) + log_prob_kernel(vel, bwd_mean,
                                                                                                         scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (x_new, vel_new, log_w, key_gen)

            if eval:
                per_step_output = (x_new, vel_new)
            else:
                per_step_output = (None, None)

            return next_state, per_step_output

        def integrate_BAOAB(state, per_step_input):
            x, vel, log_w, key_gen = state
            step = per_step_input

            step = step.astype(jnp.float32)

            # Compute SDE components
            dt = diffusion_model.delta_t_fn(step, params)
            friction = diffusion_model.friction_fn(step, params)
            eta = dt * friction  # Using * 0.5 for half steps
            scale = jnp.sqrt(2 * eta) * mass_std

            # B
            drift, aux = diffusion_model.drift_fn(step, x, params)
            vel_prime = vel + 0.5 * dt * drift

            # A
            x_prime = x + 0.5 * dt * vel_prime / (mass_std ** 2)

            # Forward kernel
            fwd_mean = vel_prime * (1 - eta) + 2 * eta * diffusion_model.forward_model(step, x_prime, vel_prime,
                                                                                       model_state, params, aux)

            # O
            key, key_gen = jax.random.split(key_gen)
            vel_prime_prime = sample_kernel(key, jax.lax.stop_gradient(fwd_mean) if stop_grad else fwd_mean, scale)

            # A
            x_new = x_prime + 0.5 * dt * vel_prime_prime / (mass_std ** 2)

            # B
            drift_new, aux_new = diffusion_model.drift_fn(step, x_new, params)
            vel_new = vel_prime_prime + 0.5 * dt * drift_new

            # Backward kernel
            bwd_mean = vel_prime_prime * (1 - eta) + 2 * eta * diffusion_model.backward_model(step + 1, x_prime,
                                                                                              vel_prime_prime,
                                                                                              model_state,
                                                                                              params, aux_new)

            # Evaluate kernels
            fwd_log_prob = log_prob_kernel(vel_prime_prime, fwd_mean, scale)
            bwd_log_prob = log_prob_kernel(vel_prime, bwd_mean, scale)

            # Update weight and return
            log_w += bwd_log_prob - fwd_log_prob

            key, key_gen = jax.random.split(key_gen)
            next_state = (x_new, vel_new, log_w, key_gen)

            if eval:
                per_step_output = (x_new, vel_new)
            else:
                per_step_output = (None, None)

            return next_state, per_step_output

        if cfg.algorithm.integrator == 'EM':
            integrate = integrate_EM
        elif cfg.algorithm.integrator == 'OBAB':
            integrate = integrate_OBAB
        elif cfg.algorithm.integrator == 'OBABO':
            integrate = integrate_OBABO
        elif cfg.algorithm.integrator == 'BAOAB':
            integrate = integrate_BAOAB
        else:
            raise ValueError(f'No integrator named {cfg.algorithm.integrator}.')

        return integrate

    return integrator
