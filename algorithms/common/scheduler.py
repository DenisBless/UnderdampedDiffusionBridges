import jax.numpy as jnp


def get_linear_schedule(total_steps, min=0.01):
    def linear_noise_schedule(step):
        t = (total_steps - step) / total_steps
        return (1. - t) * min + t

    return linear_noise_schedule


def get_cosine_schedule(total_steps, min=0.01, s=0.008, pow=2):
    def cosine_schedule(step):
        t = (total_steps - step) / total_steps
        offset = 1 + s
        return (1. - min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** pow + min

    return cosine_schedule


def get_constant_schedule():
    def constant_schedule(step):
        return jnp.array(1.)

    return constant_schedule
