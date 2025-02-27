from functools import partial
from time import time

import jax
import wandb

from algorithms.common.eval_methods.eval_fn import get_eval_fn
from algorithms.underdamped.ud_integrators import get_integrator as get_integrator_ud
from algorithms.overdamped.od_integrators import get_integrator as get_integrator_od
from algorithms.underdamped.underdamped_rnd import neg_elbo as neg_elbo_ud
from algorithms.underdamped.underdamped_rnd import log_var as log_var_ud
from algorithms.underdamped.underdamped_rnd import rnd as rnd_ud

from algorithms.overdamped.overdamped_rnd import neg_elbo as neg_elbo_od
from algorithms.overdamped.overdamped_rnd import log_var as log_var_od
from algorithms.overdamped.overdamped_rnd import rnd as rnd_od

from algorithms.common.eval_methods.utils import extract_last_entry
from utils.alg_selector import get_init_fn
from utils.print_util import print_results


def learner(cfg, target):
    alg_cfg = cfg.algorithm
    key, key_gen = jax.random.split(jax.random.PRNGKey(cfg.seed))

    diffusion_model, model_state = get_init_fn(alg_cfg.name)(key, cfg, target)

    if alg_cfg.underdamped:
        rnd, neg_elbo, log_var = rnd_ud, neg_elbo_ud, log_var_ud
        integrator = get_integrator_ud(cfg, diffusion_model)
    else:
        rnd, neg_elbo, log_var = rnd_od, neg_elbo_od, log_var_od
        integrator = get_integrator_od(cfg, diffusion_model)

    if cfg.common.loss == 'elbo':
        loss_fn = neg_elbo
    elif cfg.common.loss == 'log_var':
        loss_fn = log_var
    else:
        raise ValueError(f'No loss function named {cfg.common.loss_fn}.')

    rnd_short = partial(rnd, integrator=integrator, diffusion_model=diffusion_model, batch_size=cfg.eval_samples, stop_grad=True, eval=True)
    loss_short = partial(loss_fn, integrator=integrator, diffusion_model=diffusion_model, batch_size=alg_cfg.batch_size)
    loss = jax.jit(jax.value_and_grad(loss_short, 2))

    target_samples = target.sample(jax.random.PRNGKey(0), (cfg.eval_samples,))
    eval_fn, logger = get_eval_fn(rnd_short, diffusion_model, target, target_samples, cfg)

    eval_freq = max(alg_cfg.iters // cfg.n_evals, 1)
    timer = 0
    for step in range(alg_cfg.iters):
        # Evaluation step
        if (step % eval_freq == 0) or (step == alg_cfg.iters - 1):
            key, key_gen = jax.random.split(key_gen)
            logger["stats/step"].append(step)
            logger["stats/wallclock"].append(timer)
            logger["stats/nfe"].append((step + 1) * alg_cfg.batch_size)
            logger.update(eval_fn(model_state, key))
            try:
                logger["loss/mean"].append(loss_value)
            except:
                pass

            print_results(step, logger, cfg)

            if cfg.use_wandb:
                wandb.log(extract_last_entry(logger))

        # Update step
        key, key_gen = jax.random.split(key_gen)
        iter_time = time()
        loss_value, grads = loss(key, model_state, model_state.params)
        model_state = model_state.apply_gradients(grads=grads)
        timer += time() - iter_time


