# Underdamped Diffusion Bridges with Applications to Sampling

This repository accompanies the paper "[Underdamped Diffusion Bridges with Applications to Sampling](https://openreview.net/forum?id=Q1QTxFm0Is)" [[`ICLR'25`](https://openreview.net/forum?id=Q1QTxFm0Is),[`BibTeX`](#references)].
## Available Algorithms
The table below provides a overview of all available algorithms.

| **Acronym**                                | **Method**                                  | **Reference**                                                   | ID     |
|--------------------------------------------|---------------------------------------------|-----------------------------------------------------------------|--------|
| [ULA](algorithms/overdamped/ula.py)        | Uncorrected Langevin Annealing              | [Thin et al., 2021](https://arxiv.org/abs/2106.15921)           | ula    |
| [ULA-UD](algorithms/underdamped/ula_ud.py) | Uncorrected Hamiltonian Annealing           | [Geffner et al., 2021](https://arxiv.org/abs/2107.04150)        | ula_ud |
| [MCD](algorithms/overdamped/mcd.py)        | Monte Carlo Diffusion                       | [Doucet et al., 2022](https://arxiv.org/abs/2208.07698)         | mcd    |
| [MCD-UD](algorithms/underdamped/mcd_ud.py) | Langevin Diffusion VI                       | [Geffner et al., 2022](https://arxiv.org/abs/2208.07743)        | mcd_ud |
| [DIS](algorithms/overdamped/dis.py)        | Time-Reversed Diffusion Sampler             | [Berner et al., 2022](https://openreview.net/pdf?id=oYIjw37pTP) | dis    |
| [DIS-UD](algorithms/underdamped/dis_ud.py) | Underdamped Time-Reversed Diffusion Sampler | Ours                                                            | dis_ud |
| [DBS](algorithms/overdamped/dbs.py)        | Diffusion Bridge Sampler                    | [Richter et al., 2023](https://arxiv.org/abs/2307.01198)        | dbs    |
| [DBS-UD](algorithms/underdamped/dbs_ud.py) | Underdamped Diffusion Bridge Sampler        | Ours                                                            | dbs_ud |
The respective configuration files can be found [here](configs/algorithm).

## Available Target Densities
The table below provides a overview of available target densities. The 'ID' column provides identifier for running experiments 
via comand line. Further details in the [Running Experiments](#running-experiments) section.

|                                         | dim  | True log Z | Target Samples | ID         |
|-----------------------------------------|------|------------|----------------|------------|
| [**Funnel**](targets/funnel.py)         | 10   | ✔️         | ✔️             | funnel     |
| [**Credit**](targets/german_credit.py)  | 25   | ❌          | ❌              | credit     |
| [**Seeds**](targets/seeds.py)           | 26   | ❌          | ❌              | seeds      |
| [**Cancer**](targets/breast_cancer.py)  | 31   | ❌          | ❌              | cancer     |
| [**Brownian**](targets/brownian.py)     | 32   | ❌          | ❌              | brownian   |
| [**Ionosphere**](targets/ionosphere.py) | 35   | ❌          | ❌              | ionosphere |
| [**ManyWell**](targets/many_well.py)    | 50   | ✔️         | ✔️             | many_well  |
| [**Sonar**](targets/sonar.py)           | 61   | ❌          | ❌              | sonar      |
| [**LGCP**](targets/lgcp.py)             | 1600 | ❌          | ❌              | lgcp       |


The respective configuration files can be found [here](configs/target).

## Installation

First, clone the repo. For installation we recommend using [Conda](https://conda.io/docs/user-guide/install/download.html) to set up the codebase:
  ```
  conda create -n underdamped_sampling python==3.10.14 pip --yes
  conda activate underdamped_sampling
  ```
Install the required packages using 
  ```
  pip install -r requirements.txt
  ```
Finally, we use [`wandb`](https://wandb.ai/) for experiment tracking. Login to your wandb account:
  ```
  wandb login
  ```
  You can also omit this step and add the `use_wandb=False` command line arg to your runs.


## Running Experiments

### Configuration
We use [`hydra`](https://hydra.cc/) for config management. The [base configuration](configs/base_conf.yaml) file sets 
parameters that are agnostic to the specific choice of algorithm and target density. The `wandb` entity can be set in the [setup config file](configs/setup.yaml).
### Running a single Experiment
In the simplest case, a single run can be started using
  ```
  python run.py algorithm=<algorithm ID> target=<target ID>
  ```
The algorithm ID is identical to the Acronym in the [algorithm table](#available-algorithms). The target ID can be found in the ID column of the [target table](#available-target-densities).
### Running multiple Experiments
Running multiple experiments can be done by using the hydra multi-run flag `-m/--multirun` flag.
For instance, running multiple seeds can be done via
  ```
  python run.py -m seed=0,1,2,3  algorithm=<algorithm ID> target=<target ID>
  ```
Using comma separation can also be used for running multiple algorithms/targets. 
### Running Experiments on a Cluster via Slurm
Running experiments on a cluster using [Slurm](https://slurm.schedmd.com/documentation.html) can be done via  
  ```
  python run.py +launcher=slurm algorithm=<algorithm ID> target=<target ID>
  ```
which uses the [slurm config](configs/launcher/slurm.yaml) config. Please make sure that you adapt the default settings to your slurm configuration.


## References

If you use parts of this codebase in your research, please cite us using the following BibTeX entries.

```
@inproceedings{
blessing2025underdamped,
title={Underdamped Diffusion Bridges with Applications to Sampling},
author={Denis Blessing and Julius Berner and Lorenz Richter and Gerhard Neumann},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=Q1QTxFm0Is}
}
```

