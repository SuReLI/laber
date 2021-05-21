# Large Batch Experience Replay

This repo's contents allow to reproduce the experiments of the "Large Batch Experience Replay" paper, under review at NeurIPS 2021.

# Citation

Omitted for anonymity during the review process.

# Repository structure

The paper's experiments built on pre-existing libraries (e.g. Dopamine, Stable-Baselines3) in order to compare our algorithms to fine-tuned agents. For example, the Atari experiments use Dopamine, which itself uses Tensorflow, while the MinAtar experiments use the authors original code which uses PyTorch. This choice was done to ensure a fair comparison between our and previous papers.

To keep things simple, the stand-alone code for reproducing all experiments from our paper is provided in three independent folders `Atari_experiments`, `MinAtar_experiments` and `PyBullet_experiments`.

For the sake of didactism, we also provide a `LaBER` folder containing the same agents without any external other dependency than PyTorch. This code has been written as an effort to provide simple agents using LaBER, after the paper's results were obtained. Thus, it has not been used to obtain the paper's results and comes without any guarantee of performance.

# Source code credit

The results on the MinAtar environements build on the code provided by the authors of [MinAtar](https://github.com/kenjyoung/MinAtar) (version 1.0.6), where we have included LaBER, PER, GER and their combinations, and on the [autograd hacks](https://github.com/cybertronai/autograd-hacks) for recovering per-sample gradients in PyTorch.

The results on Atari games build on the [Dopamine](https://github.com/google/dopamine) (version 3.1.7) implementation, where we have included LaBER, PER, GER and their combinations.

The results reported in the paper "Large Batch Experience Replay" have been produced thanks to 3 existing repositories: [MinAtar](https://github.com/kenjyoung/MinAtar), [Dopamine](https://github.com/google/dopamine), or [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The agents LaBER and GER have been implemented over the agents given by these repositories. 

# Reproducing the experiments

## Atari

Atari experiments follow the model of the Dopamine original code and are specified through a gin configuration file, located in the folder of the corresponding agent, e.g. `dopamine/agents/dqn_LABER/configs/dqn_LABER_pong.gin`.

```
cd Atari_experiments
python3 -um dopamine.discrete_domains.train \
  --base_dir=results/dqn_laber_pong/ 
  --gin_files=dopamine/agents/dqn_LABER/configs/dqn_LABER_pong.gin
```

Once the experiments are run, one can recreate the paper's graphs using:
TODO

## MinAtar

MinAtar experiments follow the model of the MinAtar original code. For example, one can run:

```
cd MinAtar_experiments
python3 agents/dqn.py -g breakout -v -a 0.0001
```

For details on the arguments passed to the RL agents, see each agent's documentation (e.g. [dqn_LABER.py](agents/dqn_LABER.py)).

Once the experiments are run, one can recreate the paper's graphs using:
TODO

## PyBullet

The continuous control experiments using PyBullet exploit our modification of the Stable-Baselines3 library. To reproduce the results, run:

```
cd PyBullet_experiments
python3 experiments/sac_laber_hopper.py
```

Once the experiments are run, one can recreate the paper's graphs using:
TODO
