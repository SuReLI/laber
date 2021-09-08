# Large Batch Experience Replay

This repo's contents allow to reproduce the experiments of the "Large Batch Experience Replay" paper.

# Citation

Omitted for anonymity during the review process.

# Repository structure

The paper's experiments built on pre-existing libraries (Dopamine, Stable-Baselines3, and MinAtar) in order to compare our algorithms to fine-tuned agents. For example, the Atari experiments use Dopamine, which itself uses Tensorflow, while the MinAtar experiments use the authors original code which uses PyTorch. This choice was done to ensure a fair comparison between our and previous papers.

To keep things simple, the stand-alone code for reproducing all experiments from our paper is provided in three independent folders `Atari_experiments`, `MinAtar_experiments` and `PyBullet_experiments`.

For the sake of didactism, we also provide a `LaBER` folder containing the same agents without any external other dependency than PyTorch. This code has been written as an effort to provide simple agents using LaBER, after the paper's results were obtained. Thus, it has not been used to obtain the paper's results and comes without any guarantee of performance.

# Source code credit

The results reported in the paper "Large Batch Experience Replay" have been produced thanks to 3 existing repositories: [MinAtar](https://github.com/kenjyoung/MinAtar), [Dopamine](https://github.com/google/dopamine), or [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The agents LaBER, PER, and GER have been implemented over the agents given by these repositories. 

The results on the MinAtar environements build on the code provided by the authors of [MinAtar](https://github.com/kenjyoung/MinAtar) (version 1.0.6), where we have included LaBER, PER, GER and their combinations, and on the [autograd hacks](https://github.com/cybertronai/autograd-hacks) for recovering per-sample gradients in PyTorch.

The results on Atari games build on the [Dopamine](https://github.com/google/dopamine) (version 3.1.7) implementation, where we have included LaBER, PER, and GER.

The results on PyBullet environments build on the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (version 0.11.0a7) implementation, where we have included LaBER, PER, and GER.

# Reproducing the experiments

## Installations

For PyBullet, 
```
cd PyBullet_experiments
pip install pybullet
pip install -e stable-baselines3/
```

For Atari, 
```
cd Atari_experiments
pip install requirements.txt
```

For MinAtar, 
```
cd MinAtar_experiments
pip install .
```


## MinAtar

MinAtar experiments follow the model of the MinAtar original code. For example, one can run:

```
cd MinAtar_experiments
python3 agents/dqn_LABER.py -g breakout -o resultsbreakout/dqn_LABER -v -a 0.0001
```

For details on the arguments passed to the RL agents, see each agent's documentation (e.g. [dqn_LABER.py](agents/dqn_LABER.py)).

Once the experiments are run, one can recreate the paper's graphs using:
For Asterix: 
```
cd MinAtar_experiments
python3 agents/plot_return.py -f resultsasterix/dqn_LABER -w 1000 -s 200 -n (number of runs) 
```
For Breakout: 
```
cd MinAtar_experiments
python3 agents/plot_return.py -f resultsbreakout/dqn_LABER -w 3000 -s 200 -n (number of runs) 
```
For Freeway: 
```
cd MinAtar_experiments
python3 agents/plot_return.py -f resultsfreeway/dqn_LABER -w 30 -s 10 -n (number of runs) 
```
For Seaquest: 
```
cd MinAtar_experiments
python3 agents/plot_return.py -f resultsseaquest/dqn_LABER -w 1000 -s 200 -n (number of runs) 
```
For Space Invaders: 
```
cd MinAtar_experiments
python3 agents/plot_return.py -f resultsspaceinvaders/dqn_LABER -w 1000 -s 200 -n (number of runs) 
```

## Atari

Atari experiments follow the model of the Dopamine original code and are specified through a gin configuration file, located in the folder of the corresponding agent, e.g. `dopamine/agents/dqn_LABER/configs/dqn_LABER_pong.gin`.

```
cd Atari_experiments
python3 -um dopamine.discrete_domains.train --base_dir=results/dqn_laber_pong/ --gin_files=dopamine/agents/dqn_LABER/configs/dqn_LABER_pong.gin
```

Once the experiments are run, the results are stored in the folder results/. One can recreate the paper's graphs using:
```
cd Atari_experiments
python3 plot_return.py -w (directory of the tensorboard file) (directory to store the numpy array of results)
```

## PyBullet

The continuous control experiments using PyBullet exploit our modification of the Stable-Baselines3 library. To reproduce the results, run:

```
cd PyBullet_experiments
python3 experiments/laber_sac_hopper.py
```

Once the experiments are run, one can recreate the paper's graphs using:
```
cd PyBullet_experiments
python3 experiments/plot_return.py -w (directory of the tensorboard file) (directory to store the numpy array of results)
```

# Licenses

[Dopamine](https://github.com/google/dopamine) is distributed under the Apache V2.0 License ([included here](Atari_experiments/LICENSE (DOPAMINE))).   
[MinAtar](https://github.com/kenjyoung/MinAtar) is distributed under the GPL v3 license ([included here](MinAtar_experiments/License (MinAtar).txt)).   
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) is distributed under the MIT license ([included here](LICENSE (SB3))).

Unless in conflict with prior licenses, our work is distributed under the MIT license.

