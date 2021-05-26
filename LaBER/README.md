# Large Batch Experience Replay

PyTorch implementation of Large Batch Experience Replay (LaBER), Gradient Experience Replay (GER) and Prioritized Experience Replay (PER) over DDQN (for discrete domains), TD3 and SAC (for continuous domains). 

This implementation has been thought to be as didactic as possible, focusing on the specificities of the studied agents. Code snippets have been taken from [this](https://github.com/sfujim/LAP-PAL) repository. We mention that this implementation has not been tested: the results reported in the paper come from LaBER / PER/ GER agents directly implemented on established frameworks such as [MinAtar](https://github.com/kenjyoung/MinAtar), [Dopamine](https://github.com/google/dopamine), or [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). 

Extracting per-sample gradient norms in pytorch is not a straightforward task and GER needs these quantities. In this repository, we provide two different (and equivalent) ways of computing per-sample gradient norms (one for DDQN, one for TD3 and SAC). 

When implementing GER or PER on TD3/SAC, we have two TD errors since there are two critics. Instead of maintaining one list of priorities, as often proposed in other works, we work with two lists, two independent mini-batches, and compute back-propagation independently on each critic.  

