import numpy as np
import torch
import gym
import argparse
import os


import utils
import TD3
import PER_TD3
import GER_TD3
import LABER_TD3
import SAC
import PER_SAC
import GER_SAC
import LABER_SAC


# Runs policy for X episodes and returns average reward
def eval_policy(policy, env, seed, eval_episodes=5):
    eval_env = gym.make(env)
    eval_env.seed(seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), test=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="TD3")                       # Algorithm name
    parser.add_argument("--env", default="LunarLanderContinuous-v2")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                      # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=100, type=int)         # Time steps initial random policy is used
    parser.add_argument("--batch_size", default=100, type=int)              # Batch size for actor and critic
    parser.add_argument("--discount", default=0.99)                         # Discount factor
    parser.add_argument("--tau", default=0.005)                             # Target network update rate
    parser.add_argument("--eval_freq", default=1e4, type=int)               # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=3e6, type=int)           # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                        # Std of Gaussian exploration noise
    parser.add_argument("--policy_noise", default=0.2)                      # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)               # Frequency of delayed policy updates
    parser.add_argument("--m_factor", default=4, type=int)                  # Only used in LABER : multiplicative factor for the large batch
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.algorithm, args.env, str(args.seed))
    print("---------------------------------------")
    print(f"Settings: {file_name}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])


    kwargs = {
        "state_dim": state_dim, 
        "action_dim": action_dim, 
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq
    }


    # Initialize policy and replay buffer
    if args.algorithm == "SAC": 
        policy = SAC.SAC(**kwargs)
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    elif args.algorithm == "PER_SAC": 
        policy = PER_SAC.PER_SAC(**kwargs)
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim)

    elif args.algorithm == "GER_SAC": 
        policy = GER_SAC.GER_SAC(**kwargs)
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim)

    elif args.algorithm == "LABER_SAC": 
        policy = LABER_SAC.LABER_SAC(**kwargs)
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)


    kwargs["max_action"] = max_action
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action

    if args.algorithm == "TD3": 
        policy = TD3.TD3(**kwargs)
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    elif args.algorithm == "PER_TD3": 
        policy = PER_TD3.PER_TD3(**kwargs)
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim)

    elif args.algorithm == "GER_TD3": 
        policy = GER_TD3.GER_TD3(**kwargs)
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim)

    elif args.algorithm == "LABER_TD3": 
        policy = LABER_TD3.LABER_TD3(**kwargs)
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)] 

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            if args.algorithm == "TD3" or args.algorithm == "LABER_TD3" or args.algorithm == "PER_TD3" or args.algorithm == "GER_TD3":
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
            else:
                action = (policy.select_action(np.array(state))).clip(-max_action, max_action)
        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if args.algorithm == "LABER_SAC" or args.algorithm == "LABER_TD3" :
            if t >= int(args.m_factor*args.start_timesteps):
                policy.train(replay_buffer, args.batch_size, args.m_factor)
        else:
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save("./results/%s" % (file_name), evaluations)
