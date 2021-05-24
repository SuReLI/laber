from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3_laber.policies import TD3Policy


class TD3_LABER(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 400,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = -1,
        gradient_steps: int = -1,
        n_episodes_rollout: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(TD3_LABER, self).__init__(
            policy,
            env,
            TD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TD3_LABER, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 400) -> None:

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values_big_batch = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                states_big_batch = replay_data.observations
                actions_big_batch = replay_data.actions
                
                # We assume we sampled uniformly a large batch of size 4*B
                bs = int(batch_size / 4)

                # Select uniformly a subset for the actor
                indices_actor = np.random.randint(batch_size, size=bs)
                states_selected_for_actor = states_big_batch[indices_actor]
                
                # Compute TD errors for each critic
                current_q_values_big_batch = self.critic(states_big_batch, actions_big_batch)
                td_errors1 = F.l1_loss(current_q_values_big_batch[0], target_q_values_big_batch, reduction='none')
                td_errors2 = F.l1_loss(current_q_values_big_batch[1], target_q_values_big_batch, reduction='none')
                
                # LaBER for critic 1
                td_errors1 = td_errors1.squeeze().cpu().numpy()
                probs1 = td_errors1/td_errors1.sum()
                indices1 = np.random.choice(np.arange(batch_size), bs, p=probs1)
                td_errors_for_selected_indices1 = td_errors1[indices1]
                states_selected1 = states_big_batch[indices1]
                actions_selected1 = actions_big_batch[indices1]
                target_q_values1 = target_q_values_big_batch[indices1]
                loss_weights1 = (1.0 / td_errors_for_selected_indices1) * td_errors1.mean()
                loss_weights1 = th.from_numpy(loss_weights1).unsqueeze(1)
                
                # LaBER for critic 2
                td_errors2 = td_errors2.squeeze().cpu().numpy()
                probs2 = td_errors2/td_errors2.sum()
                indices2 = np.random.choice(np.arange(batch_size), bs, p=probs2)
                td_errors_for_selected_indices2 = td_errors2[indices2]
                states_selected2 = states_big_batch[indices2]
                actions_selected2 = actions_big_batch[indices2]
                target_q_values2 = target_q_values_big_batch[indices2]
                loss_weights2 = (1.0 / td_errors_for_selected_indices2) * td_errors2.mean()
                loss_weights2 = th.from_numpy(loss_weights2).unsqueeze(1)


            # Get current Q-values estimates for each critic network
            current_q_values1 = self.critic(states_selected1, actions_selected1)
            current_q_values2 = self.critic(states_selected2, actions_selected2)
            loss1 = F.mse_loss(current_q_values1[0], target_q_values1, reduction='none').cpu() * loss_weights1
            loss2 = F.mse_loss(current_q_values2[1], target_q_values2, reduction='none').cpu() * loss_weights2
            loss1 = loss1.mean()
            loss2 = loss2.mean()

            # Compute critic loss
            critic_loss = loss1.mean() + loss2.mean()
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            loss1.backward()
            loss2.backward()
            self.critic.optimizer.step()
            #self.critic.optimizer.zero_grad()
            #loss2.backward()
            #self.critic.optimizer.step()

            # Delayed policy updates
            if gradient_step % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(states_selected_for_actor, self.actor(states_selected_for_actor)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TD3_LABER",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(TD3_LABER, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(TD3_LABER, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
