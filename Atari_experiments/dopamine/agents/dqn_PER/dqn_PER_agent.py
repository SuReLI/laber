from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import prioritized_replay_buffer
import tensorflow as tf

import gin.tf


@gin.configurable
class DQN_PER_Agent(dqn_agent.DQNAgent):
  """A compact implementation of a PER agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=atari_lib.NatureDQNNetwork,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=False,
               optimizer=tf.compat.v1.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.compat.v1.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expects four parameters:
        (num_actions, num_atoms, support, network_type).  This class is used to
        generate network instances that are used by the agent. Each
        instantiation would have different set of variables. See
        dopamine.discrete_domains.atari_lib.NatureDQNNetwork as an example.
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: `tf.compat.v1.train.Optimizer`, for training the value
        function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """

    self._replay_scheme = replay_scheme
    # TODO(b/110897128): Make agent optimizer attribute private.
    self.optimizer = optimizer

    dqn_agent.DQNAgent.__init__(
        self,
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)


  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A `WrappedPrioritizedReplayBuffer` object.

    Raises:
      ValueError: if given an invalid replay scheme.
    """
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)


  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    td_errors = tf.sqrt(tf.compat.v1.losses.mean_squared_error(target, replay_chosen_q, reduction=tf.losses.Reduction.NONE))
    td_errors = tf.stop_gradient(td_errors)
    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    
    if self._replay_scheme == 'prioritized':
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)

      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.sqrt(td_errors + 1e-10))

      # Weight the loss by the inverse priorities.
      loss = loss_weights * loss
    else:
      update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.compat.v1.variable_scope('Losses'):
          tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))

    return self.optimizer.minimize(tf.reduce_mean(loss))
      
      
      

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    """Stores a transition when in training mode.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer (last_observation, action, reward,
    is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
    """
    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.memory.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)
