from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import prioritized_replay_buffer
import tensorflow as tf

import gin.tf


@gin.configurable
class DQN_GER_Agent(dqn_agent.DQNAgent):
  """A compact implementation of a GER agent."""

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

    self.buffer_size = tf.compat.v1.placeholder(tf.float32)

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

    


  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
#    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)


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

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_values, 1)
    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).    
    
    return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))


  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    inputs = self._replay.states
    target = tf.stop_gradient(self._build_target_q_op())
    acts = self._replay.actions
    replay_action_one_hot = tf.one_hot(acts, self.num_actions, 1., 0., name='action_one_hot')

    # function to extract per-sample gradient. See https://www.tensorflow.org/api_docs/python/tf/vectorized_map
    def model_fn(arg):
        with tf.compat.v1.GradientTape() as g:
            (inputs, targets, one_hot) = arg
            
            inputs = tf.expand_dims(inputs, 0)
            targets = tf.expand_dims(targets, 0)
            one_hot = tf.expand_dims(one_hot, 0)

            replay_net_outputs_inside = self.online_convnet(inputs)

            replay_chosen_q_inside = tf.reduce_sum(
                replay_net_outputs_inside.q_values * one_hot,
                axis=1,
                name='replay_chosen_q_inside')
            
            loss_inside = tf.compat.v1.losses.huber_loss(replay_chosen_q_inside, targets)
            

        return g.gradient(loss_inside, (self.online_convnet.conv1.kernel, self.online_convnet.conv1.bias, 
                    self.online_convnet.conv2.kernel, self.online_convnet.conv2.bias, 
                    self.online_convnet.conv3.kernel, self.online_convnet.conv3.bias,
                    self.online_convnet.dense1.kernel, self.online_convnet.dense1.bias,
                    self.online_convnet.dense2.kernel, self.online_convnet.dense2.bias)), loss_inside


    extract_gradients, loss = tf.compat.v1.vectorized_map(model_fn, (inputs, target, replay_action_one_hot))

    # Process as PER but with per-sample gradient norms
    probs = self._replay.transition['sampling_probabilities']
    loss_weights = 1.0 / probs
    loss_weights /= tf.reduce_max(loss_weights)   
    loss = loss_weights * loss
    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))



    grad_conv1_ker = extract_gradients[0]
    batch_size = tf.shape(grad_conv1_ker)[0]
    grad_conv1_ker = tf.reshape(grad_conv1_ker, (batch_size, -1))
    norm_square_grad_conv1_ker = tf.norm(grad_conv1_ker, axis=1)**2
    t1 = norm_square_grad_conv1_ker
    
    grad_conv1_bias = extract_gradients[1]
    norm_square_grad_conv1_bias = tf.norm(grad_conv1_bias, axis=1)**2
    t2 = norm_square_grad_conv1_bias

    grad_conv2_ker = extract_gradients[2]
    grad_conv2_ker = tf.reshape(grad_conv2_ker, (batch_size, -1))
    norm_square_grad_conv2_ker = tf.norm(grad_conv2_ker, axis=1)**2
    t3 = norm_square_grad_conv2_ker
    
    grad_conv2_bias = extract_gradients[3]
    norm_square_grad_conv2_bias = tf.norm(grad_conv2_bias, axis=1)**2
    t4 = norm_square_grad_conv2_bias

    grad_conv3_ker = extract_gradients[4]
    grad_conv3_ker = tf.reshape(grad_conv3_ker, (batch_size, -1))
    norm_square_grad_conv3_ker = tf.norm(grad_conv3_ker, axis=1)**2
    t5 = norm_square_grad_conv3_ker

    grad_conv3_bias = extract_gradients[5]
    norm_square_grad_conv3_bias = tf.norm(grad_conv3_bias, axis=1)**2
    t6 = norm_square_grad_conv3_bias

    grad_dense1_ker = extract_gradients[6]
    norm_square_grad_dense1_ker = tf.norm(grad_dense1_ker, axis=(1,2))**2
    t7 = norm_square_grad_dense1_ker
    
    grad_dense1_bias = extract_gradients[7]
    norm_square_grad_dense1_bias = tf.norm(grad_dense1_bias, axis=1)**2
    t8 = norm_square_grad_dense1_bias
    
    grad_dense2_ker = extract_gradients[8]
    norm_square_grad_dense2_ker = tf.norm(grad_dense2_ker, axis=(1,2))**2
    t9 = norm_square_grad_dense2_ker
    
    grad_dense2_bias = extract_gradients[9]
    norm_square_grad_dense2_bias = tf.norm(grad_dense2_bias, axis=1)**2
    t10 = norm_square_grad_dense2_bias

    per_sample_gradients_norm = tf.sqrt(t1+t2+t3+t4+t5+t6+t7+t8+t9+t10)
    
    update_priorities_op = self._replay.tf_set_priority(self._replay.indices, per_sample_gradients_norm)

    return self.optimizer.minimize(tf.reduce_mean(loss)), update_priorities_op 
   
    


  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        _, __ = self._sess.run(self._train_op, feed_dict={self.buffer_size: self._replay.memory.add_count if self._replay.memory.add_count < self._replay.memory._replay_capacity else self._replay.memory._replay_capacity})
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1



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
