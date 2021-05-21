from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dopamine.agents.dqn import dqn_agent
import tensorflow as tf

import gin.tf


@gin.configurable
class DQN_LABER_Agent(dqn_agent.DQNAgent):
  """A compact implementation of a LaBER agent."""

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

    big_batch_size_int = tf.shape(replay_chosen_q)[0]
    bs = big_batch_size_int // 4 # we assume a large batch of size 4*batch_size has been sampled

    target = tf.stop_gradient(self._build_target_q_op())
    
    # compute priorities on the large batch
    td_errors = tf.sqrt(tf.compat.v1.losses.mean_squared_error(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE))
    ones = tf.ones(big_batch_size_int)
    # The Huber loss is used, so the priorities are min(|td errors|, 1)
    td_errors = tf.compat.v1.math.minimum(ones, td_errors)
    probs = td_errors/tf.compat.v1.math.reduce_sum(td_errors)
    probs = tf.compat.v1.expand_dims(probs, axis=0)
    # Down-sample the large batch according to priorities
    indices = tf.compat.v1.multinomial(tf.compat.v1.log(probs), bs)[0]
    td_error_for_selected_indices = tf.gather(td_errors, indices)
    new_target = tf.gather(target, indices)
    new_replay_chosen_q = tf.gather(replay_chosen_q, indices)
    # Compute weights for SGD update
    loss_weights = 1.0 / td_error_for_selected_indices ### LaBER-lazy
    loss_weights = tf.stop_gradient(loss_weights)

    loss = tf.compat.v1.losses.huber_loss(
        new_target, new_replay_chosen_q, loss_weights,
        reduction=tf.losses.Reduction.NONE)

    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))

