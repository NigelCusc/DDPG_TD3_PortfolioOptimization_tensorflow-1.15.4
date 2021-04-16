"""
    TD3 CRITIC
    Tensorflow 1 implementation of paper: https://arxiv.org/abs/1802.09477
    Fujimoto et. al. "Addressing Function Approximation in Actor-Critic Methods
"""

import tensorflow as tf


class TD3CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, inp_actions):
        self.sess = sess
        assert isinstance(state_dim, list), 'state_dim must be a list.'
        self.s_dim = state_dim
        assert isinstance(action_dim, list), 'action_dim must be a list.'
        self.a_dim = action_dim
        self.inp_actions = inp_actions
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs = tf.placeholder(shape=[None] + self.s_dim + [1], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None] + self.a_dim, dtype=tf.float32)

        # Main Critic
        self.total_out, _ = self.create_critic_network('main_critic', self.inp_actions)
        self.out1, self.out2 = self.create_critic_network('main_critic', self.action, reuse=True)

        #self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main_critic')
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Critic
        self.target_out1, self.target_out2 = self.create_critic_network('target_critic', self.action)

        #self.target_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_critic')
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.out1 - self.predicted_q_value)) + tf.reduce_mean(
            tf.square(self.out2 - self.predicted_q_value))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=self.network_params)

    def create_critic_network(self, scope, actions, reuse=False):
        raise NotImplementedError('Create critic should return (inputs, action, out)')

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def predict1(self, inputs, action):
        return self.sess.run(self.out1, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict2(self, inputs, action):
        return self.sess.run(self.out2, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target1(self, inputs, action):
        return self.sess.run(self.target_out1, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target2(self, inputs, action):
        return self.sess.run(self.target_out2, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
