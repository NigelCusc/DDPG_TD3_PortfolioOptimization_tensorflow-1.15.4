"""
TD3 implementation in Tensorflow
"""
from __future__ import print_function

import os 
import traceback
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from model.replay_buffer import ReplayBuffer
from ..base_model import BaseModel


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# TO DO Remove Actor Noise (old version)

class TD3(BaseModel):
    def __init__(self, env, sess, actor, critic, actor_noise,
                 obs_normalizer=None, action_processor=None, log_return=None,
                 config_file='config/default.json',
                 model_save_path='weights/td3/td3.ckpt',
                 summary_path='results/td3/'):
        with open(config_file) as f:
            self.config = json.load(f)
        assert self.config != None, "Can't load config file"
        np.random.seed(self.config['seed'])
        if env:
            env.seed(self.config['seed'])
        self.model_save_path = model_save_path
        self.summary_path = summary_path
        self.sess = sess
        # if env is None, then TD3 just predicts
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_noise = actor_noise
        self.obs_normalizer = obs_normalizer
        self.action_processor = action_processor
        self.log_return = log_return
        self.summary_ops, self.summary_vars = build_summaries()

        self.actor_loss = -tf.reduce_mean(self.critic.total_out)
        self.actor_train_step = tf.train.AdamOptimizer(actor.learning_rate).minimize(self.actor_loss,
                                                                                               var_list=self.actor.network_params)
        
        

    def initialize(self, load_weights=True, verbose=True):
        """
        Load training history from path.
        To be add feature to just load weights, not training states
        """
        if load_weights:
            try:
                variables = tf.global_variables()
                param_dict = {}
                saver = tf.train.Saver()
                saver.restore(self.sess, self.model_save_path)
                for var in variables:
                    var_name = var.name[:-2]
                    if verbose:
                        print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))
                    param_dict[var_name] = var
            except:
                traceback.print_exc()
                print('Build model from scratch')
                self.sess.run(tf.global_variables_initializer())
        else:
            print('Build model from scratch')
            self.sess.run(tf.global_variables_initializer())

    def train(self, verbose=True, debug=False):
        """
        Must already call intialize
        Args:
            verbose:
            debug:
        Returns:
        """
        writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        print('inside TD3 train')
        self.actor.update_target_network()
        self.critic.update_target_network()

        np.random.seed(self.config['seed'])
        num_episode = self.config['episode']
        value_function_threshold = self.config['value_function_threshold']
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        self.buffer = ReplayBuffer(self.config['buffer size'])

        # TD3 parameters:
        policy_noise = self.config['td3_policy_noise']
        noise_clip = self.config['td3_noise_clip']
        policy_freq = self.config['td3_policy_freq']
        
        # Episode rewards
        ep_reward_list = []

        # main training loop
        for i in range(num_episode):
            
            # Check if threshold exceeded in last 10 episodes
            if len(ep_reward_list) > 10:
                if all(i >= value_function_threshold for i in ep_reward_list[-10:]):
                    break
            
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count))

            previous_observation, previous_observation_ti, _  = self.env.reset()
            if self.obs_normalizer:
                previous_observation = self.obs_normalizer(previous_observation, self.log_return, ti_observation=previous_observation_ti)
                        
            ep_reward = 0
            ep_ave_max_q = 0
            # keeps sampling until done
            for j in range(self.config['max step']):
                action = self.actor.predict(np.expand_dims(previous_observation, axis=0)).squeeze(axis=0) # + self.actor_noise()

                if self.action_processor:
                    action_take = self.action_processor(action)
                else:
                    action_take = action

                # step forward
                observation, ti_observation, reward, done, _, weights, _ = self.env.step(action_take)

                if self.obs_normalizer:
                    observation = self.obs_normalizer(observation, self.log_return, ti_observation=previous_observation_ti)

                # add to buffer
                self.buffer.add(previous_observation, action, reward, done, observation)

                if self.buffer.size() >= batch_size:
                    # batch update
                    s_batch, a_batch, r_batch, t_batch, next_state = self.buffer.sample_batch(batch_size)
                    
                    # Trick Three: Target Policy Smoothing. ==================
                    # TD3 adds noise to the target action, to make it harder for 
                    # the policy to exploit Q-function errors by smoothing out Q 
                    # along changes in action.
                    noise = np.random.normal(0, policy_noise, size=a_batch.shape)
                    noise = np.clip(noise, -noise_clip, noise_clip)
                    
                    action_bound = 1.
                    next_action = self.actor.predict_target(next_state) + noise
                    next_action = np.clip(next_action, -action_bound, action_bound)
                    
                    #print("next_action: {}".format(next_action))
                    
                    # Trick One: Clipped Double-Q Learning. ==================
                    # TD3 learns two Q-functions instead of one (hence “twin”), 
                    # and uses the smaller of the two Q-values to form the targets 
                    # in the Bellman error loss functions.
                    target_q1 = self.critic.predict_target1(next_state, next_action)
                    target_q2 = self.critic.predict_target2(next_state, next_action)
                    target_q = np.minimum(target_q1, target_q2)
                    

                    y_i = []
                    for k in range(batch_size):
                        if t_batch[k]:
                            y_i.append(r_batch[k])
                        else:
                            y_i.append(r_batch[k] + gamma * target_q[k])

                    self.sess.run(self.critic.train_step, feed_dict={
                        self.critic.inputs: s_batch,
                        self.critic.action: a_batch,
                        self.critic.predicted_q_value: np.reshape(y_i, [-1, 1])})

                    ep_ave_max_q += np.amax(np.reshape(y_i, [-1, 1]))
                    
                    # Trick Two: “Delayed” Policy Updates. ===================
                    # TD3 updates the policy (and target networks) less frequently 
                    # than the Q-function. The paper recommends one policy update 
                    # for every two Q-function updates.
                    if i % policy_freq == 0:
                        self.sess.run(self.actor_train_step, feed_dict={
                            self.actor.inputs: s_batch,
                            self.critic.inputs: s_batch})
                        self.actor.update_target_network()
                        self.critic.update_target_network()

                ep_reward += reward
                previous_observation = observation

                if done or j == self.config['max step'] - 1:
                    summary_str = self.sess.run(self.summary_ops, feed_dict={
                        self.summary_vars[0]: ep_reward,
                        self.summary_vars[1]: ep_ave_max_q / float(j)
                    })

                    writer.add_summary(summary_str, i)
                    writer.flush()

                    # Plot weights for each 10
                    #if i % 10 == 0:
                    #   self.plot_weights(i, a_batch)
                    
                    
                    print('sample of weights: ')
                    print(np.around(a_batch[0:2], decimals=2))
                    
                    print('Episode: {:d}, Reward: {:.2f}, Qmax: {:.4f}'.format(i, ep_reward, (ep_ave_max_q / float(j))))
                    ep_reward_list.append(ep_reward)
                    break
        print('save model.')
        self.save_model(verbose=True)
        print('Finish.')

    def predict(self, observation, ti_observation=None):
        """
        Predict the next action using actor model, only used in deploy.
        Can be used in multiple environments.
        Args:
            observation: (batch_size, num_stocks + 1, window_length)

        Returns: action array with shape (batch_size, num_stocks + 1)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation, self.log_return, ti_observation=ti_observation)
        action = self.actor.predict(observation)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def predict_single(self, observation, ti_observation=None):
        """
        Predict the action of a single observation
        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)
        """
        if self.obs_normalizer:
            observation = self.obs_normalizer(observation, self.log_return, ti_observation=ti_observation)
        action = self.actor.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        if self.action_processor:
            action = self.action_processor(action)
        return action

    def save_model(self, verbose=False):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

        saver = tf.train.Saver()
        model_path = saver.save(self.sess, self.model_save_path)
        print("Model saved in %s" % model_path)

    def plot_weights(self, i, action_list):
        # Define columns
        asset_len = len(action_list[0])

        # Create Dataframe
        df = pd.DataFrame(action_list, columns=list(range(asset_len)))
        # PLOT WEIGHTS
        plt.figure(figsize=(10, 8), dpi=100)
        plt.title('Sample of weights for episode {:d}'.format(i))
        plt.xlabel('Day')
        plt.ylabel('Weights')
        for i in range(asset_len):
            plt.plot(df[i], label=i)
        # plt.legend()
        plt.show()
