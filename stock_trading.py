"""
    Train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise
from model.td3.actor import TD3ActorNetwork
from model.td3.critic import TD3CriticNetwork
from model.td3.td3 import TD3
from model.benchmarks_olps import algos
from environment.portfolio import PortfolioEnv, max_drawdown, sharpe, sortino, create_close_dataframe, convert_prices
from utils.data import read_stock_history, normalize
from technical_indicators.technical_indicators import full_rmr_moving_average, olmar_moving_average
import numpy as np
import tflearn
import tensorflow as tf
import argparse
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

DEBUG = False

# Path Generation Functions ===========================================================================================
def get_model_path(dataset_name, framework, window_length, predictor_type, use_batch_norm, technical_indicators_flag):
    '''
    Args:
        dataset_name: e.g. 'nyse_n'
        framework: e.g. 'DDPG'
        window_length: e.g. '7'
        predictor_type: e.g. 'lstm'
        use_batch_norm: e.g. True
        technical_indicators_flag: e.g. True

    Returns: Model Path string

    '''
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
        
    if technical_indicators_flag:
        technical_indicators_str = 'technical_indicators'
        return 'weights/{}/{}/{}/window_{}/{}/{}/checkpoint.ckpt'.format(dataset_name, framework, predictor_type, window_length, batch_norm_str, technical_indicators_str)
    else: 
        #technical_indicators_str = 'no_technical_indicators'
        return 'weights/{}/{}/{}/window_{}/{}/checkpoint.ckpt'.format(dataset_name, framework, predictor_type, window_length, batch_norm_str)


def get_result_path(dataset_name, framework, window_length, predictor_type, use_batch_norm, technical_indicators_flag):
    '''

    Args:
        dataset_name: e.g. 'nyse_n'
        framework: e.g. 'DDPG'
        window_length: e.g. '7'
        predictor_type: e.g. 'lstm'
        use_batch_norm: e.g. True
        technical_indicators_flag: e.g. True

    Returns: Result Path string

    '''
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
        
    if technical_indicators_flag:
        technical_indicators_str = 'technical_indicators'
        return 'results/{}/{}/{}/window_{}/{}/{}/'.format(dataset_name, framework, predictor_type, window_length, batch_norm_str, technical_indicators_str)
    else: 
        #technical_indicators_str = 'no_technical_indicators'
        return 'results/{}/{}/{}/window_{}/{}/'.format(dataset_name, framework, predictor_type, window_length, batch_norm_str)


def get_variable_scope(dataset_name, framework, window_length, predictor_type, use_batch_norm, technical_indicators_flag):
    '''

    Args:
        dataset_name: e.g. 'nyse_n'
        framework: e.g. 'DDPG'
        window_length: e.g. '7'
        predictor_type: e.g. 'lstm'
        use_batch_norm: e.g. True
        technical_indicators_flag: e.g. True

    Returns: Variable Path string

    '''
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
        
    if technical_indicators_flag:
        technical_indicators_str = 'technical_indicators'
        return '{}_{}_{}_window_{}_{}_{}'.format(dataset_name, framework, predictor_type, window_length, batch_norm_str, technical_indicators_str)
    else: 
        #technical_indicators_str = 'no_technical_indicators'
        return '{}_{}_{}_window_{}_{}'.format(dataset_name, framework, predictor_type, window_length, batch_norm_str)


# Generate Stock Predictor (NET) based on Predictor Type ==============================================================
def stock_predictor(inputs, predictor_type, use_batch_norm):
    '''

    Args:
        inputs: 
        predictor_type: e.g. 'lstm'
        use_batch_norm: e.g. True

    Returns: Neural Network / Predictor

    '''
    window_length = inputs.get_shape()[2]
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    if predictor_type == 'cnn':
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid')
        if use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        if DEBUG:
            print('After conv2d:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    elif predictor_type == 'lstm':
        num_stocks = inputs.get_shape()[1]
        hidden_dim = 32
        net = tflearn.reshape(inputs, new_shape=[-1, window_length, 1])
        if DEBUG:
            print('Reshaped input:', net.shape)
        net = tflearn.lstm(net, hidden_dim)
        if DEBUG:
            print('After LSTM:', net.shape)
        net = tflearn.reshape(net, new_shape=[-1, num_stocks, hidden_dim])
        if DEBUG:
            print('After reshape:', net.shape)
        net = tflearn.flatten(net)
        if DEBUG:
            print('Output:', net.shape)
    else:
        raise NotImplementedError

    return net


# Classes for ACTORS AND CRITICS (Might need to move this out) ========================================================
# DDPG
class StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape
        """
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1], name='input')

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


class StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1])
        action = tflearn.input_data(shape=[None] + self.a_dim)

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        net = tf.add(t1, t2)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })


# TD3
class TD3StockActor(ActorNetwork):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        TD3ActorNetwork.__init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size)

    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape
        """
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1], name='input')

        net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)

        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        if self.use_batch_norm:
            net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out
    
    def train(self, inputs, a_gradient):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        window_length = self.s_dim[1]
        inputs = inputs[:, :, -window_length:, :]
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })


class TD3StockCritic(CriticNetwork):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars,
                 predictor_type, use_batch_norm, inp_actions):
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm
        TD3CriticNetwork.__init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, inp_actions)

    def create_critic_network(self, scope, actions, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            # NET1 -------------------------------
            net1 = stock_predictor(self.inputs, self.predictor_type, self.use_batch_norm)
            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            t1 = tflearn.fully_connected(net1, 64)
            t2 = tflearn.fully_connected(actions, 64)

            net1 = tf.add(t1, t2)
            if self.use_batch_norm:
                net1 = tflearn.layers.normalization.batch_normalization(net1)
            net1 = tflearn.activations.relu(net1)

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out1 = tflearn.fully_connected(net1, 1, weights_init=w_init)

            # NET2 -------------------------------
            net2 = stock_predictor(self.inputs, self.predictor_type, self.use_batch_norm)
            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases
            t1 = tflearn.fully_connected(net2, 64)
            t2 = tflearn.fully_connected(actions, 64)

            net2 = tf.add(t1, t2)
            if self.use_batch_norm:
                net2 = tflearn.layers.normalization.batch_normalization(net2)
            net2 = tflearn.activations.relu(net2)

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            out2 = tflearn.fully_connected(net2, 1, weights_init=w_init)

            return out1, out2

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


# Observation Normalizing Function ====================================================================================
def obs_normalizer(observation, log_return_flag=False, ti_observation=None):   
    ''' 
    Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info
        log_return_flag: e.g. True - Form the returns in Log format
        ti_observation: observation consisting of the predefined technical indicators. To be included in our observation/state

    '''

    if isinstance(observation, tuple):
        observation = observation[0]
    
    if observation.shape[2] == 4:        # Multiple Features
        # directly use close/open ratio as feature
        observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    elif observation.shape[2] == 2:     # Close Feature
        # Use Previous Close
        observation = observation[:, :, 1] / observation[:, :, 0]
        observation = observation.reshape(observation.shape[0], observation.shape[1], 1)

    if log_return_flag:
        observation = np.log(observation)
    
    if ti_observation is not None:
        new_observation = []
        for i in range(ti_observation.shape[0]):
            # Turn to Array
            t = np.vstack(ti_observation[i])
            
            # Append to our second part of the observation
            new_observation.append(np.vstack((observation[i], t)))
        new_observation = np.array(new_observation)   
        observation = new_observation
                
    observation = normalize(observation)
    
    #print("observation shape: {} ---------------------------------------------------".format(observation.shape))
    #print("observation: {}".format(observation))
    
    #print("ti_observation shape: {} ---------------------------------------------------".format(ti_observation.shape))
    #print("ti_observation: {}".format(ti_observation))
    
    return observation


# Test Model Functions ================================================================================================
def test_model(env, model):
    ''' 
    Test Model on a defined environment

    Args:
        env: a defined environment
        model: DRL model

    '''
    observation, observation_ti, info = env.reset()
    
    done = False
    dates = []
    actions = []
    weights_list = []
    portfolio_value = []
    returns = []
    market_value = []
    while not done:
        action = model.predict_single(observation, ti_observation = observation_ti)
        observation, observation_ti, _, done, info, weights, _ = env.step(action)
        actions.append(action)
        weights_list.append(weights)
        portfolio_value.append(extract_from_infos([info], 'portfolio_value')[0])
        returns.append(extract_from_infos([info], 'return')[0])
        market_value.append(info['market_value'])
        dates.append(info['date'])
    df_performance = env.render()
    return dates, observation, info, actions, weights_list, df_performance, portfolio_value, market_value


def test_model_multiple(env, models):
    ''' 
    Test list of Models on a defined environment

    Args:
        env: a defined environment
        models: list of DRL models

    '''
    observations_list = []
    actions_list = []
    info_list = []
    observation, observation_ti, info = env.reset()
    done = False
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation, ti_observation = observation_ti))
        actions = np.array(actions)
        observation, observation_ti, _, done, info, _ = env.step(actions)

        observations_list.append(observation)
        actions_list.append(actions)
        info_list.append(info)
    df_performance = env.render()
    return observations_list, info_list, actions_list, df_performance


# Test Online Portfolio Selection Model Functions =====================================================================
def get_algo(env, model):
    ''' 
    Set up OLPS model

    Args:
        env: a defined environment
        model: OLPS model name in all caps e.g. 'OLMAR'

    '''
    if model == 'CRP':
        return algos.CRP()
    elif model == 'BCRP':
        return algos.BCRP()
    elif model == 'OLMAR':
        return algos.OLMAR() #(window=env.window_length, eps=5)
    elif model == 'RMR':
        return algos.RMR() #(window=env.window_length, eps=5)
    elif model == 'PAMR':
        return algos.PAMR()
    elif model == 'WMAMR':
        return algos.WMAMR() #(window=env.window_length)
    elif model == 'EG':
        return algos.EG()
    elif model == 'ONS':
        return algos.ONS(delta=0.125, beta=1., eta=0.)
    elif model == 'UP':
        return algos.UP()
    elif model == 'CORN':
        return algos.CORN()
    elif model == 'ANTICOR':
        return algos.Anticor()
    else:
        return print("Not Implemented...")


def test_portfolio_selection(env, model):
    ''' 
    Test OLPS model

    Args:
        env: a defined environment
        model: OLPS model name in all caps e.g. 'OLMAR'

    '''
    algo = get_algo(env, model)
    observations_list, infos, weights_array = algo.run(env)
    env.render()
    return observations_list, infos, weights_array


def test_portfolio_selection_multiple(env, models):
    ''' 
    Test OLPS models

    Args:
        env: a defined environment
        models: List of OLPS model names in all caps e.g. 'OLMAR'

    '''
    if isinstance(models, list):
        observations_list = []
        portfolio_values_list = []
        weights_list = []
        for model in models:
            print("========={}=========".format(model))
            algo = get_algo(env, model)
            observations, portfolio_values, weights, dates = algo.run(env)

            observations_list.append(observations)
            portfolio_values_list.append(portfolio_values)
            weights_list.append(weights)

        return observations_list, portfolio_values_list, weights_list, dates
    else:
        print("Models object not in list format")


def test_with_given_weights(env, weights_array):
    '''
    Iterate through testing phase with the given weights to generate the portfolio.
    
    Args:
    env : Portfolio environment
    weights : Array of daily portfolio weights

    '''

    rewards = []
    portfolio_values = []
    weights_list = []
    dates_list = []
    reward_list = []

    observation, observation_ti, info = env.reset()
    done = False
    i = 0
    while not done:
        w = weights_array[i]
        if isinstance(w, list):
            w = np.array(w)

        observation, observation_ti, reward, done, info, weights, _ = env.step(w)
        rewards.append(reward)
        portfolio_values.append(info['portfolio_value'])
        weights_list.append(weights)
        dates_list.append(info['date'])
        reward_list.append(reward)
        i += 1
    
    return dates_list, observation, portfolio_values, weights_list, reward_list



# Functions on results =====================================================================================
        
def plot_portfolio_values(models, portfolio_values_list, dates, log_y=True):
    ''' 
    Plot the portfolio Results

    Args:
        models: List of DRL and/or OLPS model names in all caps e.g. 'OLMAR'
        portfolio_values_list: List of results corresponding to the model names in the previous parameter
        dates: List of dates corresponding to the results
        Log_y: Flag indicating if the y axis is to be set in Log format. e.g. True
    '''
    if len(models) == len(portfolio_values_list):
        df = pd.DataFrame()
        df["Date"] = dates
        df.set_index('Date', inplace=True)
        plt.figure(figsize=(10, 6), dpi=100)
        if log_y:
            plt.title('Portfolio Values (LOG Y)')
            plt.ylabel('Portfolio Value (LOG Y)')
        else:
            plt.title('Portfolio Values')
            plt.ylabel('Portfolio Value')
        plt.xlabel('Day')
        for i in range(len(models)):
            df[models[i]] = portfolio_values_list[i]
            plt.plot(df[models[i]], label=models[i])
        if log_y:
            plt.yscale('log')
        plt.xticks(np.arange(0, len(dates), 200))
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()
    else:
        print("Models and Infos_list have different sizes")


def plot_weights(weights_array):
    ''' 
    Plot the portfolio Weights

    Args:
        weights_array: Array of weights 
    '''
    # Define columns
    asset_length = len(weights_array[0])
    # Create Dataframe
    df = pd.DataFrame(weights_array, columns=list(range(asset_length)))
    # PLOT WEIGHTS
    plt.figure(figsize=(8, 6), dpi=100)
    plt.title('Asset Weights')
    plt.xlabel('Day')
    plt.ylabel('Weights')
    for i in range(asset_length):
        plt.plot(df[i], label=i)
    # plt.legend()
    plt.show()


def extract_from_infos(infos, item):
    ''' 
    Extract an item from the info list

    Args:
        infos: List of information gathered during training or testing a DRL model, or multiple models
        item: string name of item to be extracted
    '''
    result = []
    for info in infos:
        result.append(info[item])
    return result


# RESULTS TABLE
def results_table(models, portfolio_values_list):
    ''' 
    Calculate results from the cumulative returns table. Including:
    'Portfolio', 'Average Daily Yield (%)', 'Sharpe Ratio (%)', 'Sortino Ratio (%)', 'Maximum Drawdown (%)', 'Final Portfolio Value'

    Args:
        models: List of Model names
        portfolio_values_list: List of cumulative returns corresponding to the model names
    '''
    # RESULT
    df = pd.DataFrame(columns=['Portfolio', 'Average Daily Yield (%)', 'Sharpe Ratio (%)', 'Sortino Ratio (%)', 'Maximum Drawdown (%)',
                               'Final Portfolio Value'])
    for i in range(len(models)):
        df = results_table_row(df, models[i],
                               returns_from_cumulative(portfolio_values_list[i]),
                               portfolio_values_list[i])
    return df


# Used in the above method
def results_table_row(df, name, returns, portfolio_values):
    ''' 
    Calculate row of results from the cumulative returns table. Including:
    'Portfolio', 'Average Daily Yield (%)', 'Sharpe Ratio (%)', 'Sortino Ratio (%)', 'Maximum Drawdown (%)', 'Final Portfolio Value'

    Args:
        df: Dataframe to append to
        name: Name of current model
        returns: List of returns
        portfolio_values: List of cumulative returns
    '''
    df = df.append({'Portfolio': name,
                    'Average Daily Yield (%)': round(float(np.mean(returns))*100, 4),
                    'Sharpe Ratio (%)': round(sharpe(returns)*100, 4),
                    'Sortino Ratio (%)': round(sortino(returns)*100, 4),
                    'Maximum Drawdown (%)': round(max_drawdown(portfolio_values)*100, 4),
                    'Final Portfolio Value': round(float(portfolio_values[-1]), 3)
                    }, ignore_index=True)
    return df


def returns_from_cumulative(cumulative_returns):
    ''' 
    Generate returns from cumulative returns

    Args:
        cumulative_returns: List of cumulative returns
    '''
    returns = []
    for i in range(len(cumulative_returns)-1):
        i += 1
        returns.append(cumulative_returns[i] / cumulative_returns[i-1] - 1)
    return returns


# Generate Technical Indicators
def Generate_technical_indicators(window_length, history, debug = False):
    if debug:
        print('history.shape: {}'.format(history.shape))

    # Check if just close is fed or full dataset
    if history.shape[2] == 4:
        # Get Close
        history_close = history[:, :, 3]
    elif history.shape[2] == 2:
        # Assume the one sent is the close price
        history_close = history[:, :, 1]

    if debug:
        print('history_close.shape: {}'.format(history_close.shape))

    # Add Technical Indicators to be included in state
    technical_indicator_history = []

    # Close Price DataFrame for on-line Portfolio Selection
    temp_close_df = create_close_dataframe(history, assets, date_list)
    temp_close_df = convert_prices(temp_close_df, 'raw', True)    
    if debug:
        print('temp_close_df.shape: {}'.format(temp_close_df.shape))

    rmr_moving_average_df = full_rmr_moving_average(temp_close_df, window_length)
    for i in range(len(assets)):
        # Create List
        ti = []
        #ti.append([olmar_moving_average(temp_close_df[assets[i]], window_length)]) # Based on OLMAR
        ti.append([rmr_moving_average_df[assets[i]]]) # Based on RMR

        # Turn to Array
        ti = np.vstack(ti)

        ti_reshaped = []
        for j in range(temp_close_df.shape[0]):
            ti_reshaped.append(ti[:, j])   

        technical_indicator_history.append(ti_reshaped)

    technical_indicator_history = np.array(technical_indicator_history)    
    if debug:
        print('technical_indicators.shape: {}'.format(technical_indicator_history.shape)) 
        
    return technical_indicator_history

# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provide arguments for training different models')

    parser.add_argument('--debug', '-d', help='print debug statement', default=True)
    parser.add_argument('--framework', '-f', help='DDPG or TD3', default='DDPG')
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', default='lstm')
    parser.add_argument('--window_length', '-w', help='observation window length', default=3)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', default='True')
    parser.add_argument('--technical_indicators', '-t', help='whether to use technical indicators', default='False')

    args = vars(parser.parse_args())

    pprint.pprint(args)
    
    # Hardcoded
    log_return = True
    config_file_path = 'config/stock.json'    
    window_length = int(args['window_length'])    
    
    assert args['framework'] in ['DDPG', 'TD3'], 'Framework must be either PG, DDPG or TD3'
    framework = args['framework']
    assert args['predictor_type'] in ['cnn', 'lstm'], 'Predictor must be either cnn or lstm'
    predictor_type = args['predictor_type']
    if args['batch_norm'] == 'True':
        use_batch_norm = True
    elif args['batch_norm'] == 'False':
        use_batch_norm = False
    else:
        raise ValueError('Unknown batch norm argument')
        
    if args['technical_indicators'] == 'True':
        technical_indicators_flag = True
        # use batch norm must be aplied as well
        use_batch_norm = True
    elif args['technical_indicators'] == 'False':
        technical_indicators_flag = False
    else:
        raise ValueError('Unknown batch norm argument')
    
    # Print
    print("Model: {}-{}-{}".format(framework, predictor_type, window_length))
    

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False
        
    # Open Config
    with open(config_file_path) as f:
        config = json.load(f)
    assert config != None, "Can't load config file"
        
    # Stock History
    dataset_name = config['dataset']
    history, assets, date_list = read_stock_history(filepath='utils/datasets/{}.h5'.format(dataset_name))
    history = history[:, :, :4]
    print("Dataset: {}".format(dataset_name))
    print("Stock History Shape: {}".format(history.shape))
    print("Full Stock History Date Range: {} -> {}".format(date_list[0], date_list[-1]))
        
    # Training/Testing Date Range
    if dataset_name == 'Hegde':
        full_length = len(date_list)
        train_ratio = 7/10
        validation_ratio = 2/10
        train_start_date = date_list[window_length]
        train_end_date = date_list[(int)(full_length * train_ratio) - 1]
        #validation_start_date = date_list[(int)(full_length * train_ratio)]
        #validation_end_date = date_list[(int)(full_length * (train_ratio + validation_ratio)) - 1]
        test_start_date = date_list[(int)(full_length * (train_ratio + validation_ratio))]
        test_end_date = date_list[full_length - 2]
    else:
        full_length = len(date_list)
        train_test_ratio = 6/7
        train_start_date = date_list[window_length]
        train_end_date = date_list[(int)(full_length * train_test_ratio)-1]
        test_start_date = date_list[(int)(full_length * train_test_ratio)]
        test_end_date = date_list[full_length-2]
    print("Training Date Range: {} -> {} ({} Steps)".format(train_start_date, train_end_date, 
                                                        (int)(date_list.index(train_end_date) - date_list.index(train_start_date))))
    print("Testing Date Range: {} -> {} ({} Steps)".format(test_start_date, test_end_date, 
                                                        (int)(date_list.index(test_end_date) - date_list.index(test_start_date))))
    
    # Episode steps
    steps = 1000
    print("Episode Steps: {}".format(steps))
    
    # Generate Technical Indicators
    print('history.shape: {}'.format(history.shape))
    
    # Check if just close is fed or full dataset
    if history.shape[2] == 4:
        # Get Close
        history_close = history[:, :, 3]
    elif history.shape[2] == 2:
        # Assume the one sent is the close price
        history_close = history[:, :, 1]
    else:
        print("Invalid History Fomrat. Must be (x, y) or (x, y, 4)")
        sys.exit(1)
    
    print('history_close.shape: {}'.format(history_close.shape))
    
    # Add Technical Indicators to be included in state
    if technical_indicators_flag:
        technical_indicator_history = []
        
        # Close Price DataFrame for on-line Portfolio Selection
        temp_close_df = create_close_dataframe(history, assets, date_list)
        temp_close_df = convert_prices(temp_close_df, 'raw', True)    
        print('temp_close_df.shape: {}'.format(temp_close_df.shape))
        
        rmr_moving_average_df = full_rmr_moving_average(temp_close_df, window_length)
        print(rmr_moving_average_df.head())
        for i in range(len(assets)):
            # Create List
            ti = []
            #ti.append([moving_average(history_close[i], window_length)]) 
            #ti.append([olmar_moving_average(temp_close_df[assets[i]], window_length)]) # Based on OLMAR
            ti.append([rmr_moving_average_df[assets[i]]]) # Based on RMR
            #ti.append([moving_average(history_close[i], round(window_length/2))])
            #ti.append([momentum(history_close[i], window_length)])
            #ti.append([rate_of_change(history_close[i], window_length)])
        
            #Upper, Lower = bollinger_bands(history_close[i], window_length)
            #ti.append([Upper])
            #ti.append([Lower])
            #ti.append([standard_deviation(history_close[i], window_length)])
            
            # Turn to Array
            ti = np.vstack(ti)
            
            ti_reshaped = []
            for j in range(temp_close_df.shape[0]):
                ti_reshaped.append(ti[:, j])   
            
            technical_indicator_history.append(ti_reshaped)
            
        technical_indicator_history = np.array(technical_indicator_history)    
        print('technical_indicators.shape: {}'.format(technical_indicator_history.shape)) 
    else:
        technical_indicator_history = None

    # setup environment
    env = PortfolioEnv(history, assets, date_list, end_date=train_end_date, steps=steps, window_length=window_length, 
                       technical_indicators_flag=technical_indicators_flag, technical_indicator_history=technical_indicator_history)

    nb_classes = len(assets) + 1
    action_dim = [nb_classes]
    if technical_indicators_flag:
        state_dim = [nb_classes, window_length+len(ti)]
    else:
        state_dim = [nb_classes, window_length]
        
    batch_size = config['batch size']
    tau = config['tau']
    action_bound = 1.
    
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    model_save_path = get_model_path(dataset_name, framework, window_length, predictor_type, use_batch_norm, technical_indicators_flag)
    summary_path = get_result_path(dataset_name, framework, window_length, predictor_type, use_batch_norm, technical_indicators_flag)
    variable_scope = get_variable_scope(dataset_name, framework, window_length, predictor_type, use_batch_norm, technical_indicators_flag)
    
        
    actor_learning_rate = config['actor learning rate']
    critic_learning_rate = config['critic learning rate']
    
    with tf.variable_scope(variable_scope):
        sess = tf.Session()
        
        if(framework == 'DDPG'):
            actor = StockActor(sess, state_dim, action_dim, action_bound, actor_learning_rate, tau, batch_size,
                               predictor_type, use_batch_norm)
            critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=tau,
                                 learning_rate=critic_learning_rate, num_actor_vars=actor.get_num_trainable_vars(),
                                 predictor_type=predictor_type, use_batch_norm=use_batch_norm)
            model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer, 
                         log_return=log_return, config_file=config_file_path, 
                         model_save_path=model_save_path, summary_path=summary_path)
            model.initialize(load_weights=False)
            print('calling DDPG train')
            model.train()
            
        elif(framework =='TD3'):
            actor = TD3StockActor(sess, state_dim, action_dim, action_bound, actor_learning_rate, tau, batch_size,
                               predictor_type, use_batch_norm)
            critic = TD3StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=tau,
                                 learning_rate=critic_learning_rate, num_actor_vars=actor.get_num_trainable_vars(),
                                 predictor_type=predictor_type, use_batch_norm=use_batch_norm, 
                                 inp_actions=actor.scaled_out)
            model = TD3(env, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
                         log_return=log_return, config_file=config_file_path, 
                         model_save_path=model_save_path, summary_path=summary_path)
            model.initialize(load_weights=False)
            print('calling TD3 train')
            model.train()
            
