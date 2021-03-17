"""
 https://github.com/ankonzoid/LearningX/tree/master/classical_RL/multiarmed_bandit
 multiarmed_bandit.py  (author: Anson Wong / git: ankonzoid)
 We solve the multi-armed bandit problem using a classical epsilon-greedy
 agent with reward-average sampling as the estimate to action-value Q.
 This algorithm follows closely with the notation of Sutton's RL textbook.
 We set up bandit arms with fixed probability distribution of success,
 and receive stochastic rewards from each arm of +1 for success,
 and 0 reward for failure.
 The incremental update rule action-value Q for each (action a, reward r):
   n += 1
   Q(a) <- Q(a) + 1/n * (r - Q(a))
 where:
   n = number of times action "a" was performed
   Q(a) = value estimate of action "a"
   r(a) = reward of sampling action bandit (bandit) "a"
 Derivation of the Q incremental update rule:
   Q_{n+1}(a)
   = 1/n * (r_1(a) + r_2(a) + ... + r_n(a))
   = 1/n * ((n-1) * Q_n(a) + r_n(a))
   = 1/n * (n * Q_n(a) + r_n(a) - Q_n(a))
   = Q_n(a) + 1/n * (r_n(a) - Q_n(a))
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import json
from environment.portfolio import max_drawdown, sharpe, sortino
from stock_trading import returns_from_cumulative

class Environment:

    def __init__(self, models_df, model_names):
        self.models_df = models_df  
        self.model_names = model_names

    def step(self, action, row):
        selected_model = self.model_names[action]
        return row[selected_model]

class Agent:

    def __init__(self, nActions, eps):
        self.nActions = nActions
        self.eps = eps
        self.n = np.zeros(nActions, dtype=np.int) # action counts n(a)
        self.Q = np.zeros(nActions, dtype=np.float) # value Q(a)

    def update_Q(self, action, reward):
        # Update Q action-value given (action, reward)
        self.n[action] += 1
        self.Q[action] += (1.0/self.n[action]) * (reward - self.Q[action])

    def get_action(self):
        # Epsilon-greedy policy
        if np.random.random() < self.eps: # explore
            return np.random.randint(self.nActions)
        else: # exploit
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))


def experiment(models_df, model_names, eps):
    ''' 
    Start multi-armed bandit simulation

    Args:
        models_df: dataframe of model returns. Columns need to match up to model_names
        model_names: List of model names
        eps: Epsilon value
    '''
    # ENV
    env = Environment(models_df, model_names) # initialize arm probabilities
    # AGENT
    agent = Agent(len(env.model_names), eps)  # initialize agent
    actions, rewards, portfolio_values = [], [], []
    portfolio_value = 1
    for index, row in models_df.iterrows():
        action = agent.get_action() # sample policy
        #print("action: {}".format(action))
        reward = env.step(action, row) # take step + get reward
        #print("reward: {}".format(reward))
        portfolio_value = portfolio_value + (portfolio_value * reward)
        #print("portfolio_value: {}".format(portfolio_value))
        agent.update_Q(action, reward) # update Q
        actions.append(action)
        rewards.append(reward)
        portfolio_values.append(portfolio_value)
    return np.array(actions), np.array(rewards), np.array(portfolio_values)


def init_weights(columns):
        """ Set initial weights.
        :param m: Number of assets.
        """
        return np.zeros(len(columns))
    
def experiment_env(portfolio_env, weights_list, model_names, eps, window):
    ''' 
    Start multi-armed bandit simulation

    Args:
        models_df: dataframe of model returns. Columns need to match up to model_names
        model_names: List of model names
        eps: Epsilon value
    '''
        
    # ENV
    #env = Environment(portfolio_env, model_names) # initialize arm probabilities
    # AGENT
    agent = Agent(len(model_names), eps)  # initialize agent
    portfolio_choice_list, output_weights_list, return_list, portfolio_values_list = [], [], [], []
    portfolio_value = 1
    portfolio_values_list.append(portfolio_value)
            
    # Init Portfolio Environment
    _, _, info = portfolio_env.reset()    
        
    # Calculate number of iterations given window
    iterations = int(portfolio_env.sim.steps/window)
    index = 0
    # Iterations with window
    for i in range(iterations):
        # Get Action
        action = agent.get_action() # sample policy        
        
        # Retrieve windowed reward from Portfolio_Env
        window_rewards = []
        for w in range(window):
            # Use action throughout window by getting the weights
            weights = weights_list[action][index]
                
            # Set weights as numpy array
            if isinstance(weights, list):
                weights = np.array(weights)
            if isinstance(weights, pd.core.series.Series):
                weights = weights.to_numpy()
            
            # step forward in Portfolio Environment
            _, _, reward, done, info, _, _return = portfolio_env.step(weights)
            
            # Calculate new portfolio value
            portfolio_value = portfolio_value + (portfolio_value * _return)
            
            # Append to lists
            portfolio_choice_list.append(action)
            output_weights_list.append(weights)
            return_list.append(_return)
            portfolio_values_list.append(portfolio_value)
            window_rewards.append(reward)
            
            index = index + 1
        
        # Calculate reward from window_rewards
        reward = np.mean(window_rewards) # Mean returns
        agent.update_Q(action, reward) # update Q
    
    # Last bit (must be less than window) (using same action)
    while not done:
        # Use action throughout window by getting the weights
        weights = weights_list[action][index]
            
        # Set weights as numpy array
        if isinstance(weights, list):
            weights = np.array(weights)
        if isinstance(weights, pd.core.series.Series):
            weights = weights.to_numpy()
        
        # step forward in Portfolio Environment
        _, _, reward, done, _, _, _return = portfolio_env.step(weights)
            
        # Calculate new portfolio value
        portfolio_value = portfolio_value + (portfolio_value * _return)
        
        # Append to lists
        portfolio_choice_list.append(action)
        output_weights_list.append(weights)
        return_list.append(_return)
        portfolio_values_list.append(portfolio_value)
        
        index = index + 1
        
    return np.array(portfolio_choice_list), np.array(output_weights_list), np.array(return_list), np.array(portfolio_values_list)


def execute_various_epsilon(df, model_names, num, seed = 0):
    ''' 
    Execute Epsilon Greedy Bandit algorithm 
    with a variety of epsilon to find best performing epsilon

    Args:
        df: dataframe of model returns. Columns need to match up to model_names
        model_names: List of model names
        num: number of different epsilon values to be generated from 0 to 1
        seed: random seed
    '''
    np.random.seed(seed)
    # Check if model_names are in dataframe
    for model_name in model_names:
        if model_name not in df.columns:
            sys.exit("Model Name ({}) is missing".format(model_name))
    
    _df = pd.DataFrame()
    
    # Split by 100 (num) different epsilon values
    for i in range(num):
        eps = round(i * (1/num), 3)
        _, _, portfolio_values = experiment(df, model_names, eps)
        
        returns = returns_from_cumulative(portfolio_values)
                
        _df = _df.append({'eps': eps,
                    'Average Daily Yield (%)': round(float(np.mean(returns))*100, 4),
                    'Sharpe Ratio (%)': round(sharpe(returns)*100, 4),
                    'Sortino Ratio (%)': round(sortino(returns)*100, 4),
                    'Maximum Drawdown (%)': round(max_drawdown(portfolio_values)*100, 4),
                    #'Final Portfolio Value': round(float(portfolio_values[-1]), 3)
                    }, ignore_index=True)
        print("eps: {} - Sharpe Ratio (%): {}".format(eps, round(sharpe(returns)*100, 4)))
        
    return _df


def execute_various_epsilon_env(portfolio_env, weights_list, model_names, num, seed = 0, window = 1):
    ''' 
    Execute Epsilon Greedy Bandit algorithm 
    with a variety of epsilon to find best performing epsilon

    Args:
        df: dataframe of model returns. Columns need to match up to model_names
        model_names: List of model names
        num: number of different epsilon values to be generated from 0 to 1
        seed: random seed
    '''
    np.random.seed(seed)
    
    assert len(model_names) == len(weights_list), 'Number of models is to be equal to models in weights list'
    assert portfolio_env.sim.steps == len(weights_list[0])-1, 'Number of steps is to be equal to weights list'
    
    _df = pd.DataFrame()
    
    # Split by 100 (num) different epsilon values
    for i in range(num):
        eps = round(i * (1/num), 3)
        _, _, _, portfolio_values = experiment_env(portfolio_env, weights_list, model_names, eps, window)
        
        returns = returns_from_cumulative(portfolio_values)
                
        _df = _df.append({'eps': eps,
                    'Average Daily Yield (%)': round(float(np.mean(returns))*100, 4),
                    'Sharpe Ratio (%)': round(sharpe(returns)*100, 4),
                    'Sortino Ratio (%)': round(sortino(returns)*100, 4),
                    'Maximum Drawdown (%)': round(max_drawdown(portfolio_values)*100, 4),
                    #'Final Portfolio Value': round(float(portfolio_values[-1]), 3)
                    }, ignore_index=True)
        print("eps: {} - Sharpe Ratio (%): {}".format(eps, round(sharpe(returns)*100, 4)))
        
    return _df


def execute(df, model_names, eps, seed = 0):
    ''' 
    Execute Epsilon Greedy Bandit algorithm 

    Args:
        df: dataframe of model returns. Columns need to match up to model_names
        model_names: List of model names
        eps: Epsilon value
        seed: random seed
    '''
    np.random.seed(seed)
    # Settings
    N_steps = len(df.index) # number of steps (episodes)
    print("N_steps: {}".format(N_steps))
    
    #print("model_names: {}".format(model_names))
    # Let's just use the last two models
    #model_names = model_names[:2]
    print("model_names: {}".format(model_names))
    
    # Run multi-armed bandit experiments
    print("Running multi-armed bandits with nActions = {}, eps = {}".format(len(model_names), eps))
    actions, rewards, portfolio_values = experiment(df, model_names, eps)  # perform experiment
    
    return actions, rewards, portfolio_values
    

def execute_env(portfolio_env, weights_list, model_names, eps, seed = 0, window = 1):
    ''' 
    Execute Epsilon Greedy Bandit algorithm 

    Args:
        df: dataframe of model returns. Columns need to match up to model_names
        model_names: List of model names
        eps: Epsilon value
        seed: random seed
    '''
    np.random.seed(seed)
        
    assert len(model_names) == len(weights_list), 'Number of models is to be equal to models in weights list'
    assert portfolio_env.sim.steps == len(weights_list[0])-1, 'Number of steps is to be equal to weights list'

    print("model_names: {}".format(model_names))
    
    # Run multi-armed bandit experiments
    print("Running multi-armed bandits with nActions = {}, eps = {}".format(len(model_names), eps))
    portfolio_choice_list, weights_list, return_list, portfolio_values_list = experiment_env(portfolio_env, weights_list, model_names, eps, window)  # perform experiment
    
    return portfolio_choice_list, weights_list, return_list, portfolio_values_list



