"""
Modified from https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
"""
from __future__ import print_function
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import gym.spaces
eps = 1e-8


def random_shift(x, fraction):
    """ Apply a random shift to a pandas series. """
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def scale_to_start(x):
    """ Scale pandas series so that it starts at one. """
    x = (x + eps) / (x[0] + eps)
    return x


def sharpe(returns, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe """
    if isinstance(returns, list): 
        returns = np.array(returns)
    return (np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def sortino(returns, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sortino """
    if isinstance(returns, list): 
        returns = np.array(returns)
    return (np.mean(returns - rfr + eps)) / np.std([r for r in (returns - rfr + eps) if r < 0])


def max_drawdown(portfolio_value):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    mdd = 0
    peak = 0
    for i in list(range(len(portfolio_value))): 
        # Check if possible peak
        if(portfolio_value[i] > peak):
            # Max drawdown assuming this is the peak
            peak = portfolio_value[i]
            trough = min(portfolio_value[i:])
            if(peak > trough):
                d = abs((trough - peak) / peak)
                if(d > mdd):
                    mdd = d
    return mdd


def create_close_dataframe(history, abbreviation, date_list):
    """
    Args:
        history: numpy array with full data (open, high, low, close, volume)
        abbreviation: list of Assets
        date_list: list of dates corresponding to history parameter

    Returns: Pandas Dataframe consisting of all the Close Prices
    """
    # Check if just close is fed or full dataset
    if history.shape[2] == 4:
        # Get Close
        history_close = history[:, :, 3]
    elif history.shape[2] == 2:
        # Assume the one sent is the close price
        history_close = history[:, :, 1]
    else:
        print("Invalid History Fomrat. Must be (x, y) or (x, y, 4)")
        return
    
    # Convert to Pandas
    transposed_target_history = history_close.transpose()
    date_list = [date_list[i] for i in range(transposed_target_history.shape[0])]

    df = pd.DataFrame(data=transposed_target_history,
                      index=date_list,
                      columns=abbreviation)

    return df

def convert_prices(S, method, replace_missing=False):
    """
    S: data in pandas dataframe format
    Convert prices to format suitable for weight or step function.
    Available price types are:
        ratio:  pt / pt_1
        log:    log(pt / pt_1)
        raw:    pt (normalized to start with 1)
    """
    if method == 'raw':
        # normalize prices so that they start with 1.
        r = {}
        for name, s in S.items():
            init_val = s.loc[s.first_valid_index()]
            r[name] = s / init_val
        X = pd.DataFrame(r)

        if replace_missing:
            X.iloc[0] = 1.
            X = X.fillna(method='ffill')

        return X

    elif method == 'absolute':
        return S

    elif method in ('ratio', 'log'):
        # be careful about NaN values
        X = S / S.shift(1).fillna(method='ffill')
        for name, s in X.iteritems():
            X[name].iloc[s.index.get_loc(s.first_valid_index()) - 1] = 1.

        if replace_missing:
            X = X.fillna(1.)

        return np.log(X) if method == 'log' else X

    else:
        raise ValueError('invalid price conversion method')


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history, abbreviation, dates, steps=730, window_length=50, start_idx=0,
                 start_date=None, end_date=None, technical_indicators_flag=False, technical_indicator_history=None):
        """

        Args:
            history: (num_stocks, timestamp, 5) open, high, low, close, volume
            abbreviation: a list of length num_stocks with assets name
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        assert history.shape[0] == len(abbreviation), 'Number of stock is not consistent'
        import copy
        self.step = 0
        self.steps = steps + 1
        self.window_length = window_length
        self.start_idx = start_idx
        self.start_date = start_date
        self.end_date = end_date
        self.dates = dates

        # make immutable class
        self._data = history.copy()  # all data
        self.asset_names = copy.copy(abbreviation)
        self.data = self._data
        self.idx = np.random.randint(low=self.window_length, high=self._data.shape[1] - self.steps)
        
        self.technical_indicators_flag = technical_indicators_flag
        
        if self.technical_indicators_flag:
            self._technical_indicator_data = technical_indicator_history.copy()
            self.technical_indicator_data = self._technical_indicator_data
            

    def _step(self):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes. Normalize could be critical here
        self.step += 1

        # normalize obs with open price
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()
        
        ti_obs = None
        # Include Technical indicator data
        if self.technical_indicators_flag:
            ti_obs = self.technical_indicator_data[:, self.step + self.window_length, :].copy()

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()

        done = self.step >= self.steps
        return obs, ti_obs, done, ground_truth_obs
    
    def reset(self):
        self.step = 0

        # get data for this episode, each episode might be different.
        if self.start_date is None:
            if self.end_date is None:
                self.idx = np.random.randint(low=self.window_length, high=self._data.shape[1] - self.steps)
            else:
                self.idx = np.random.randint(low=self.window_length, high=self.dates.index(self.end_date) - self.steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = self.dates.index(self.start_date) - self.start_idx
            assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'

        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :]
        # apply augmentation?
        self.data = data        
        
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()
        
        ti_obs = None
        if self.technical_indicators_flag:
            technical_indicator_data = self._technical_indicator_data[:, self.idx - self.window_length:self.idx + self.steps + 1, :]
            self.technical_indicator_data = technical_indicator_data
            
            ti_obs = self.technical_indicator_data[:, self.step + self.window_length, :].copy()
        
        return obs, ti_obs,\
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=list(), steps=730, trading_cost=0.0025, time_cost=0.0):
        self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.p0 = 0
        self.infos = []

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        #p0 = self.p0

        dw1 = (y1 * w1) / (np.dot(y1, w1) + eps)  # (eq7) weights evolve into

        mu1 = self.cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio

        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = self.p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding
        
        rho1 = p1 / self.p0 - 1  # rate of returns
        
        r1 = np.log((p1 + eps) / (self.p0 + eps))  # log rate of return
        
        reward = r1 / self.steps * 1000  # (22) average logarithmic accumulated return
        
        # PRINT
        #print("Current Portfolio Value: {}".format(self.p0))
        #print("Weights: {}".format(np.around(w1, decimals=2)))
        #print("DW1: {}".format(np.around(dw1, decimals=2)))
        #print("Prices: {}".format(y1))
        #print("Cost to Change Portfolio: {}".format(mu1))
        #print("New Portfolio Value: {}".format(p1))
        #print("Reward: {}".format(reward))
        #print("Actual Portfolio Value: {}".format(subset_portfolio_df.iloc[i+2].values, subset_portfolio_df.index[i+2]))
        #print("-------------------------------------------------------")
        
        # remember for next step
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        
        return reward, info, done, rho1

    def reset(self):
        self.infos = []
        self.p0 = 1.0


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 history,
                 abbreviation,
                 date_list,
                 start_date=None,
                 end_date=None,
                 steps=730,  # 2 years
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0,
                 olps=False,
                 technical_indicators_flag=False,
                 technical_indicator_history=None
                 ):
        """
        An environment for financial portfolio management.
        Params:
            history - asset dataframe
            abbreviation - asset names list
            date_list - full date list
            start_date - starting date for subset
            end_date - ending date for subset
            steps - instead of end date the number of steps can be utilised
            trading_cost - the transaction cost
            time cost - additional charges added to holding assets
            window_length - the lookback used to create the state
            start - the number of days from the start_date
            olps - includes functionality to allow for on-line portfolio selection algorithms
            technical_indicators_flag - includes technical indicators inside observation state
        """

        self.dates = date_list
        self.start_date = start_date
        self.end_date = end_date
        # Get the number of steps
        if start_date is not None and end_date is not None:
            steps = date_list.index(self.end_date) - date_list.index(self.start_date)
            
        self.window_length = window_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx
        self.df_portfolio_performance = None

        # Check number of features (Accepting 2 and 4 for now)
        self.feature_length = history.shape[2]

        self.src = DataGenerator(history, abbreviation, date_list, steps=steps, window_length=window_length,
                                 start_idx=start_idx, start_date=start_date, end_date=end_date, 
                                 technical_indicators_flag=technical_indicators_flag, technical_indicator_history=technical_indicator_history)

        self.sim = PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.src.asset_names) + 1,), dtype=np.float32)  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(abbreviation), window_length,
                                                                                 history.shape[-1]), dtype=np.float32)
        self.infos = []

        # For Portfolio Selection
        self.history = history
        self.abbreviation = abbreviation
        self.close_df = create_close_dataframe(history, abbreviation, date_list) # Dataframe that holds the Close data in Different formats
        
        if self.start_date is not None and self.end_date is not None:        
            self.subset_dates = date_list[date_list.index(self.start_date) : date_list.index(self.end_date)+1]
            self.close_df_subset = self.close_df[self.close_df.index.isin(self.subset_dates)] # Dataframe that holds subset of the Close data in Different formats
            
        self.olps = olps
        self.technical_indicators_flag = technical_indicators_flag
        
    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names) + 1,)
        )

        # normalise just in case
        action = np.clip(action, 0, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights = weights.astype('float')
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        observation, ti_observation, done1, ground_truth_obs = self.src._step()           
        
        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        
        if self.technical_indicators_flag:
            cash_ti_observation = np.ones((1, ti_observation.shape[1]))
            ti_observation = np.concatenate((cash_ti_observation, ti_observation), axis=0)

        # relative price vector of last observation day
        # Based on Feature length
        if self.feature_length == 2:    # Using Close only
            # (close / prev_close)
            close_price_vector = observation[:, -1, 1]
            prev_price_vector = observation[:, -1, 0]
            y1 = close_price_vector / prev_price_vector
        else:
            # (close / open)
            close_price_vector = observation[:, -1, 3]
            open_price_vector = observation[:, -1, 0]
            y1 = close_price_vector / open_price_vector
        reward, info, done2, _return = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.dates[self.start_idx + self.src.idx + self.src.step]
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs
        if self.olps:
            _date = self.dates[self.start_idx + self.src.idx + self.src.step - 1]
            info['close'] = self.close_df[self.close_df.index == _date].values.tolist()
            info['price_history'] = self.close_df[:_date] #self.close_df[self.close_df.index.to_series().between('', _date)]

        self.infos.append(info)        

        return observation, ti_observation, reward, done1 or done2, info, weights, _return
    
    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ti_observation, ground_truth_obs = self.src.reset()    
        
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)    
        
        if self.technical_indicators_flag:        
            cash_ti_observation = np.ones((1, ti_observation.shape[1]))
            ti_observation = np.concatenate((cash_ti_observation, ti_observation), axis=0)
        
        info = {}
        info['next_obs'] = ground_truth_obs
        if self.olps:
            _date = self.dates[self.start_idx + self.src.idx + self.src.step - 1]
            info['close'] = self.close_df[self.close_df.index == _date].values.tolist()
            info['price_history'] = self.close_df[:_date]
            
            
        return observation, ti_observation, info

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == 'ansi':
            print(self.infos[-1])
        elif mode == 'human':
            return self.plot()
            
    def render(self, mode='human', close=False):
        return self._render(mode='human', close=False)

    # Adjust ENV data based on PRICE_TYPE and REPLACE_MISSING
    def _format_data(self, PRICE_TYPE, REPLACE_MISSING):
        """
        REPLACE_MISSING: if true, replace missing values by last values
        PRICE_TYPE: type of prices going into weights or step function
            ratio:  pt / pt-1
            log:    log(pt / pt-1)
            raw:    pt
        """
        # Close Price DataFrame for on-line Portfolio Selection
        self.close_df = create_close_dataframe(self.history, self.abbreviation, self.dates)
        # Add empty asset in the beginning
        self.close_df.insert(0, 'Unassigned', 1)
        self.close_df = convert_prices(self.close_df, PRICE_TYPE, REPLACE_MISSING)
        # Subset with just test dates
        self.close_df_subset = self.close_df[self.close_df.index.isin(self.subset_dates)] # Dataframe that holds subset of the Close data in Different formats
        

    def format_data(self, PRICE_TYPE, REPLACE_MISSING):
        return self._format_data(PRICE_TYPE, REPLACE_MISSING)

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info['date'] = pd.to_datetime(df_info['date'], format='%d/%m/%Y')
        df_info.set_index('date', inplace=True)
        #mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        #title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        title = 'sharpe_ratio={: 2.4f}'.format(sharpe_ratio)        
        return df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)
        

class MultiActionPortfolioEnv(PortfolioEnv):
    def __init__(self,
                 history,
                 abbreviation,
                 model_names,
                 date_list,
                 start_date=None,
                 end_date=None,
                 steps=730,  # 2 years
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0,
                 technical_indicators_flag=False,
                 technical_indicator_history=None
                 ):
        super(MultiActionPortfolioEnv, self).__init__(history, abbreviation, date_list, start_date, end_date, steps,
                                                      trading_cost, time_cost, window_length, start_idx, 
                                                      technical_indicators_flag=technical_indicators_flag, 
                                                      technical_indicator_history=technical_indicator_history)    
        
        self.model_names = model_names
        # need to create a simulator for each model
        self.sim = [PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)for _ in range(len(self.model_names))]
        self.infos = []
        self.technical_indicators_flag = technical_indicators_flag

    def _step(self, action):
        """ Step the environment by a vector of actions

        Args:
            action: (num_models, num_stocks + 1)

        Returns:

        """
        assert action.ndim == 2, 'Action must be a two dimensional array with shape (num_models, num_stocks + 1)'
        assert action.shape[1] == len(self.sim[0].asset_names) + 1
        assert action.shape[0] == len(self.model_names)
        # normalise just in case
        action = np.clip(action, 0, 1)
        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (np.sum(weights, axis=1, keepdims=True) + eps)
        # so if weights are all zeros we normalise to [1,0...]
        weights[:, 0] += np.clip(1 - np.sum(weights, axis=1), 0, 1)
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(np.sum(weights, axis=1), np.ones(shape=(weights.shape[0])), 3,
                                       err_msg='weights should sum to 1. action="%s"' % weights)
        
        observation, ti_observation, done1, ground_truth_obs = self.src._step()  
        
        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)        
        
        if self.technical_indicators_flag:
            cash_ti_observation = np.ones((1, ti_observation.shape[1]))
            ti_observation = np.concatenate((cash_ti_observation, ti_observation), axis=0)

        # relative price vector of last observation day
        if self.feature_length == 2:  # Using Close only
            # (close / prev_close)
            close_price_vector = observation[:, -1, 1]
            prev_price_vector = observation[:, -1, 0]
            y1 = close_price_vector / prev_price_vector
        else:
            # (close / open)
            close_price_vector = observation[:, -1, 3]
            open_price_vector = observation[:, -1, 0]
            y1 = close_price_vector / open_price_vector

        rewards = np.empty(shape=(weights.shape[0]))
        info = {}
        dones = np.empty(shape=(weights.shape[0]), dtype=bool)
        for i in range(weights.shape[0]):
            reward, current_info, done2, _ = self.sim[i]._step(weights[i], y1)
            rewards[i] = reward
            info[self.model_names[i]] = current_info['portfolio_value']
            info['return'] = current_info['return']
            dones[i] = done2

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.dates[self.start_idx + self.src.idx + self.src.step]
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs
        #_date = self.dates[self.start_idx + self.src.idx + self.src.step - 1]
        #info['close'] = self.close_df[self.close_df.index == _date].values.tolist()
        #info['price_history'] = self.close_df[:_date]

        self.infos.append(info)

        return observation, ti_observation, rewards, np.all(dones) or done1, info, weights

    def _reset(self):
        self.infos = []
        for sim in self.sim:
            sim.reset()
        observation, ti_observation, ground_truth_obs = self.src.reset()      
            
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)        
        
        if self.technical_indicators_flag:
            cash_ti_observation = np.ones((1, ti_observation.shape[1]))
            ti_observation = np.concatenate((cash_ti_observation, ti_observation), axis=0)
        
        info = {}
        info['next_obs'] = ground_truth_obs
                
        return observation, ti_observation, info

    def plot(self):
        df_info = pd.DataFrame(self.infos)
        fig=plt.gcf()
        title = 'Trading Performance of Models'
        df_info['date'] = pd.to_datetime(df_info['date'], format='%d/%m/%Y')
        df_info.set_index('date', inplace=True)
        #print(df_info[self.model_names + ['market_value']])
        df_info[self.model_names + ['market_value']].plot(title=title, fig=fig, rot=30)
        return df_info[self.model_names + ['market_value']]
