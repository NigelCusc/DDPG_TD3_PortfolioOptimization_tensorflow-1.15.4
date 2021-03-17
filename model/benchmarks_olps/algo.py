import sys
import numpy as np
import pandas as pd
import itertools
import logging
import inspect
import copy
from .result import AlgoResult, ListResult
from scipy.special import comb
from . import tools
from scipy.special import comb
import math


class Algo(object):
    """ Base class for algorithm calculating weights for online portfolio.
    You have to subclass either step method to calculate weights sequentially
    or weights method, which does it at once. weights method might be useful
    for better performance when using matrix calculation, but be careful about
    look-ahead bias.

    Upper case letters stand for matrix and lower case for vectors (such as
    B and b for weights).
    """

    # if true, replace missing values by last values
    REPLACE_MISSING = False

    # type of prices going into weights or step function
    #    ratio:  pt / pt-1
    #    log:    log(pt / pt-1)
    #    raw:    pt
    PRICE_TYPE = 'ratio'

    def __init__(self, min_history=None, frequency=1):
        """ Subclass to define algo specific parameters here.
        :param min_history: If not None, use initial weights for first min_window days. Use
            this if the algo needs some history for proper parameter estimation.
        :param frequency: algorithm should trade every `frequency` periods
        """
        self.min_history = min_history or 0
        self.frequency = frequency

    def init_weights(self, columns):
        """ Set initial weights.
        :param m: Number of assets.
        """
        return np.zeros(len(columns))

    def init_step(self, X):
        """ Called before step method. Use to initialize persistent variables.
        :param X: Entire stock returns history.
        """
        pass

    def step(self, x, w1, history=None):
        """ Calculate new portfolio weights. If history parameter is omited, step
        method gets passed just parameters `x` and `last_b`. This significantly
        increases performance.
        :param x: Last returns.
        :param last_b: Last weights.
        :param history: All returns up to now. You can omit this parameter to increase
            performance.
        """
        raise NotImplementedError('Subclass must implement this!')

    def weights(self, env, min_history=None):
        """ Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk. """
        # Init
        previous_observation, previous_observation_ti, info = env.reset()

        # Init step if exists
        w1 = self.init_weights([0] + env.abbreviation)

        # Initializing returning variables
        weights_array = []

        # Init step if exists
        self.init_step(info['price_history'])
        
        done = False
        while not done:
            # Format weight and prices
            w1_series = pd.Series(w1, info['price_history'].columns)
            p_series = pd.Series(info['close'][0], info['price_history'].columns)

            # Algorithm Step (predict and update, to generate new weights)
            w2 = self.step(p_series, w1_series, info['price_history'])

            # Set weights as numpy array
            if isinstance(w2, list):
                w2 = np.array(w2)
            if isinstance(w2, pd.core.series.Series):
                w2 = w2.to_numpy()

            # step forward
            observation, observation_ti, reward, done, info, weights, _ = env.step(w2)

            # Append to result arrays
            weights_array.append(w2)
            w1 = w2.copy()

        return weights_array

    def _split_index(self, ix, nr_chunks, freq):
        """ Split index into chunks so that each chunk except of the last has length
        divisible by freq. """
        chunksize = int(len(ix) / freq / nr_chunks + 1) * freq
        return [ix[i*chunksize:(i+1)*chunksize] for i in range(int(len(ix) / chunksize + 1))]


    def run(self, env, initial_investment=1, n_jobs=1):
        """ Run algorithm and get weights.
        """
        # Adjust ENV data based on PRICE_TYPE and REPLACE_MISSING!
        env.format_data(self.PRICE_TYPE, self.REPLACE_MISSING)

        # Get Weights
        weights_array = self.weights(env)
        if isinstance(weights_array, pd.DataFrame):
            weights_array = weights_array.values.tolist()
        
        print("len(weights_array): {}".format(len(weights_array)))
        
        rewards = []
        portfolio_values = []
        weights_list = []
        dates_list = []

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
            i += 1
        
        #print("i: {}".format(i))
        #print("weights: {}".format(len(weights_array)))

        return rewards, portfolio_values, weights_list, dates_list


    def run_subsets(self, S, r, generator=False):
        """ Run algorithm on all stock subsets of length r. Note that number of such tests can be
        very large.
        :param S: stock prices
        :param r: number of stocks in a subset
        :param generator: yield results
        """
        def subset_generator():
            total_subsets = comb(S.shape[1], r)

            for i, S_sub in enumerate(tools.combinations(S, r)):
                # run algorithm on given subset
                result = self.run(S_sub, log_progress=False)
                name = ', '.join(S_sub.columns.astype(str))

                # log progress by 1 pcts
                tools.log_progress(i, total_subsets, by=1)

                yield result, name
            raise StopIteration

        if generator:
            return subset_generator()
        else:
            results = []
            names = []
            for result, name in subset_generator():
                results.append(result)
                names.append(name)
            return ListResult(results, names)

    @classmethod
    def _convert_prices(self, S, method, replace_missing=False):
        """ Convert prices to format suitable for weight or step function.
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

    @classmethod
    def run_combination(cls, S, **kwargs):
        """ Get equity of algo using all combinations of parameters. All
        values in lists specified in kwargs will be optimized. Other types
        will be passed as they are to algo __init__ (like numbers, strings,
        tuples).
        Return ListResult object, which is basically a wrapper of list of AlgoResult objects.
        It is possible to pass ListResult to Algo or run_combination again
        to get AlgoResult. This is useful for chaining of Algos.

        Example:
            S = ...load data...
            list_results = Anticor.run_combination(S, alpha=[0.01, 0.1, 1.])
            result = CRP().run(list_results)

        :param S: Stock prices.
        :param kwargs: Additional arguments to algo.
        :param n_jobs: Use multiprocessing (-1 = use all cores). Use all cores by default.
        """
        if isinstance(S, ListResult):
            S = S.to_dataframe()

        n_jobs = kwargs.pop('n_jobs', -1)

        # extract simple parameters
        simple_params = {k: kwargs.pop(k) for k in tuple(kwargs.keys()) if not isinstance(kwargs[k], list)}

        # iterate over all combinations
        names = []
        params_to_try = []
        for seq in itertools.product(*kwargs.values()):
            params = dict(zip(kwargs.keys(), seq))

            # run algo
            all_params = dict(list(params.items()) + list(simple_params.items()))
            params_to_try.append(all_params)

            # create name with format param:value
            name = ','.join([str(k) + '=' + str(v) for k, v in params.items()])
            names.append(name)

        # try all combinations in parallel
        with tools.mp_pool(n_jobs) as pool:
            results = pool.map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])
        results = map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])

        return ListResult(results, names)

    def copy(self):
        return copy.deepcopy(self)


def _parallel_weights(tuple_args):
    self, X, min_history, log_progress = tuple_args
    try:
        return self.weights(X, min_history=min_history, log_progress=log_progress)
    except TypeError:   # weights are missing log_progress parameter
        return self.weights(X, min_history=min_history)


def _run_algo_params(tuple_args):
    S, cls, params = tuple_args
    logging.debug('Run combination of parameters: {}'.format(params))
    return cls(**params).run(S)
