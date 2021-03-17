from ..algo import Algo
from .. import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CRP(Algo):
    """ Constant rebalanced portfolio = use fixed weights all the time. Uniform weights
    are commonly used as a benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, b=None):
        """
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super().__init__()
        self.b = b


    def step(self, x, w1, history):
        # init b to default if necessary
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        return self.b

    def weights(self, env, min_history=None):
        # Get full array of close prices
        if isinstance(env, pd.DataFrame):   # When inherited
            X = env
        else:
            X = env.close_df

        if self.b is None:
            b = X * 0 + 1
            #b.loc[:, 'CASH'] = 0
            b = b.div(b.sum(axis=1), axis=0)
            return b
        elif self.b.ndim == 1:
            return np.repeat([self.b], X.shape[0], axis=0)
        else:
            return self.b

