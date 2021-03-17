# -*- coding: utf-8 -*-
from ..algo import Algo
from .. import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EG(Algo):
    """ Exponentiated Gradient (EG) algorithm by Helmbold et al.

    Reference:
        Helmbold, David P., et al.
        "On‐Line Portfolio Selection Using Multiplicative Updates."
        Mathematical Finance 8.4 (1998): 325-347.
    """

    def __init__(self, eta=0.05):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super(EG, self).__init__()
        self.eta = eta


    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m


    def step(self, x, last_b, history):
        #print("x: {}".format(x))
        #print("last_b: {}".format(last_b))
        mult = np.exp(self.eta * x / sum(x * last_b)) 
        #print("mult: {}".format(mult))
        b = last_b * mult    
        #print("b: {}".format(b))
        result = b / sum(b)
        #print("result: {}".format(result))
        
        return result


if __name__ == '__main__':
    data = tools.dataset('nyse_n')
    tools.quickrun(EG(eta=0.5), data)
