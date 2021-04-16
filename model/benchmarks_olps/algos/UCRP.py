#-*- coding:utf-8 -*-
'''
@Author: Louis Liang
@time:2018/9/15 9:13
'''
import numpy as np

class UCRP:
    def __init__(self):
        self.a_dim = 0

    def predict(self, s, a):
        #print("len(a[0]): " + str(len(a[0])))
        weights = np.ones(len(a[0])-1)/(len(a[0])-1)
        #print("Weights: " + str(weights))
        #weights.insert(0, weights)
        weights = np.insert(weights, 0, 0)

        weights = weights[None, :]
        #print("Weights: " + str(weights))
        return weights