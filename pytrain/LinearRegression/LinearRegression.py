#
# Linear Regression
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

import numpy as np
from pytrain.lib import convert
import math
import time
import random
import sys

class LinearRegression:

    def __init__(self, mat_data, label_data):
        self.mat_data = convert.list2npfloat(mat_data)
        self.label_data = convert.list2npfloat(label_data)
        
        self.out_bit = len(label_data[0])
        self.mat_w =  np.array([ [ (random.random() * 0.00001 + sys.float_info.epsilon)\
                            for i in range(len(mat_data[0]))] \
                                for j in range(self.out_bit) ], dtype = np.float64)

        self.mat_w0 = np.array([ random.random() * 0.00001 + sys.float_info.epsilon\
                            for i in range(self.out_bit) ], dtype = np.float64)

    #
    # Description of differential equation
    #
    # k = w0 + w1 x1 + w2 x2 + w3 x3 + .. + wn xn
    # J(k) = (y - k1)^2 + (y - k2)^2 + .. + (y - kn)^2
    #
    # dJ/dw1 = dJ/dk * dk/dw1 = 
    #                    - 2 * (y - k1) * x1_1
    #                    - 2 * (y - k2) * x1_2 
    #                    - 2 * (y - k3) * x1_3
    #                    ...
    # 
    # UPDATE w1 with gradient
    # w1 = w1 - lr * dJ/dw1
    #
    # dJ/dw0 = dJ/dk0 * dk0/dw0 = 
    #                   - 2 * (y - k1) * 1
    #                   - 2 * (y - k2) * 1
    #                   ...
    #
    # w0 = w0 - lr * dJ/w0
    #

    def batch_update_w(self, out_bit_index, data, label):
        w = self.mat_w[out_bit_index]
        w0 = self.mat_w0[out_bit_index]
        tiled_w0 = np.tile(w0,(len(data)))
        k = (w * data).sum(axis=1) + tiled_w0
        gd = (k - label.T[out_bit_index])
        # dJ_dw is gradient of J(w) function
        dJ_dw = (gd * data.T).sum(axis=1)/len(data)
        dJ_dw0  = gd.sum(axis=0)/len(data)
        w = w - (dJ_dw * self.lr)
        w0 = w0 - (dJ_dw0 * self.lr)
        self.mat_w[out_bit_index] = w
        self.mat_w0[out_bit_index] = w0

    def fit(self, lr, epoch, batch_size):
        self.lr = lr
        self.epoch = epoch
        start = 0
        end = batch_size
        datalen = len(self.mat_data)
        for ep in range(epoch):
            start = 0
            end = batch_size
            while start < datalen :
                for i in range(self.out_bit):
                    self.batch_update_w(i, self.mat_data[start:end],\
                        self.label_data[start:end])
                start = end
                end += batch_size

    def predict(self, array_input):
        array_input = convert.list2npfloat(array_input)
        return (array_input * self.mat_w).sum(axis=1) \
                + self.mat_w0

