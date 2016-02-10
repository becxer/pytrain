#
# Logistic Regression
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
seterr(all='raise')
from pytrain.lib import convert
import math
import time
import random

class LogisticRegression:

    def __init__(self, mat_data, label_data):
        if mat_data.__class__.__name__ != 'ndarray':
            mat_data = convert.mat2arr(mat_data)
        if label_data.__class__.__name__ != 'ndarray':
            label_data = convert.mat2arr(label_data)
        self.mat_data = mat_data
        self.label_data = label_data
        self.out_bit = len(label_data[0])
        self.mat_w =  [ [random.random() for i in range(len(mat_data[0]))] \
                        for j in range(self.out_bit) ]
        self.mat_w0 = [random.random() for i in range(self.out_bit) ]

    def sigmoid(self,k):
        return 1.0 / ( 1.0 + math.exp(-k))

    def dsigmoid(self,k):
        return (1.0 - self.sigmoid(k)) * self.sigmoid(k)

    #
    # Description of differential equation
    #
    # k = w0 + w1 x1 + w2 x2 + w3 x3 + .. + wn xn
    # sigmoid(k) = 1 / ( 1 + e^(-1 * k))
    # J(k) = (y - sigmoid(k1))^2 + (y - sigmoid(k2))^2 + .. + (y - sigmoid(kn))^2
    # dsigmoid(k)/dk = (1 - sigmoid(k)) * sigmoid(k)
    # dJ/dw1 = dJ/dsigmoid(k) * dsigmoid(k)/dk * dk/dw1 = 
    #                    - 2 * (y - sigmoid(k1)) * (1 - sigmoid(k1)) * sigmoid(k1) * x1_1
    #                    - 2 * (y - sigmoid(k2)) * (1 - sigmoid(k2)) * sigmoid(k2) * x1_2
    #                    - 2 * (y - sigmoid(k3)) * (1 - sigmoid(k3)) * sigmoid(k3) * x1_3
    #                    ...
    # 
    # UPDATE w1 with gradient
    # w1 = w1 - lr * dJ/dw1
    #
    # dJ/dw0 = dJ/dk0 * dk0/dw0 = 
    #                    - 2 * (y - sigmoid(k1)) * (1 - sigmoid(k1)) * sigmoid(k1) * 1
    #                    - 2 * (y - sigmoid(k2)) * (1 - sigmoid(k2)) * sigmoid(k2) * 1
    #                    - 2 * (y - sigmoid(k3)) * (1 - sigmoid(k3)) * sigmoid(k3) * 1
    #                   ...
    #
    # w0 = w0 - lr * dJ/w0
    #

    def batch_update_w(self, out_bit_index, data, label):
        w = self.mat_w[out_bit_index]
        w0 = self.mat_w0[out_bit_index]
        tiled_w0 = tile(w0,(len(data)))
        k = (w * data).sum(axis=1) + tiled_w0
        sig_k = map(self.sigmoid,k)
        dsig_k = map(self.dsigmoid,k)
        gd = (label.T[out_bit_index] - sig_k) * dsig_k
        
        # dJ_dw is gradient of J(w) function
        dJ_dw = (gd * data.T).sum(axis=1) * -2
        dJ_dw0  = gd.sum(axis=0) * -2
        w = w - (dJ_dw * self.lr)
        w0 = w0 - (dJ_dw0 * self.lr)
        self.mat_w[out_bit_index] = w
        self.mat_w0[out_bit_index] = w0

    def fit(self, lr, epoch, stoc):
        self.lr = lr
        self.epoch = epoch
        start = 0
        end = stoc
        datalen = len(self.mat_data)
        while start < datalen :
            for ep in range(epoch):
                for i in range(self.out_bit):
                    self.batch_update_w(i, self.mat_data[start:end],\
                        self.label_data[start:end])
            start = end
            end += stoc

    def predict(self, array_input):
        if array_input.__class__.__name__ != 'ndarray':
            array_input = convert.list2arr(array_input)
        return map(round,map(self.sigmoid,(array_input * self.mat_w).sum(axis=1) \
                + self.mat_w0))

