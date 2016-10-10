#
# Logistic Regression
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
seterr(all='raise')
from pytrain.lib import convert
from pytrain.lib import ptmath
import math
import time
import random
import sys

class LogisticRegression:

    def __init__(self, mat_data, label_data):
        self.mat_data = convert.list2npfloat(mat_data)
        self.label_data = convert.list2npfloat(label_data)

        self.out_bit = len(label_data[0])
        self.mat_w =  [ [random.random() * 0.0001 + sys.float_info.epsilon\
                            for i in range(len(mat_data[0]))] \
                                for j in range(self.out_bit) ]
        self.mat_w0 = [random.random() * 0.0001 + sys.float_info.epsilon\
                            for i in range(self.out_bit) ]

    def batch_update_w(self, out_bit_index, data, label):
        w = self.mat_w[out_bit_index]
        w0 = self.mat_w0[out_bit_index]
        tiled_w0 = tile(w0,(len(data)))
        k = (w * data).sum(axis=1) + tiled_w0
        sig_k = map(ptmath.sigmoid,k)
        gd = (label.T[out_bit_index] - sig_k)
        dw = (gd * data.T).sum(axis=1)/len(data) * -1
        dw0  = gd.sum(axis=0)/len(data) * -1
        w = w - (dw * self.lr)
        w0 = w0 - (dw0 * self.lr)
        self.mat_w[out_bit_index] = w
        self.mat_w0[out_bit_index] = w0

    def fit(self, lr, epoch, batch_size):
        self.lr = lr
        self.epoch = epoch
        start = 0
        end = batch_size
        datalen = len(self.mat_data)
        while start < datalen :
            for ep in range(epoch):
                for i in range(self.out_bit):
                    self.batch_update_w(i, self.mat_data[start:end],\
                        self.label_data[start:end])
            start = end
            end += batch_size

    def predict(self, array_input):
        array_input = convert.list2npfloat(array_input)
        return map(round,map(ptmath.sigmoid,(array_input * self.mat_w).sum(axis=1) \
                + self.mat_w0))

