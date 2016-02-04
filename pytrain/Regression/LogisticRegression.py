#
# Logistic Regression
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
import math

class LogisticRegression:

    def __init__(self, mat_data, label_data):
        if mat_data.__class__.__name__ != 'ndarray':
            mat_data = convert.mat2arr(mat_data)
        if label_data.__class__.__name__ != 'ndarray':
            label_data = convert.mat2arr(label_data)
        self.mat_data = mat_data
        self.label_data = label_data
        self.out_bit = len(label_data[0])
        self.mat_w =  [ [0.8 for i in range(len(mat_data[0]))] \
                        for j in range(self.out_bit) ]
        self.mat_w0 = [ 0.8 for i in range(self.out_bit) ]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x)) 

    def df_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    #
    # Description of differential equation
    #
    # k1 = w0 + w1 x11 + w2 x12 + w3 x13 + .. + wn x1n
    # J(k) = sum { (origin - sigmoid(k))^2 }
    #
    # dJ/dw1 = dJ/dk1 * dk1/dw1 = 
    #                    - 2 * (origin1 - sigmoid(k1)) * df_sigmoid(k1) * x11
    #                    - 2 * (origin2 - sigmoid(k2)) * df_sigmoid(k2) * x21 
    #                    - 2 * (origin3 - sigmoid(k3)) * df_sigmoid(k3) * x31
    #                    ...
    # 
    # UPDATE w1 with gradient
    # w1 = w1 - lr * dJ/dw1
    #
    # dJ/dw0 = dJ/dk0 * dk0/dw0 = 
    #                   - 2 * (origin1 - sigmoid(k1)) * df_sigmoid(k1) * 1
    #                   - 2 * (origin2 - sigmoid(k2)) * df_sigmoid(k2) * 1
    #                   ...
    #
    # w0 = w0 - lr * dJ/w0
    #

    def batch_update_w(self, out_bit_index, data, label):
        w = self.mat_w[out_bit_index]
        w0 = self.mat_w0[out_bit_index]
        tiled_w = tile(w,(len(data),1)) 
        tiled_w0 = tile(w0,(len(data)))
        k = (tiled_w * data).sum(axis=1) + tiled_w0
        sig_k = map(lambda x : self.sigmoid(x), k)
        df_sig_k = map(lambda x : self.df_sigmoid(x), k)
        gd = (label.T[out_bit_index] - sig_k) * (-2) * df_sig_k
        gd_mat = tile(gd,(len(w),1)).T
        # dJ_dw is gradient of J(w) function
        dJ_dw = (gd_mat * data).sum(axis=0)
        w = w - (dJ_dw * self.lr)
        w0 = w0 - (gd.sum(axis=0) * self.lr)
        self.mat_w[out_bit_index] = w
        self.mat_w0[out_bit_index] = w0

    def fit(self, lr, epoch):
        self.lr = lr
        self.epoch = epoch
        for ep in range(epoch):
            for i in range(self.out_bit):
                self.batch_update_w(i, self.mat_data, self.label_data)

    def predict(self, array_input):
        if array_input.__class__.__name__ != 'ndarray':
            array_input = convert.list2arr(array_input)
        return map(\
                lambda x : round(self.sigmoid(x)),\
                (tile(array_input, (self.out_bit,1)) * self.mat_w).sum(axis=1) \
                + self.mat_w0)

