#
# Feedforward Neural Network
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
import sys

class FNN:

    def __init__(self, mat_data, label_data, hl_size, ol_size):
        self.mat_data = convert.list2npfloat(mat_data)
        self.label_data = convert.list2npfloat(label_data)
        self.il_size = self.mat_data.shape[1]
        self.hl_size = hl_size
        self.ol_size = ol_size
        self.W =  array([ [random.random() * 0.0000001 + sys.float_info.epsilon\
                            for i in range(len(mat_data[0]))] \
                                for j in range(hl_size) ])
        self.W0 = array([random.random() * 0.0000001 + sys.float_info.epsilon\
                            for i in range(hl_size) ])
        self.V =  array([ [random.random() * 0.0000001 + sys.float_info.epsilon\
                            for i in range(hl_size)] \
                                for j in range(ol_size) ])
        self.V0 = array([random.random() * 0.0000001 + sys.float_info.epsilon\
                            for i in range(ol_size) ])
        self.makelabelmap()

    def makelabelmap(self):
        self.y = array(self.label_data)
    
    def match_label(self, output):
        return output

    def sigmoid(self,k):
        return 1.0 / ( 1.0 + exp(-k))

    def sigmoid_delta(self,k):
        return self.sigmoid(k) * (1.0 - self.sigmoid(k))

    def feedforward(self, x):
        hin = self.W.dot(x) + self.W0
        hot = self.sigmoid(hin)
        oin = self.V.dot(hot) + self.V0
        oot = self.sigmoid(oin)
        return hin, hot, oin, oot
        
    def fit(self, lr, epoch, err_th):
        err = 9999.0
        npoch = 0
        while err > err_th and npoch < epoch :
            err = 0
            npoch += 1
            for idx, x in enumerate(self.mat_data):
                hin, hot, oin, oot = self.feedforward(x)
                now_err = ((self.y[idx] - oot) * (self.y[idx] - oot)).sum(axis=0)
                err += now_err
                odelta = -1 * self.sigmoid_delta(oin) * (self.y[idx] - oot)
                hdelta = self.sigmoid_delta(hin) * (odelta.dot(self.V))
                dV = array([odelta]).transpose().dot(array([hot]))
                dW = array([hdelta]).transpose().dot(array([x]))
                self.V = self.V - lr * dV
                self.W = self.W - lr * dW
                self.V0 = self.V0 - lr * odelta
                self.W0 = self.W0 - lr * hdelta

    def predict(self, array_input):
        array_input = convert.list2npfloat(array_input)
        hin, hot, oin, oot = self.feedforward(array_input)
        return self.match_label(oot)
