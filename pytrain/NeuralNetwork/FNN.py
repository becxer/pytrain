#
# Feedforward Neural Network
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
from pytrain.lib import ptmath
import sys

class FNN:

    def __init__(self, mat_data, label_data, hl_list):
        self.mat_data = convert.list2npfloat(mat_data)
        self.label_data = label_data
        self.il_size = self.mat_data.shape[1]
        self.idx_label_map = list(set(self.label_data))
        self.ol_size = len(self.idx_label_map)
        self.W = {}
        self.B = {}
        last_layer_num = self.il_size
        for idx, hl_num in enumerate(hl_list):
            self.W['WD_' + str(idx + 1)] = 0.1 * random.randn(last_layer_num, hl_num)
            self.B['BD_' + str(idx + 1)] = 0.1 * random.randn(hl_num)
            last_layer_num = hl_num
        self.W['out'] = 0.1 * random.randn(last_layer_num, self.ol_size)
        self.B['out'] = 0.1 * random.randn(self.ol_size)
        self.makelabelmap()

    def makelabelmap(self):
        self.label_onehot_map = {}
        for i, label in enumerate(self.idx_label_map):
            onehot = zeros(self.ol_size)
            onehot[i] = 1
            self.label_onehot_map[label] = onehot
        self.y = []
        for each_label in self.label_data:
            self.y.append(self.label_onehot_map[each_label])
        self.y = array(self.y)
            
    def match_label(self, output):
        return self.idx_label_map[argmax(self.output)]

    def feedforward(self, x):
        last_input = x
        layer = {}
        for idx, hl_num in enumerate(self.hl_list):
            tmp = self.W['WD_' + str(idx)].dot(last_input) + self.B['BD_' + str(idx)]
            last_input = ptmath.sigmoid(tmp)
            layer['D_' + str(idx)] = last_input
        layer['out'] = self.W['out'].dot(last_input) + self.B['out']
        return layer

    def backprop_dw(self, now_x, now_W, now_B, top_delta, top_W = None):
        err_toss = top_delt
        top_Wx = now_W.dot(now_x) + now_B
        if top_W != None:
            err_toss = err_toss.dot(top_W)
        now_delta = pt.math.sigmoid_delta(top_Wx) * err_th
        now_dW = array([now_delta]).transpose().dot(array([now_x]))
        return now_delta, now_dW
        
    def fit(self, lr, epoch, err_th):
        err = 9999.0
        npoch = 0
        while err > err_th and npoch < epoch :
            err = 0
            npoch += 1
            for idx, x in enumerate(self.mat_data):
                layer = self.feedforward(x)
                now_err = ((self.y[idx] - layer['out']) * (self.y[idx] - layer['out'])).sum(axis=0)
                err += now_err
                top_delta, top_W = (layer['out'] - self.y[idx], None)
                for top

                odelta = ptmath.sigmoid_delta(oin) * (oot - self.y[idx])
                hdelta = ptmath.sigmoid_delta(hin) * (odelta.dot(self.V))
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
