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
        self.hl_list = hl_list
        self.idx_label_map = list(set(self.label_data))
        self.W = {}
        self.B = {}
        self.hl_list.append(len(self.idx_label_map))
        last_layer_num = self.il_size
        for idx, hl_num in enumerate(hl_list):
            self.W['WD_' + str(idx)] = 0.1 * random.randn(hl_num, last_layer_num)
            self.B['BD_' + str(idx)] = 0.1 * random.randn(hl_num)
            last_layer_num = hl_num
        self.makelabelmap()

    def makelabelmap(self):
        self.label_onehot_map = {}
        for i, label in enumerate(self.idx_label_map):
            onehot = zeros(self.hl_list[-1])
            onehot[i] = 1
            self.label_onehot_map[label] = onehot
        self.y = []
        for each_label in self.label_data:
            self.y.append(self.label_onehot_map[each_label])
        self.y = array(self.y)
            
    def match_label(self, output):
        return self.idx_label_map[argmax(output)]

    def feedforward(self, x):
        last_input = x
        layer = {}
        for idx, hl_num in enumerate(self.hl_list):
            tmp = self.W['WD_' + str(idx)].dot(last_input) + self.B['BD_' + str(idx)]
            last_input = ptmath.sigmoid(tmp)
            layer['OUT_' + str(idx)] = last_input
        return layer['OUT_' + str(len(self.hl_list)-1)], layer

    def backprop_dw(self, now_x, now_W, now_B, top_delta, top_W = None):
        err_toss = top_delta
        now_out = now_W.dot(now_x) + now_B
        if top_W != None:
            err_toss = err_toss.dot(top_W)
        now_delta = ptmath.sigmoid_delta(now_out) * err_toss
        now_dW = array([now_delta]).transpose().dot(array([now_x]))
        return now_delta, now_dW
        
    def fit(self, lr, epoch, err_th):
        err = 9999.0
        npoch = 0
        while err > err_th and npoch < epoch :
            err = 0
            npoch += 1
            for idx, x in enumerate(self.mat_data):
                out, layer = self.feedforward(x)
                now_err = ((self.y[idx] - out) * (self.y[idx] - out)).sum(axis=0)
                err += now_err
                top_delta, top_W = (out - self.y[idx], None)
                dW = {}
                delta = {}
                for layer_idx in range(len(layer))[::-1]:
                    if layer_idx > 0:
                        now_x = layer['OUT_' + str(layer_idx-1)]
                    else:
                        now_x = x
                    now_W = self.W['WD_' + str(layer_idx)]
                    now_B = self.B['BD_' + str(layer_idx)]
                    delta['BD_' + str(layer_idx)], dW['WD_' + str(layer_idx)] = \
                      self.backprop_dw(now_x, now_W, now_B, top_delta, top_W)
                    top_delta = delta['BD_'+str(layer_idx)]
                    top_W = now_W

                for layer_idx in range(len(layer))[::-1]:
                    self.W['WD_'+ str(layer_idx)] -= lr * dW['WD_' + str(layer_idx)]
                    self.B['BD_' + str(layer_idx)] -= lr * delta['BD_' + str(layer_idx)]

    def predict(self, array_input):
        array_input = convert.list2npfloat(array_input)
        out, layer = self.feedforward(array_input)
        return self.match_label(out)
