#
# SVC (SVM Multi classifier)
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

import numpy as np
from pytrain.SVM import SVM
from pytrain.lib import convert
from pytrain.lib import ptmath

class SVC:

    def __init__(self, mat_data, label_data):
        self.x = np.mat(convert.list2npfloat(mat_data))
        self.ys = np.mat(np.sign(convert.list2npfloat(label_data) - 0.5))
        self.outbit = self.ys.shape[1]
        self.svm4bit = []
        for i in range(self.outbit):
            self.svm4bit.append(SVM(self.x, self.ys[:,i]))
        
    def fit(self, C, toler, epoch, kernel = 'Linear', kernel_params = {}):
        for i in range(self.outbit):
            self.svm4bit[i].fit(C, toler, epoch, kernel, kernel_params)
        
    def predict(self, array_input):
        array_input = np.mat(convert.list2npfloat(array_input))
        output = []
        for i in range(self.outbit):
            output.append(self.svm4bit[i].predict(array_input))
        return list(np.sign(np.array(output) + 1))
