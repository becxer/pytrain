#
# test SVM
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.SVM import SVM
from pytrain.lib import autotest
import numpy as np

class test_SVM(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        train_mat = np.mat([\
                     [0.12, 0.25],\
                     [3.24, 4.33],\
                     [0.14, 0.45],\
                     [7.30, 4.23],\
                     ])
        train_label = np.mat([[-1.0], [1.0], [-1.0], [1.0]]) # out bit is 1

        svm = SVM(train_mat, train_label)
        svm.fit(C = 5.0, toler = 0.001, epoch = 50)
        
        r1 = autotest.eval_predict_one(svm,np.mat([0.10,0.33]), -1.0, self.logging)
        r2 = autotest.eval_predict_one(svm,np.mat([4.40,4.37]), 1.0, self.logging)

        assert r1
        assert r2
        
