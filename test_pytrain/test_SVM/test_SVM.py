#
# test SVM
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.SVM import SVM
from pytrain.lib import autotest 

class test_SVM(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        train_mat = [\
                     [0.12, 0.25],\
                     [3.24, 4.33],\
                     [0.14, 0.45],\
                     [7.30, 4.23],\
                     ]
        train_label = [[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]] # out bit is 1

        svm = SVM(train_mat, train_label)
        svm.fit(C = 0.1, toler = 0.001, epoch = 40)
        
        r1 = autotest.eval_predict_one(svm,[0.10,0.33], [-1.0, 1.0] , self.logging)
        r2 = autotest.eval_predict_one(svm,[4.40,4.37], [1.0, -1.0] , self.logging)

        assert r1
        assert r2
        
