#
# test Feedforward Neural Network
#
# @ author becxer
# @ e-mail becxer87@gmail.com
# 

from test_pytrain import test_Suite
from pytrain.NeuralNetwork import FNN
from pytrain.lib import autotest
from pytrain.lib import dataset
from numpy import *

class test_FNN(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):

        train_mat = [\
                     [0.12, 0.25],\
                     [3.24, 4.33],\
                     [0.14, 0.45],\
                     [7.30, 4.23],\
                     ]
        train_label = [[0,1],[1,0],[0,1],[1,0]] # out bit is 1
        
        fnn = FNN(train_mat, train_label, [3])
        fnn.fit(lr = 0.01, epoch = 2000, err_th = 0.001, batch_size = 4)
        
        r1 = autotest.eval_predict_one(fnn,[0.10,0.33],[0, 1],self.logging, one_hot=True)
        r2 = autotest.eval_predict_one(fnn,[4.40,4.37],[1, 0],self.logging, one_hot=True)
        
class test_FNN_iris(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        iris_mat_train, iris_label_train = dataset.load_iris("sample_data", "training", one_hot=True)
        iris_mat_test, iris_label_test = dataset.load_iris("sample_data", "testing", one_hot=True)

        fnn = FNN(iris_mat_train, iris_label_train, [2])
        fnn.fit(lr = 0.001, epoch = 4000, err_th = 0.00001, batch_size = 30)
        error_rate = autotest.eval_predict(fnn, iris_mat_test, iris_label_test, self.logging, one_hot=True)
        self.tlog("iris predict (with fnn) error rate :" + str(error_rate))

class test_FNN_mnist(test_Suite):
    
    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        dg_mat_train, dg_label_train = dataset.load_mnist("sample_data", "training", one_hot=True) 
        dg_mat_test, dg_label_test = dataset.load_mnist("sample_data", "testing", one_hot=True)

        fnn = FNN(dg_mat_train, dg_label_train, [400, 100])
        fnn.fit(lr = 0.00001, epoch = 100, err_th = 0.00001, batch_size = 100)
        error_rate = autotest.eval_predict(fnn, dg_mat_test, dg_label_test, self.logging, one_hot=True)
        self.tlog("digit predict (with logistic regression) error rate :" + str(error_rate))
