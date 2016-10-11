#
# test Logistic Regression
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.LogisticRegression import LogisticRegression
from pytrain.lib import fs
from pytrain.lib import autotest
from pytrain.lib import dataset

class test_LogisticRegression(test_Suite):

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
        
        logistic_reg =\
            LogisticRegression(train_mat, train_label)
        logistic_reg.fit(lr = 0.001, epoch = 2000, batch_size = 4)
        
        r1 = autotest.eval_predict_one(logistic_reg,[0.10,0.33],[0, 1],self.logging, one_hot=True)
        r2 = autotest.eval_predict_one(logistic_reg,[4.40,4.37],[1, 0],self.logging, one_hot=True)
        
class test_LogisticRegression_iris(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        iris_mat_train, iris_label_train = dataset.load_iris("sample_data/iris", "training", one_hot=True)
        iris_mat_test, iris_label_test = dataset.load_iris("sample_data/iris", "testing", one_hot=True)

        logistic_reg = LogisticRegression(iris_mat_train, iris_label_train)
        logistic_reg.fit(lr = 0.001, epoch = 2000, batch_size = 30)
        error_rate = autotest.eval_predict(logistic_reg, iris_mat_test, iris_label_test, self.logging, one_hot=True)
        self.tlog("iris predict (with logistic  regression) error rate :" + str(error_rate))

class test_LogisticRegression_mnist(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        print "__TODO__ : MNIST test to be implement"
        pass
    
