#
# test Linear Regression
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.LinearRegression import LinearRegression
from pytrain.lib import dataset
from pytrain.lib import autotest

class test_LinearRegression(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):

        train_mat = [\
                     [0.12, 0.25],\
                     [3.24, 4.33],\
                     [0.14, 0.45],\
                     [7.30, 4.23],\
                     ]
        train_label = [[0,1],[1,0],[0,1],[1,0]]
        
        linear_reg =\
            LinearRegression(train_mat, train_label)
        linear_reg.fit(lr = 0.001, epoch = 1000, batch_size = 4)
        
        r1 = autotest.eval_predict_one(linear_reg,[0.10,0.33],[0, 1],self.logging, one_hot = True)
        r2 = autotest.eval_predict_one(linear_reg,[4.40,4.37],[1, 0],self.logging, one_hot = True)
        
class test_LinearRegression_iris(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        iris_mat_train, iris_label_train = dataset.load_iris("sample_data", "training", one_hot=True)
        iris_mat_test, iris_label_test = dataset.load_iris("sample_data", "testing", one_hot=True)

        linear_reg = LinearRegression(iris_mat_train, iris_label_train)
        linear_reg.fit(lr = 0.0001, epoch = 1000, batch_size = 20)
        error_rate = autotest.eval_predict(linear_reg, iris_mat_test, iris_label_test, self.logging, one_hot=True)
        self.tlog("iris predict (with linear regression) error rate :" + str(error_rate))

class test_LinearRegression_mnist(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        dg_mat_train, dg_label_train = dataset.load_mnist("sample_data", "training", one_hot=True) 
        dg_mat_test, dg_label_test = dataset.load_mnist("sample_data", "testing", one_hot=True)

        linear_reg = LinearRegression(dg_mat_train, dg_label_train)
        linear_reg.fit(lr = 0.0000001, epoch = 1000, batch_size = 100)
        error_rate = autotest.eval_predict(linear_reg, dg_mat_test, dg_label_test, self.logging, one_hot=True)
        self.tlog("digit predict (with linear regression) error rate :" + str(error_rate))
