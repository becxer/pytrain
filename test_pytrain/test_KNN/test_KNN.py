#
# test KNN
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.KNN import KNN
from pytrain.lib import autotest
from pytrain.lib import dataset
import numpy as np

class test_KNN_iris(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        iris_mat_train, iris_label_train = dataset.load_iris("sample_data", "training")
        iris_mat_test, iris_label_test = dataset.load_iris("sample_data", "testing")
        
        knn = KNN(iris_mat_train, iris_label_train, 3, 'manhattan')
        error_rate = autotest.eval_predict(knn, iris_mat_test, iris_label_test, self.logging)
        self.tlog("iris predict (with basic knn) error rate :" + str(error_rate))        


class test_KNN_mnist(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        dg_mat_train, dg_label_train = dataset.load_mnist("sample_data/mnist", "training") 
        dg_mat_test, dg_label_test = dataset.load_mnist("sample_data/mnist", "testing")

        dg_mat_train = np.reshape(dg_mat_train,[-1, 28 * 28])
        dg_mat_test = np.reshape(dg_mat_test,[-1, 28 * 28])

        knn_digit = KNN(dg_mat_train, dg_label_train, 10, 'euclidean')
        error_rate = autotest.eval_predict(knn_digit, dg_mat_test, dg_label_test, self.logging)
        self.tlog("digit predict (with basic knn) error rate :" + str(error_rate))

