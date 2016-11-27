#
# test lib.normalize
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.lib import dataset

class test_dataset(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_load_iris(self):
        iris_mat_train, iris_label_train = dataset.load_iris("sample_data", "training")
        iris_mat_test, iris_label_test = dataset.load_iris("sample_data", "testing")
        self.tlog("iris train data size : " + str(len(iris_mat_train)))
        self.tlog("iris test data size : " + str(len(iris_mat_test)))

    def test_load_iris_one_hot(self):
        iris_mat_train, iris_label_train = dataset.load_iris("sample_data", "training", one_hot = True)
        iris_mat_test, iris_label_test = dataset.load_iris("sample_data", "testing", one_hot = True)
        self.tlog("iris train data size : " + str(len(iris_mat_train)))
        self.tlog("iris test data size : " + str(len(iris_mat_test)))
        
    def test_load_mnist(self):
        mnist_mat_train, mnist_label_train \
          = dataset.load_mnist("sample_data", "training", [0,1,2,3,4])
        mnist_mat_test, mnist_label_test \
          = dataset.load_mnist("sample_data", "testing", [0,1,2,3,4])
        self.tlog("mnist train data size : " + str(len(mnist_mat_train)))
        self.tlog("mnist test data size : " + str(len(mnist_mat_test)))
        
    def test_process(self):
        self.test_load_iris()
        self.test_load_iris_one_hot()
        self.test_load_mnist()

