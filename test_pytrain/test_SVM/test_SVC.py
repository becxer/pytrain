#
# test SVC
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.SVM import SVC
from pytrain.lib import dataset
from pytrain.lib import autotest

class test_SVC(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        train_mat = [\
                     [0.12, 0.25],\
                     [3.24, 4.33],\
                     [0.14, 0.45],\
                     [7.30, 4.23],\
                     ]
        train_label = [[0,1], [1,0], [0,1], [1,0]] # out bit is 2

        svc = SVC(train_mat, train_label)
        svc.fit(C = 5.0, toler = 0.001, epoch = 50)
        
        r1 = autotest.eval_predict_one(svc,[0.10,0.33], [0., 1.], self.logging)
        r2 = autotest.eval_predict_one(svc,[4.40,4.37], [1., 0.], self.logging)

        assert r1
        assert r2
               
class test_SVC_iris(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        iris_mat_train, iris_label_train = dataset.load_iris("sample_data", "training", one_hot=True)
        iris_mat_test, iris_label_test = dataset.load_iris("sample_data", "testing", one_hot=True)

        svc = SVC(iris_mat_train, iris_label_train)
        svc.fit(C = 1.5, toler = 0.0001, epoch = 1000, kernel = "Polynomial", kernel_params = {"degree" : 3})
        error_rate = autotest.eval_predict(svc, iris_mat_test, iris_label_test, self.logging, one_hot=True)
        self.tlog("iris predict (with svc) error rate :" + str(error_rate))

class test_SVC_mnist(test_Suite):
    
    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        dg_mat_train, dg_label_train = dataset.load_mnist("sample_data", "training", one_hot=True) 
        dg_mat_test, dg_label_test = dataset.load_mnist("sample_data", "testing", one_hot=True)

        svc = SVC(dg_mat_train, dg_label_train)
        svc.fit(C = 1.5, toler = 0.0001, epoch = 1000, kernel = "RBF" , kernel_params = {"gamma" : 0.7})       
        error_rate = autotest.eval_predict(svc, dg_mat_test, dg_label_test, self.logging, one_hot=True)
        self.tlog("digit predict (with svc) error rate :" + str(error_rate))
