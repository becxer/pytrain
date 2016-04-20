#
# test KNN
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.KNN import KNN
from pytrain.lib import fs
from pytrain.lib import batch


class test_KNN(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        sample_mat = [[1.0,1.1] , [1.0,1.0], [0,0], [0,0.1]]
        sample_label = ['A','A','B','B']
        knn = KNN(sample_mat, sample_label, 3, 'manhattan')
        
        r1 = batch.eval_predict_one(knn, [0.9,0.9] , 'A', self.logging)
        r2 = batch.eval_predict_one(knn, [0.1,0.4] , 'B', self.logging)

        assert r1 == True
        assert r2 == True


class test_KNN_digit(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        dg_mat_train, dg_label_train = fs.f2mat("sample_data/digit/digit-train.txt",0)
        dg_mat_test, dg_label_test = fs.f2mat("sample_data/digit/digit-test.txt",0)
        knn_digit = KNN(dg_mat_train, dg_label_train, 3, 'euclidean')
        error_rate = batch.eval_predict(knn_digit, dg_mat_test, dg_label_test, self.logging)
        self.tlog("digit predict (with basic knn) error rate :" + str(error_rate))

