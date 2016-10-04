#
# test lib.autotest
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.KNN import KNN
from pytrain.lib import autotest

class test_autotest(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        normed_dmat_train = self.get_global_value('normed_iris_mat_train')
        normed_dmat_test = self.get_global_value('normed_iris_mat_test')
        dlabel_train = self.get_global_value('iris_label_train')
        dlabel_test = self.get_global_value('iris_label_test')

        knn_date = KNN(normed_dmat_train, dlabel_train, 3, 'euclidean')
        error_rate = autotest.eval_predict(knn_date, normed_dmat_test, dlabel_test, self.logging)
        self.tlog("date predict (with basic knn) error rate : " + str(error_rate))
