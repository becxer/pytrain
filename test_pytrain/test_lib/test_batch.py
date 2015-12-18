#
# test lib.batch
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.KNN import KNN
from pytrain.lib import batch


class test_batch(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        normed_dmat_train = self.get_global_value('normed_dmat_train')
        normed_dmat_test = self.get_global_value('normed_dmat_test')
        dlabel_train = self.get_global_value('dlabel_train')
        dlabel_test = self.get_global_value('dlabel_test')

        knn_date = KNN(normed_dmat_train, dlabel_train, 3)
        error_rate = batch.eval_predict(knn_date, normed_dmat_test, dlabel_test, False)
        self.tlog("date predict (with basic knn) error rate : " + str(error_rate))
        
        assert error_rate == 0.05
