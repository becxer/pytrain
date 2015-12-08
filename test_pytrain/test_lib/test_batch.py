#
# test batch
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite
from pytrain.knn import basic_knn
from pytrain.lib import batch

class test_batch(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        normed_dmat_train = self.get_gvalue('normed_dmat_train')
        normed_dmat_test = self.get_gvalue('normed_dmat_test')
        dlabel_train = self.get_gvalue('dlabel_train')
        dlabel_test = self.get_gvalue('dlabel_test')

        knn_date = basic_knn(normed_dmat_train, dlabel_train, 3)
        error_rate = batch.eval_predict(knn_date, normed_dmat_test, dlabel_test, False)
        self.tlog("date predict (with basic knn) error rate : " + str(error_rate))
        
        assert error_rate == 0.05
