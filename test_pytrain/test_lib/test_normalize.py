#
# test lib.normalize
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.lib import normalize

class test_normalize(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_normalize_quantile(self):
        iris_mat_train = self.get_global_value('iris_mat_train')
        iris_mat_test = self.get_global_value('iris_mat_test')
        
        normed_imat_train = normalize.quantile(iris_mat_train)
        normed_imat_test = normalize.quantile(iris_mat_test)

        self.tlog("before normalized : \n" + str(iris_mat_train[:3]))
        self.tlog("normalized sample : \n" + str(normed_imat_train[:3]))

        assert normed_imat_train[0][0] <= 1.0 and \
                normed_imat_train[0][0] >= 0.0

        self.set_global_value('normed_iris_mat_train',normed_imat_train)
        self.set_global_value('normed_iris_mat_test',normed_imat_test)

    def test_process(self):
        self.test_normalize_quantile()
