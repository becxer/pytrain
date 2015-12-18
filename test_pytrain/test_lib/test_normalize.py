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

    def test_process(self):
        dmat_train = self.get_global_value('dmat_train')
        dmat_test = self.get_global_value('dmat_test')
        
        normed_dmat_train = normalize.quantile(dmat_train)
        normed_dmat_test = normalize.quantile(dmat_test)

        self.tlog(normed_dmat_train[0:10])

        assert len(normed_dmat_train) == 900
        assert len(normed_dmat_test) == 100

        self.set_global_value('normed_dmat_train',normed_dmat_train)
        self.set_global_value('normed_dmat_test',normed_dmat_test)
