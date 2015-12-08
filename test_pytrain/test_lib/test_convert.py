#
# test lib.convert
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite
from pytrain.lib import convert

class test_convert(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        dmat_train = self.get_gvalue('dmat_train')
        dmat_test = self.get_gvalue('dmat_test')
        
        normed_dmat_train = convert.norm(dmat_train)
        normed_dmat_test = convert.norm(dmat_test)

        self.tlog(normed_dmat_train[0:10])

        assert len(normed_dmat_train) == 900
        assert len(normed_dmat_test) == 100

        self.set_gvalue('normed_dmat_train',normed_dmat_train)
        self.set_gvalue('normed_dmat_test',normed_dmat_test)
