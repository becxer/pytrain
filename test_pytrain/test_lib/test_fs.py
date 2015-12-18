#
# test lib.fs
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.lib import fs

class test_fs(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        sample_data = "sample_data/dating/date_info.txt"

        self.tlog("loading => " + sample_data)

        dmat_train, dlabel_train, dmat_test, dlabel_test \
            = fs.f2mat(sample_data, 0.1)
        assert len(dmat_train) == 900
        assert len(dlabel_train) == 900
        assert len(dmat_test) == 100
        assert len(dlabel_test) == 100

        self.set_global_value('dmat_train',dmat_train)
        self.set_global_value('dlabel_train',dlabel_train)
        self.set_global_value('dmat_test',dmat_test)
        self.set_global_value('dlabel_test',dlabel_test)


