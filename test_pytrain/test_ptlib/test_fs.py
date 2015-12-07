#
# test ptlib.fs
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite
from pytrain.ptlib import fs

class test_fs(test_suite):

    def __init__(self):
        test_suite.__init__(self)

    def test_process(self):
        dmat_train, dlabel_train, dmat_test, dlabel_test \
            = fs.f2mat("test_data/dating/date_info.txt", 0.1)
        assert len(dmat_train) == 900
        assert len(dlabel_train) == 900
        assert len(dmat_test) == 100
        assert len(dmat_test) == 100

