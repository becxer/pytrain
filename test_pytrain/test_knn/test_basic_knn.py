#
# test knn.basic_knn
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite
from pytrain.knn import basic_knn

class test_basic_knn(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        sample_mat = [[1.0,1.1] , [1.0,1.0], [0,0], [0,0.1]]
        sample_label = ['A','A','B','B']
        
        knn = basic_knn(sample_mat, sample_label, 3)

        f1 = knn.predict([0.9,0.9])
        f2 = knn.predict([0.1,0.4])

        self.tlog("predict [0.9,0.9] to " + f1)
        self.tlog("predict [0.1,0.4] to " + f2)

        assert f1 == 'A'
        assert f2 == 'B'

