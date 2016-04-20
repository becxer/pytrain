#
# test HierarchicalClustering
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.HierarchicalClustering import HierarchicalClustering
from pytrain.lib import batch
from numpy import *

class test_HierarchicalClustering(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        # below data shows 6 group of each 5 point data
        sample_mat = [\
                          [0.2,0.3],[1.0,0.28],[1.98,0.7],\
                          [0.1,1.11],[1.0,1.12],\
                          [5.94,0.4],[6.73,0.38],[7.42,0.97],\
                          [6.74,1.23],[5.91,1.20],\
                          [2.0,4.8],[2.74,4.78],[3.6,5.1],\
                          [3.1,5.3],[1.95,5.8],\
                          [8.94,5.2],[9.6,5.12],[10.31,5.29],\
                          [8.73,6.0],[9.54,5.99],\
                          [5.17,9.1],[5.64,8.97],[6.56,9.39],\
                          [4.99,9.82],[5.5,9.74],\
                          [11.8,1.8],[12.04,1.74],[12.9,2.0],\
                          [11.74,2.4],[12.11,2.32]
                      ]

        # labeling cluster for each data
        hac = HierarchicalClustering(sample_mat, K = 6, dist_func = "euclidean")
        label_data =  hac.cluster() # hac.fit() also same

        for idx, row in enumerate(sample_mat):
            self.tlog(str(row) + " -> " + str(label_data[idx]))
        self.tlog("labels are " + str(list(set(label_data))))

        # clustering test with unknown data
        r1 = batch.eval_predict_one(hac, [11.70, 3.0], \
                            hac.predict([11.74, 2.4]), self.logging)
        r2 = batch.eval_predict_one(hac, [8.40, 5.8], \
                            hac.predict([8.73, 6.0]), self.logging)
        r3 = batch.eval_predict_one(hac, [0.7, 0.1], \
                            hac.predict([1.08, 0.7]), self.logging)

        assert (r1 and r2 and r3)
