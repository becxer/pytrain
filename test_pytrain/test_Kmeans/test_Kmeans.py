#
# test Kmeans
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.Kmeans import Kmeans
from numpy import *

class test_Kmeans(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        # Above data shows 6 group of each 5 point data
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
            
        kmeans = Kmeans(sample_mat)

        # finding cluster (Fixed K)
        cluster_point_fixed = \
                kmeans.cluster(K = 6, epoch = 30, dist_func = 'euclidean')
        self.tlog("fixed point count : " + str(len(cluster_point_fixed)))
        self.tlog("cluster point : \n" + str(cluster_point_fixed))

        # Auto finding good cluster (Flexible K)
        cluster_point_flexible = \
          kmeans.fit(max_K = 7, random_iter = 10, epoch = 30, dist_func = 'euclidean')
        self.tlog("flexible point count : " + str(len(cluster_point_flexible)))
        self.tlog("cluster point : \n" + str(cluster_point_flexible))

        
        assert len(cluster_point_flexible) < 8
        
