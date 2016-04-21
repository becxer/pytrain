#
# HierarchicalClustering
#
# @ author becxer
# @ email becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
from pytrain.lib import ptmath
import operator

class HierarchicalClustering:
    
    def __init__(self, mat_data, K, dist_func):
        self.mat_data = convert.list2npfloat(mat_data)
        self.dist_func = ptmath.distfunc(dist_func)
        self.K = K
        self.col_len = len(self.mat_data[0])
        self.row_len = len(self.mat_data)
        
    def fit(self):
        return self.cluster()
    
    def cluster(self):

        class line :
            def __init__(self, src, trg, distance):
                self.src = src
                self.trg = trg
                self.distance = distance

        class closest :
            def __init__(self, ):
                self.data_idxs = []
                
            def merge(self, other):
                self.data_idxs.append(idx)
             
                
        # 0.create cluster which 
                
        # 1.create line instance each pair of mat_data

        # 
        
        
        label_of_data = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
        self.label_of_data = label_of_data
        return label_of_data

    # default predictor using KNN
    def predict(self, input_array):
        input_array = convert.list2npfloat(input_array)
        pass
