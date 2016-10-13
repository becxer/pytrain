#
# Kmeans
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
from pytrain.lib import ptmath
import math

class Kmeans:
    
    def __init__(self, mat_data, dist_func):
        self.mat_data = convert.list2npfloat(mat_data)
        self.dist_func = ptmath.distfunc(dist_func)
        self.col_len = float(len(self.mat_data[0]))
        self.row_len = float(len(self.mat_data))
        self.min_col = self.mat_data.min(axis=0)
        self.max_col = self.mat_data.max(axis=0)

    # assign input array to cluster
    def predict(self, input_array):
        input_array = convert.list2npfloat(input_array)
        return self.assign_row(self.cluster_points, input_array)
        
    def assign_row(self, cluster_points, row):
        min_idx = -1
        min_dist = None
        for i, cp in enumerate(cluster_points):
            cp_dist = self.dist_func(row, cp)
            if min_dist == None or min_dist > cp_dist:
                min_dist = cp_dist
                min_idx = i
        return min_idx
            
    def assign_mat_data(self, cluster_points):
        cluster_set = {}
        # assign class for every data
        for i, row in enumerate(self.mat_data):
            min_idx = self.assign_row(cluster_points, row)
            cluster_set[min_idx] = cluster_set.get(min_idx, [])
            cluster_set[min_idx].append(row)
        return cluster_set

    
    # silhouette cluster metric
    #
    # s(i) = 1 - a(i)/b(i), if a(i) < b(i)
    # s(i) = 0, if a(i) = b(i)
    # s(i) = b(i)/a(i) - 1, if a(i) > b(i)
    def metric(self, cluster_points):
        cluster_set = self.assign_mat_data(cluster_points)
        sil_res = 0
        for idx in cluster_set:
            cl = cluster_set[idx]
            for i_data in cl:
                a_i = 0.
                for oth_data in cl:
                    a_i += self.dist_func(i_data, oth_data)
                if len(cl) > 0 :
                    a_i /= float(len(cl))
                b_i = inf
                for jdx in cluster_set:
                    if jdx != idx:
                        for oth_data in cluster_set[jdx]:
                            dist = self.dist_func(i_data, oth_data)
                            if b_i == None or dist < b_i:
                                b_i = dist
                sil_i = 0
                if max(b_i,a_i) != 0 :
                    sil_i = float((b_i - a_i) / max(b_i,a_i))
                sil_res += sil_i
        sil_res /= float(self.row_len)
        return sil_res
    
    def cluster(self, K, epoch):
        # set random point for K class
        cluster_points = random.random_sample((K, int(self.col_len)))
        cluster_points = (cluster_points * (self.max_col - self.min_col))\
          + self.min_col

        # Lloyd algorithm  
        for ep in range(epoch):
            cluster_set = self.assign_mat_data(cluster_points)
            # reassign K class with average of class items
            for idx in cluster_set:
                new_k_row = array(cluster_set[idx])
                cluster_points[idx] = new_k_row.sum(axis=0) / len(new_k_row)

        self.cluster_points = cluster_points
        return cluster_points

                
    # Good clustering finding method
    def fit(self, max_K ,random_try_count, epoch):

        cluster_points_good = None
        cluster_points_good_metric = None

        for K in range(2,max_K+1):
            for i in range(random_try_count):
                cluster_points = self.cluster(K, epoch)
                metric_val = self.metric(cluster_points)
                if cluster_points_good_metric == None \
                  or metric_val > cluster_points_good_metric:
                    cluster_points_good_metric = metric_val
                    cluster_points_good = cluster_points
        self.cluster_points = cluster_points_good        
        return self.cluster_points

