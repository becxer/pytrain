#
# Kmeans
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
import math

class Kmeans:

    def __init__(self, mat_data):
        if mat_data.__class__.__name__ != 'ndarray':
            mat_data = convert.mat2arr(mat_data)
        self.mat_data = mat_data
        self.col_len = float(len(mat_data[0]))
        self.row_len = float(len(mat_data))

    def extract_minmax(self, mat_data):
        self.min_col = mat_data.min(axis=0)
        self.max_col = mat_data.max(axis=0)

    def dist_func_set(self, dist_func):
        #
        # dist_func must to be form like
        # def dist_func(vector_x, vector_y):
        #     return distance(vector_x,vector_y)
        #
        def euclidean (vx,vy):
            return linalg.norm(vx - vy)

        func_set = {'euclidean' : euclidean}

        if type(dist_func).__name__ != 'function':
            if dist_func in func_set:
                dist_func = func_set[dist_func]
        return dist_func
            

    def assign_data(self, cluster_points, mat_data, dist_func):
        cluster_set = {}
        # assign class for every data
        for i, row in enumerate(mat_data):
            min_j = -1
            min_dist = inf
            for j, cp in enumerate(cluster_points):
                row_cp_dist = dist_func(row,cp)
                if min_dist > row_cp_dist:
                    min_dist = row_cp_dist
                    min_j = j
            cluster_set[min_j] = cluster_set.get(min_j, [])
            cluster_set[min_j].append(row)
        return cluster_set

    
    def cluster(self, K, epoch, dist_func):
        dist_func = self.dist_func_set(dist_func)
        # extract min,max for each column
        # randomly assign for K class
        self.extract_minmax(self.mat_data)
        cluster_points = random.random_sample((K,self.col_len))
        cluster_points = (cluster_points * (self.max_col - self.min_col))\
          + self.min_col

        # Lloyd algorithm  
        for ep in range(epoch):
            cluster_set = self.assign_data(cluster_points, self.mat_data, dist_func)
            # reassign K class with average of class items
            for idx in cluster_set:
                new_k_row = array(cluster_set[idx])
                cluster_points[idx] = new_k_row.sum(axis=0)/len(new_k_row)
        return cluster_points


    def metric(self, cluster_points, dist_func):
        # assign class for every data
        cluster_set = self.assign_data(cluster_points, self.mat_data, dist_func)

        #
        # silhouette cluster metric
        #
        # s(i) = 1 - a(i)/b(i), if a(i) < b(i)
        # s(i) = 0, if a(i) = b(i)
        # s(i) = b(i)/a(i) - 1, if a(i) > b(i)
        #

        sil_res = 0
        for idx in cluster_set:
            cl = cluster_set[idx]
            for i in cl:
                a_i = 0
                for oth in cl:
                    a_i += dist_func(i,oth)
                a_i /= float(len(cl))
                b_i = inf
                for jdx in cluster_set:
                    if jdx != idx:
                        for oth in cluster_set[jdx]:
                            dist = dist_func(i,oth)
                            if dist < b_i:
                                b_i = dist
                sil_res += ((b_i - a_i) / max(b_i,a_i))
        sil_res /= self.row_len
        return sil_res
                

    # Good clustering finding method
    def fit(self, max_K ,random_iter, epoch, dist_func):
        dist_func = self.dist_func_set(dist_func)        
        self.cluster_points_good = None
        self.cluster_points_good_metric = -inf

        for K in range(2,max_K+1):
            for i in range(random_iter):
                cluster_points = self.cluster(K, epoch, dist_func)
                metric_val = self.metric(cluster_points, dist_func)
                if metric_val > self.cluster_points_good_metric:
                    self.cluster_points_good_metric = metric_val
                    self.cluster_points_good = cluster_points
                
        return self.cluster_points_good

