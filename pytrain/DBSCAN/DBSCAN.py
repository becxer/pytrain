#
# DBSCAN
#
# @ author becxer
# @ email becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert
from pytrain.lib import ptmath
import operator

class DBSCAN:
    
    def __init__(self, mat_data, eps, min_pts, dist_func):
        self.mat_data = convert.list2npfloat(mat_data)
        self.dist_func = ptmath.distfunc(dist_func)
        self.eps = eps
        self.min_pts = min_pts
        self.col_len = len(self.mat_data[0])
        self.row_len = len(self.mat_data)
        
    def fit(self):
        return self.cluster()
    
    def cluster(self):
        # border point : connected to corepoint but less than 'min_pts' neighbor
        # core point : neighbor more than 'min_pts' in 'eps' area
        map_of_data = [[] for x in range(self.row_len)]
        core_point_flag = [False for x in range(self.row_len)]
        
        visited = [False for x in range(self.row_len)]
        label_of_data = [None for x in range(self.row_len)]

        # 1. struct map (with eps-neighbor of a point)
        for i, src in enumerate(self.mat_data):
            for j, trg in enumerate(self.mat_data):
                if (i is not j) and self.dist_func(src, trg) <= self.eps :
                    map_of_data[i].append(j)
                    if len(map_of_data[i]) >= self.min_pts:
                        core_point_flag[i] = True
                        
        # 2. DFS map search and tag label
        cluster_idx = 0
        for idx, core in enumerate(core_point_flag):
            if core and not visited[idx]:
                stack = [idx]
                while len(stack) != 0 :
                    now_p = stack.pop()
                    if not visited[now_p] :
                        label_of_data[now_p] = cluster_idx
                        visited[now_p] = True
                        if core_point_flag[now_p]:
                            for next_p in map_of_data[now_p]:
                                if not visited[next_p] :
                                    stack.append(next_p)
                cluster_idx += 1

        self.label_of_data = label_of_data
        return label_of_data

    # default predictor using KNN
    def predict(self, input_array):
        input_array = convert.list2npfloat(input_array)
        label_count = {}
        for idx, trg in enumerate(self.mat_data):
            if self.dist_func(input_array, trg) <= self.eps:
                label = self.label_of_data[idx]
                label_count[label] = label_count.get(label, 0) + 1
        sorted_label_count = sorted(label_count.iteritems()\
                , key=operator.itemgetter(1), reverse=True)
        return sorted_label_count[0][0]

