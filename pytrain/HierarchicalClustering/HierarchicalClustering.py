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
        self.unique_idx = 0
        
    def fit(self):
        return self.cluster()

    unique_idx = 0
    group_list = []
    dist_list = []
    group_map = {}

    class Group:
        data_idx = []  # original data index
        vector = [] # group's vector
        def __init__(self, vt, didx):
            HierarchicalClustering.unique_idx += 1
            self.unique_idx = HierarchicalClustering.unique_idx
            self.vector = vt
            self.data_idx.extend(didx)
            HierarchicalClustering.group_map[self.unique_idx] = self

    class Dist:
        def __init__(self, s_grp, t_grp, dist_func):
            self.src_idx = s_grp.unique_idx
            self.trg_idx = t_grp.unique_idx
            self.distance = dist_func(s_grp.vector, t_grp.vector)

    def remove_from_dist_list(self, grp_idx):
        for dist_obj in self.dist_list:
            if dist_obj.src_idx == grp_idx or \
              dist_obj.trg_idx == grp_idx:
              self.dist_list.remove(dist_obj)

    def remove_from_group_list(self, grp_idx):
        for grp in self.group_list:
            if grp.unique_idx == grp_idx:
                self.group_list.remove(grp)
                break
        
    def insert_new_group(self, grp):
        for oth in self.group_list:
            new_dis = self.Dist(grp, oth, self.dist_func)
            for idx, old_dis in enumerate(self.dist_list):
                if new_dis.distance <= old_dis.distance:
                    self.dist_list.insert(idx, new_dis)
                    break
                    
        self.group_list.append(grp)
    
    def merge_group(self, grp_1_idx, grp_2_idx):
        grp_1 = self.group_map[grp_1_idx]
        grp_2 = self.group_map[grp_2_idx]
        
        mgd_vt = ( (grp_1.vector * len(grp_1.data_idx)) \
                       + (grp_2.vector + len(grp_2.data_idx)) ) \
                           / (len(grp_1.data_idx) + len(grp_2.data_idx))
        mgd_didx = []
        mgd_didx.extend(grp_1.data_idx)
        mgd_didx.extend(grp_2.data_idx)
        mgd_grp = self.Group(mgd_vt, mgd_didx)
        return mgd_grp
    
    def cluster(self):
        # make initial groups
        self.group_list = [ self.Group(vt,[idx]) for idx, vt in enumerate(self.mat_data)]
        # make dist_list
        for i, src_g in enumerate(self.group_list):
            for j in range(i+1,len(self.group_list)):
                trg_g = self.group_list[j]
                self.dist_list.append(self.Dist(src_g, trg_g, self.dist_func))

        # merge group until length of group list less than K
        self.dist_list.sort(key=lambda x : x.distance,reverse=False)
        while len(self.group_list) > self.K :
            selected_dist = self.dist_list.pop()
            new_group = self.merge_group(selected_dist.src_idx, selected_dist.trg_idx)

            self.remove_from_dist_list(selected_dist.src_idx)
            self.remove_from_dist_list(selected_dist.trg_idx)
            self.remove_from_group_list(selected_dist.src_idx)
            self.remove_from_group_list(selected_dist.trg_idx)
        
            self.insert_new_group(new_group)
            
        # loop group list & fill label data
        self.label_data = [-1 for x in range(len(self.mat_data))]
        for grp in self.group_list:
            for idx in grp.data_idx:
                self.label_data[idx] = grp.unique_idx
        return self.label_data

    # default predictor using KNN
    def predict(self, input_array):
        # TODO
        input_array = convert.list2npfloat(input_array)
        pass
