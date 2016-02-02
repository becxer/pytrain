#
# Apriori 
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *

class Apriori:

    def __init__(self, mat_data):
        self.mat_data = mat_data
        self.len_data = float(len(mat_data))
        self.clusters = []
        self.support_data = {}

    # make unique item clusters of data set
    def gen_first_clusters(self, mat_data):
        first_clusters = []
        for row in mat_data:
            for item in row:
                if not [item] in first_clusters:
                    first_clusters.append([item])
        return map(frozenset, first_clusters)

    # filtering and terminate cluster which under the min_support
    def filter_clusters(self, mat_data, clusters, min_support):
        cl_occur_count = {}
        for row in mat_data:
            for cl in clusters:
                if cl.issubset(row):
                    cl_occur_count[cl] = cl_occur_count.get(cl, 0) + 1

        filtered_clusters = []
        support_data = {}
        for key in cl_occur_count:
            support = cl_occur_count[key] / self.len_data
            if support >= min_support:
                filtered_clusters.append(key)
            support_data[key] = support
        return filtered_clusters, support_data

    # generate clusters from previous cluster
    def gen_next_clusters(self, clusters):
        next_clusters = {}
        len_cl = len(clusters)
        for i in range(len_cl):
            for j in range(i+1, len_cl):
                next_clusters[clusters[i] | clusters[j]] = 1
        return next_clusters.keys()
    
    def cluster(self, min_support):
        fcl = self.gen_first_clusters(self.mat_data)
        flt_fcl, fspd = self.filter_clusters(self.mat_data, fcl, min_support)
        res_cl = [flt_fcl]
        res_spd = fspd
        now_cl = flt_fcl
        while len(now_cl) > 0 :
            next_cl = self.gen_next_clusters(now_cl)
            flt_ncl, nspd = self.filter_clusters(self.mat_data, next_cl, min_support)
            res_spd.update(nspd)
            res_cl.append(flt_ncl)
            now_cl = flt_ncl

        self.clusters = res_cl
        self.support_data = res_spd
        return res_cl, res_spd

    def recommend(self, array_input):
        pass

