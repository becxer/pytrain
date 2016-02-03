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
        self.itemsets = []
        self.support_data = {}
        self.rules = []

    # make unique item itemsets of data set
    def gen_first_itemsets(self, mat_data):
        first_itemsets = []
        for row in mat_data:
            for item in row:
                if not [item] in first_itemsets:
                    first_itemsets.append([item])
        return map(frozenset, first_itemsets)

    # filtering and terminate item which under the min_support
    def filter_itemsets(self, mat_data, itemsets, min_support):
        itemset_occur_count = {}
        for row in mat_data:
            for item in itemsets:
                if item.issubset(row):
                    itemset_occur_count[item] = \
                            itemset_occur_count.get(item, 0) + 1

        filtered_itemsets = []
        support_data = {}
        for key in itemset_occur_count:
            support = itemset_occur_count[key] / self.len_data
            if support >= min_support:
                filtered_itemsets.append(key)
            support_data[key] = support
        return filtered_itemsets, support_data

    # generate itemsets from previous item
    def gen_next_itemsets(self, itemsets):
        next_itemsets = {}
        len_itemset = len(itemsets)
        for i in range(len_itemset):
            for j in range(i+1, len_itemset):
                next_itemsets[itemsets[i] | itemsets[j]] = 1
        return next_itemsets.keys()
    
    def fit(self, min_support, min_confidence):
        first_itemsets = self.gen_first_itemsets(self.mat_data)
        flt_first_itemsets, fspd = \
            self.filter_itemsets(self.mat_data, first_itemsets, min_support)
        res_itemsets = [flt_first_itemsets]
        res_spd = fspd
        now_itemsets = flt_first_itemsets
        while len(now_itemsets) > 0 :
            next_itemsets = self.gen_next_itemsets(now_itemsets)
            flt_itemsets, nspd = \
                self.filter_itemsets(self.mat_data, next_itemsets, min_support)
            res_spd.update(nspd)
            res_itemsets.append(flt_itemsets)
            now_itemsets = flt_itemsets

        self.itemsets = res_itemsets
        self.support_data = res_spd

        #TODO -> implement generating rules!!!

    def get_itemsets(self):
        return self.itemsets

    def get_support_data(self):
        return self.support_data

    def get_rules(self):
        return self.rules

    def recommend(self, array_input):
        pass

