#
# Apriori 
#
# @ author becxer
# @ reference Machine Learning in Action by Peter Harrington
# @ e-mail becxer87@gmail.com
#

from numpy import *

class Apriori:

    def __init__(self, mat_data):
        self.mat_data = mat_data
        self.len_data = float(len(mat_data))
        self.itemsets_list = []
        self.support_data = {}
        self.rules = []

    def fit(self, min_support, min_confidence):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets_list, self.support_data = \
            self.generate_itemsets_support_data(min_support)
        self.rules = \
            self.generate_rules(min_confidence)
    
    # make unique item itemsets of data set
    def gen_first_itemsets(self):
        first_itemsets = []
        for row in self.mat_data:
            for item in row:
                if not [item] in first_itemsets:
                    first_itemsets.append([item])
        return map(frozenset, first_itemsets)
    
    # generate itemsets from previous item
    def gen_next_itemsets(self, itemsets):
        next_itemsets = {}
        len_itemset = len(itemsets)
        for i in range(len_itemset):
            for j in range(i+1, len_itemset):
                next_itemsets[itemsets[i] | itemsets[j]] = 1
        return next_itemsets.keys()

    # filtering and terminate item which under the min_support
    def filter_itemsets(self, itemsets, min_support):
        itemset_occur_count = {}
        for row in self.mat_data:
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

    def generate_itemsets_support_data(self, min_support):
        first_itemsets = self.gen_first_itemsets()
        flt_first_itemsets, fspd = \
            self.filter_itemsets(first_itemsets, min_support)
        res_itemsets = [flt_first_itemsets]
        res_spd = fspd
        now_itemsets = flt_first_itemsets
        while len(now_itemsets) > 0 :
            next_itemsets = self.gen_next_itemsets(now_itemsets)
            flt_itemsets, nspd = \
                self.filter_itemsets(next_itemsets, min_support)
            res_spd.update(nspd)
            res_itemsets.append(flt_itemsets)
            now_itemsets = flt_itemsets
        return res_itemsets, res_spd

    def generate_rules(self, min_confidence):
        res_rules = []
        for itemsets in self.itemsets_list:
            for itemset in itemsets:
                first_itemsets = [frozenset([item]) for item in itemset]
                new_itemsets = first_itemsets
                while len(new_itemsets) > 0 :
                    now_itemsets = new_itemsets
                    new_itemsets = []
                    for toset in now_itemsets:
                        if len(toset) < len(itemset):
                            fromset = itemset - toset
                            conf = self.support_data[itemset] / self.support_data[fromset]
                            if conf > min_confidence:
                                res_rules.append((fromset,toset,conf))
                                new_itemsets.append(toset)
                    new_itemsets = self.gen_next_itemsets(new_itemsets)
        return res_rules

    def get_itemsets(self):
        return self.itemsets_list

    def get_support_data(self):
        return self.support_data

    def get_rules(self):
        return self.rules

    def recommend(self, array_input):
        user_input = frozenset(array_input)
        res = frozenset([]) 
        for itemsets in self.itemsets_list:
            for itemset in itemsets:
                if user_input.issubset(itemset):
                    res = res|(itemset-user_input)
        return res

