#
# Basic Decision Tree
#
# @ author becxer
# @ reference Machine Learning in Action by Peter Harrington
# @ e-mail becxer87@gmail.com
#

from numpy import *
from math import log
import operator

class DecisionTreeID3:
    def __init__(self, mat_data, label_data):
        self.mat_data = mat_data
        self.label_data = label_data
        self.tree = {}

    # make tree with matrix_data & label_data
    def fit(self):
        self.tree = self.create_tree(self.mat_data,self.label_data)
        return self.tree

    def build(self):
        return self.fit()

    # search array_input in tree
    def predict(self, array_input):
        return self.search_tree(self.tree, array_input)

    # search array_input's feature in tree recursively
    # if tree node is dictionary recursive
    # else return label data
    def search_tree(self, tree, array_input):
        searched_label = "not found"
        node_col = tree.keys()[0]
        node_dict = tree[node_col]
        for node_val in node_dict.keys():
            if array_input[node_col] == node_val:
                if type(node_dict[node_val]).__name__ == 'dict':
                    next_input = array_input[:node_col]
                    next_input.extend(array_input[node_col+1:])    
                    searched_label = self.search_tree(node_dict[node_val],next_input)
                else : searched_label = node_dict[node_val]
        return searched_label

    # create tree to lower entropy recursively
    # when split data, calculate each feature split matrix entropy and compare
    # select most lower entropy and split
    # Example)  matrix => label ::  [[A,B] , [A,C], [A,D], [A,E] ,[B,D]] => ['YES','YES','YES','YES',NO' ]
    # output example tree )   { 0 , { 'A' : 'YES', 'B' : 'No'}}
    #                          |      |      |     |     |
    #                          |      |      |     |     |
    #                       column  value    |    value  |
    #                                      label        label
    #
    def create_tree(self, mat_data, label_data):
        # if left data has same label, then return label
        if label_data.count(label_data[0]) == len(label_data):
            return label_data[0]
        # if there is no feature to split, then return most major label
        if len(mat_data[0]) == 0 or ( len(mat_data[0]) == 1 and \
                len(set([row[0] for row in mat_data])) == 1 ) :
            return self.major_label_count(label_data)
        best_col_index = self.choose_col_to_split(mat_data, label_data)
        tree = {best_col_index:{}}
        best_col = [row[best_col_index] for row in mat_data]
        uniq_val = list(set(best_col)) + [None]
        for val in uniq_val:
            split_mat, split_label = self.split_data(\
                                mat_data, label_data, best_col_index, val)
            tree[best_col_index][val] = self.create_tree(split_mat, split_label)
        return tree

    # split matrix & label data with axis and it's value
    def split_data(self, mat_data, label_data, axis, split_value):
        ret_data = []
        ret_label = []
        for index, row in enumerate(mat_data):
            if row[axis] == split_value or split_value == None:
                temp = row[:axis]
                temp.extend(row[axis+1:])
                ret_data.append(temp)
                ret_label.append(label_data[index])
        return ret_data, ret_label
    
    # choose column to split comparing entropy
    def choose_col_to_split(self, mat_data, label_data):
        num_cols = len(mat_data[0]) 
        base_ent = self.calc_shannon_ent(label_data)
        max_info = 0.0
        best_col = -1
        for i in range(num_cols):
            col = [row[i] for row in mat_data]
            uniq_col = set(col)
            new_ent = 0.0
            for val in uniq_col:
                split_mat_data, split_label_data = \
                         self.split_data(mat_data, label_data, i ,val)
                prob = len(split_label_data) / float(len(label_data))
                new_ent += prob * self.calc_shannon_ent(split_label_data)
            info = base_ent - new_ent
            if info >= max_info:
                max_info = info
                best_col = i
        return best_col

    def calc_shannon_ent(self, label_data):
        num_entry = len(label_data)
        label_count = {}
        for label in label_data:
            if label not in label_count.keys():
                label_count[label] = 0
            label_count[label] += 1
        shannon_ent = 0.0
        for key in label_count:
            prob = float(label_count[key]) / num_entry
            shannon_ent -= prob * log(prob,2)
        return shannon_ent    

    def major_label_count(self, label_data):
        label_count = {}
        for label in label_data:
            if label not in label_count.keys():
                label_count[label] = 0
            label_count[label] += 1
        sorted_label_count = sorted(label_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
        return sorted_label_count[0][0]


