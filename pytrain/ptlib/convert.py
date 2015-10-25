#
# library for data manipulation & etc.
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
import operator
import math
import sys

def mat2arr(data_mat):
    return array(map(lambda x:map(float,x),data_mat))

def list2arr(data_list):
    return array(map(float,data_list))

#normalize matrix feature with base-min & base-max
def norm(data_mat):
    if data_mat.__class__.__name__ != 'ndarray':
        data_mat = mat2arr(data_mat)
    min_vals = data_mat.min(0)
    max_vals = data_mat.max(0)
    ranges = max_vals - min_vals
    ranges = map(lambda x : x + sys.float_info.epsilon ,ranges)
    normed_data_mat = zeros(shape(data_mat))
    rowsize = data_mat.shape[0]
    normed_data_mat = data_mat - tile(min_vals, (rowsize,1))
    normed_data_mat = normed_data_mat / tile(ranges,(rowsize,1))
    return normed_data_mat
