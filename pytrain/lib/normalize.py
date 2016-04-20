#
# library for normalize
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
import operator
import math
import sys
from pytrain.lib import convert

# normalize matrix feature with base-min & base-max
def quantile(data_mat):
    data_mat = convert.list2npfloat(data_mat)
    min_vals = data_mat.min(0)
    max_vals = data_mat.max(0)
    ranges = max_vals - min_vals
    ranges = map(lambda x : x + sys.float_info.epsilon ,ranges)
    normalized_data_mat = zeros(shape(data_mat))
    rowsize = data_mat.shape[0]
    normalized_data_mat = data_mat - tile(min_vals, (rowsize,1))
    normalized_data_mat = normalized_data_mat / tile(ranges,(rowsize,1))
    return normalized_data_mat
