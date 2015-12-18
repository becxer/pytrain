#
# library for convert data format
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *


def mat2arr(data_mat):
    return array(map(lambda x:map(float,x),data_mat))


def list2arr(data_list):
    return array(map(float,data_list))

