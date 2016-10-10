#
# library for convert data format
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *

def list2npfloat(list_data):
    ldtype = type(list_data).__name__
    if ldtype == 'str' or ldtype == 'long' or ldtype == 'int' or ldtype == 'int32' or\
      ldtype == 'int64' or ldtype == 'float' or ldtype == 'float32' or ldtype == 'float64':
        return float(list_data)
    elif ldtype == 'list':
        return array(map(list2npfloat, list_data))
    else :
        return list_data
