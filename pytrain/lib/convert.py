#
# library for convert data format
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

import numpy as np

def list2npfloat(list_data):
    ldtype = type(list_data).__name__
    if ldtype == 'str' or ldtype == 'long' or ldtype == 'int' or ldtype == 'int32' or\
      ldtype == 'int64' or ldtype == 'float' or ldtype == 'float32' or ldtype == 'float64':
        return float(list_data)
    elif ldtype == 'list':
        return np.array(map(list2npfloat, list_data), dtype=np.float64)
    else :
        return list_data
