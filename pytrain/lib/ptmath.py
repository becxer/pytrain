#
# library for mathmatics using by pytrain
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *

def euclidean (vx, vy):
    return linalg.norm(vx - vy)

dfunc_set = {\
                 'euclidean' : euclidean,\
                 'default' : euclidean\
            }

def distfunc(dfunc_keyword):
    dfunc = dfunc_keyword
    if type(dfunc).__name__ == 'str':
        if dfunc in dfunc_set:
            dfunc = dfunc_set[dfunc]
    if type(dfunc).__name__ == 'function':
            return dfunc
    else :
        return None
    
