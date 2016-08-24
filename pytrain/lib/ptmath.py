#
# library for mathmatics using by pytrain
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *

def euclidean (vx, vy):
    return linalg.norm(vx - vy)

def manhattan (vx, vy):
    return array(map(abs, vx - vy)).sum(axis = 0)

def cosine_similarity (vx, vy):
    return dot(vx, vy) / (linalg.norm(vx) * linalg.norm(vy))

def sigmoid(k):
    return 1.0 / ( 1.0 + exp(-k))

def sigmoid_delta(k):
    return sigmoid(k) * (1.0 - sigmoid(k))

dfunc_set = {\
                 'euclidean' : euclidean,\
                 'default' : euclidean,\
                 'manhattan' : manhattan,\
                 'cosine_similarity' : cosine_similarity\
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
    
