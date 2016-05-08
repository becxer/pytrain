#
# test HMM
#
# @ author becxer
# @ e-mail becxer87@gmail.com
# 

from test_pytrain import test_Suite
from pytrain.HMM import HMM
from pytrain.lib import batch
from numpy import *

class test_HMM(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):

        train_mat = [\
                     [0.12, 0.25],\
                     [3.24, 4.33],\
                     [0.14, 0.45],\
                     [7.30, 4.23],\
                     ]
        train_label = [0,1,0,1] # out bit is 1
        
        hm = HMM(train_mat, train_label)
        r1 = batch.eval_predict_one(hm, [4.40,4.37], 1, self.logging)
        
