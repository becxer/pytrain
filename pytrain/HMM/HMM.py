#
# HMM
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
from pytrain.lib import convert

class HMM:

    def __init__(self, mat_data, label_data):
        self.mat_data = convert.list2npfloat(mat_data)
        self.label_data = convert.list2npfloat(label_data)

    def fit(self, lr, epoch, stoc):

        # TODO : Implement HMM code
        
        pass
    
    def predict(self, array_input):
        pass
