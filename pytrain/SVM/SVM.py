#
# SVM
#
# @ author becxer
# @ reference Machine Learning in Action by Peter Harrington
# @ e-mail becxer87@gmail.com
#

from numpy import *

class SVM:

    def __init__(self, _X, _Y, _C, _toler):
        self.X = _X
        self.Y = _Y
        self.m = shape()

        self.C = _C
        self.tol = _toler
        

    
