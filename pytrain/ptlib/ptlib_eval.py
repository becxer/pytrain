#
# library for evaluation module
#
# @ author becxer
# @ e-mail becxer@gmail.com
#

from numpy import *
import operator
import math
import sys

#abstracted evaluation logic 
#p_module is pytrain module that you already trained
def eval_predict(p_module ,mat_test, label_test, log_on = True):
    rsize_test = len(mat_test)
    error_count = 0.0
    for i in range(rsize_test):
        res = p_module.predict(mat_test[i])
        if log_on : print "predicted : '" + str(res) + "' --- origin : '" \
                        + str(label_test[i]) + "'"
        if(res != label_test[i]): error_count += 1.0
    if log_on : print "<" + p_module.__class__.__name__ + ">" +\
                    " error rate is " + str(error_count / float(rsize_test))
    return error_count/rsize_test

