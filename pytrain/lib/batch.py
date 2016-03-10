#
# library for batch processing module
#
# @ author becxer
# @ e-mail becxer@gmail.com
#

from numpy import *
import operator
import math
import sys


# abstracted evaluation logic
# p_module is pytrain module that you already trained
def eval_predict(p_module, mat_test, label_test, log_on = True):
    test_row_size = len(mat_test)
    error_count = 0.0
    for i in range(test_row_size):
        res = eval_predict_one(p_module, mat_test[i], label_test[i], log_on)
        if not res : error_count += 1.0
    if log_on: print "<" + p_module.__class__.__name__ + ">" +\
        " error rate is " + str(error_count / float(test_row_size))
    return error_count/test_row_size

def eval_predict_one(p_module, input_array_test, label_one_test, log_on = True):
    res = p_module.predict(input_array_test)
    if log_on : print "input : '" + str(input_array_test[:2]) + \
            "' --> predicted : '" + str(res) + "' --? origin : '" \
                    + str(label_one_test) + "'"
    if list(str(res)) != list(str(label_one_test)) :
        return False
    else :
        return True
