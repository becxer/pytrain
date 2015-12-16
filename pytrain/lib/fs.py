# library for data manipulation & etc.
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
import operator
import math
import sys

# convert file which format is
# [label, feature1, feature2 ... , featureN]
# to matrix_train, label_train, matrix_test, label_test
# according to ho_ratio
# ho_ratio is test_set ratio how you want


def f2mat(filename, ho_ratio):
    fr = open(filename)
    lines = fr.readlines()
    lnum_test = math.ceil(len(lines) * ho_ratio) 
    lnum_train = len(lines) - lnum_test
    colmax = len(lines[0].strip().split('\t'))
    mat_train = []
    mat_test = [] 
    label_train = []
    label_test = []
    train_index = 0
    test_index = 0
    split_index = 0
    if ho_ratio != 0 :
        split_index = 1.0 / ho_ratio
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        if ho_ratio == 0 or (train_index + test_index) % split_index != 0 :
            mat_train.append(listFromLine[1:colmax])
            label_train.append(listFromLine[0])
            train_index += 1
        else :
            mat_test.append(listFromLine[1:colmax])
            label_test.append(listFromLine[0])
            test_index += 1
    if ho_ratio == 0:
        return mat_train,label_train
    else :
        return mat_train, label_train, mat_test, label_test


# saving module to file
def store_module(module, filename):
    import pickle
    module_f = open(filename, 'w')
    pickle.dump(module,module_f)
    module_f.close()

# loading module into object
def restore_module(filename):
    import pickle
    module_f = open(filename)
    return pickle.load(module_f)

