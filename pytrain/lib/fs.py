# library for file system processing
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
import operator
import math
import sys
from pytrain.lib import nlp


# convert file which format is
# [label, feature1, feature2 ... , featureN]
# to matrix_train, label_train, matrix_test, label_test
# according to ho_ratio
# ho_ratio is test_set ratio how you want
def f2mat(filename, ho_ratio):
    fr = open(filename)
    lines = fr.readlines()
    col_max = len(lines[0].strip().split('\t'))
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
        list_from_line = line.split('\t')
        if ho_ratio == 0 or (train_index + test_index) % split_index != 0 :
            mat_train.append(list_from_line[1:col_max])
            label_train.append(list_from_line[0])
            train_index += 1
        else :
            mat_test.append(list_from_line[1:col_max])
            label_test.append(list_from_line[0])
            test_index += 1
    if ho_ratio == 0:
        return mat_train,label_train
    else :
        return mat_train, label_train, mat_test, label_test

def f2wordmat(filename, ho_ratio):
    fr = open(filename)
    lines = fr.readlines()
    print lines
    mat_train = []
    mat_test = [] 
    label_train = []
    label_test = []

    train_index = 0
    test_index = 0
    split_index = 0
    if ho_ratio != 0:
        split_index = 1.0 / ho_ratio
    for line in lines:
        line = line.strip()
        list_from_line = line.split('\t')
        if ho_ratio == 0 or (train_index + test_index) % split_index != 0 :
            mat_train.append(list_from_line[1:])
            label_train.append(list_from_line[0])
            train_index += 1
        else :
            mat_test.append(list_from_line[1:])
            label_test.append(list_from_line[0])
            test_index += 1

    vocabulary = nlp.extract_vocabulary(mat_train)
    if ho_ratio == 0:
        return mat_train,label_train, vocabulary
    else :
        return mat_train, label_train, vocabulary, mat_test, label_test

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

