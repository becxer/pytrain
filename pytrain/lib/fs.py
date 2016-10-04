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
def csv_loader(filename, ho_ratio):
    fr = open(filename)
    lines = fr.readlines()
    mat_train = []
    mat_test = [] 
    label_train = []
    label_test = []
    train_index = 0
    test_index = 0
    split_index = 0
    if ho_ratio != 0 :
        split_index = math.floor(1.0 / ho_ratio)
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
    if ho_ratio == 0:
        return mat_train,label_train
    else :
        return mat_train, label_train, mat_test, label_test


def csv_loader_with_nlp(filename, ho_ratio, nlp_lib):
    wmat = csv_loader(filename, ho_ratio)
    wmat_train, label_train  = wmat[:2]

    mat_train = []
    mat_test = []
    label_test = []
    
    vocabulary = nlp_lib.extract_vocabulary(wmat_train)
    
    for row in wmat_train:
        mat_train.append(nlp_lib.bag_of_word2vector(vocabulary, row))

    if len(wmat) > 2 and ho_ratio != 0:
        wmat_test, label_test = wmat[2:4]
        for row in wmat_test:
            mat_test.append(nlp_lib.bag_of_word2vector(vocabulary, row))

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

