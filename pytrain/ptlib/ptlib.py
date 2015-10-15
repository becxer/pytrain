#
# library for data manipulation & etc.
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

from numpy import *
import operator
import math
import sys

#convert file which format is 
#[label, feature1, feature2 ... , featureN]
#to matrix_train, label_train, matrix_test, label_test
#according to ho_ratio
#ho_ratio is test_set ratio how you want
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

def mat2arr(data_mat):
	return array(map(lambda x:map(float,x),data_mat))

def list2arr(data_list):
	return array(map(float,data_list))

#normalize matrix feature with base-min & base-max
def norm(data_mat):
	if data_mat.__class__.__name__ != 'ndarray':
		data_mat = mat2arr(data_mat)
	min_vals = data_mat.min(0)
	max_vals = data_mat.max(0)
	ranges = max_vals - min_vals
	ranges = map(lambda x : x + sys.float_info.epsilon ,ranges)
	normed_data_mat = zeros(shape(data_mat))
	rowsize = data_mat.shape[0]
	normed_data_mat = data_mat - tile(min_vals, (rowsize,1))
	normed_data_mat = normed_data_mat / tile(ranges,(rowsize,1))
	return normed_data_mat

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

#saving module to file
def store_module(module, filename):
	import pickle
	module_f = open(filename, 'w')
	pickle.dump(module,module_f)
	module_f.close()

#loading module into object
def restore_module(filename):
	import pickle
	module_f = open(filename)
	return pickle.load(module_f)

#test for ptlib
def hello():
	print "hello this is ptlib"

