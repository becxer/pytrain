#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com
#

#import from ptlib
from pytrain.ptlib import ptlib

#import from modules
from pytrain.knn import naive_knn
from pytrain.dtree import naive_dtree

#testing ptlib
ptlib.hello()

#testing ptlib.f2mat
print "-Testing file 2 matrix"
dmat_train, dlabel_train, dmat_test, dlabel_test \
	= ptlib.f2mat("sample/dating/date_info.txt", 0.1)

print "train mat count : " + str(len(dmat_train))
print dmat_train[0:10]
print "train label count : " + str(len(dlabel_train))
print dlabel_train[0:10]

print "test mat count : " + str(len(dmat_test))
print dmat_test[0:10]
print "test label count : " + str(len(dlabel_test))
print dlabel_test[0:10]

#testing ptlib.norm
print "-Testing normalization"
normed_dmat_train = ptlib.norm(dmat_train)
normed_dmat_test = ptlib.norm(dmat_test)
print normed_dmat_train[0:10]

#testing knn
print "-Testing KNN"
simple_mat_1 = [[1.0,1.1] , [1.0,1.0], [0,0], [0,0.1]]
simple_label_1 = ['A','A','B','B']
knn = naive_knn(simple_mat_1, simple_label_1, 3)
print "knn predict [0.9,0.9] : " + str(knn.predict([0.9,0.9]))
print "knn predict [0.1,0.4] : " + str(knn.predict([0.1,0.4]))

#eval knn date
knn_date = naive_knn(normed_dmat_train, dlabel_train, 3)
error_rate = ptlib.eval_predict(knn_date, normed_dmat_test, dlabel_test, False)
print "<knn> date error rate : " + str(error_rate)

#eval knn digits
#dg_mat_train, dg_label_train = ptlib.f2mat("sample/digit/digit-train.txt",0)
#dg_mat_test, dg_label_test = ptlib.f2mat("sample/digit/digit-test.txt",0)
#knn_digit = naive_knn(dg_mat_train, dg_label_train, 3)
#error_rate = ptlib.eval_predict(knn_digit, dg_mat_test, dg_label_test)

#testing dtree
print "-Testing Dtree"
simple_mat_2 = [[7,8,8],[8,7,8],[8,8,8],[8,8,8],[8,7,7],[7,7,8],[7,7,7],[7,8,7],[8,8,8]]
simple_label_2 = ['yes',  'yes',  'yes', 'no',  'no',  'yes',   'no',   'no', 'no']
tree = naive_dtree(simple_mat_2, simple_label_2)
print "tree fit : " + str(tree.fit())
print "tree predict : " + str(tree.predict([8,8,8]))

#testing store & restore
print "-Testing store & restore"
ptlib.store_module(tree,"tmp/tree_878.dat")
mod = ptlib.restore_module("tmp/tree_878.dat")
print "restored tree : " + str(mod.tree)
print "restored tree predict : " + str(mod.predict([8,8,7]))

#eval dtree lense
lense_mat_train, lense_label_train, lense_mat_test, lense_label_test = \
							ptlib.f2mat("sample/lense/lense.txt", 0.4)
dtree_lense = naive_dtree(lense_mat_train,lense_label_train)
dtree_lense.fit()
error_rate = ptlib.eval_predict(dtree_lense, lense_mat_test, lense_label_test)

