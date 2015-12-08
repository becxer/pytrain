#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com
#

#import from ptlib
from pytrain.ptlib import batch
from pytrain.ptlib import convert
from pytrain.ptlib import nlp
from pytrain.ptlib import fs

#import from modules
from pytrain.knn import basic_knn
from pytrain.dtree import basic_dtree
from pytrain.nbayes import basic_nbayes

#testing fs.f2mat
print "-Testing file 2 matrix"
dmat_train, dlabel_train, dmat_test, dlabel_test \
    = fs.f2mat("sample/dating/date_info.txt", 0.1)

print "train mat count : " + str(len(dmat_train))
print dmat_train[0:10]
print "train label count : " + str(len(dlabel_train))
print dlabel_train[0:10]

print "test mat count : " + str(len(dmat_test))
print dmat_test[0:10]
print "test label count : " + str(len(dlabel_test))
print dlabel_test[0:10]

#testing convert
print "-Testing normalization"
normed_dmat_train = convert.norm(dmat_train)
normed_dmat_test = convert.norm(dmat_test)
print normed_dmat_train[0:10]

#testing knn
print "-Testing KNN"
sample_mat_1 = [[1.0,1.1] , [1.0,1.0], [0,0], [0,0.1]]
sample_label_1 = ['A','A','B','B']
knn = basic_knn(sample_mat_1, sample_label_1, 3)
print "knn predict [0.9,0.9] : " + str(knn.predict([0.9,0.9]))
print "knn predict [0.1,0.4] : " + str(knn.predict([0.1,0.4]))

#batch knn date
knn_date = basic_knn(normed_dmat_train, dlabel_train, 3)
error_rate = batch.eval_predict(knn_date, normed_dmat_test, dlabel_test, False)
print "<basic knn> date error rate : " + str(error_rate)

#batch knn digits
dg_mat_train, dg_label_train = fs.f2mat("sample/digit/digit-train.txt",0)
dg_mat_test, dg_label_test = fs.f2mat("sample/digit/digit-test.txt",0)
knn_digit = basic_knn(dg_mat_train, dg_label_train, 3)
error_rate = batch.eval_predict(knn_digit, dg_mat_test, dg_label_test)

#testing dtree
print "-Testing Dtree"
sample_mat_2 = [[7,8,8],[8,7,8],[8,8,8],[8,8,8],[8,7,7],[7,7,8],[7,7,7],[7,8,7],[8,8,8]]
sample_label_2 = ['yes',  'yes',  'yes', 'no',  'no',  'yes',   'no',   'no', 'no']
tree = basic_dtree(sample_mat_2, sample_label_2)
print "tree fit : " + str(tree.fit())
print "tree predict : " + str(tree.predict([8,8,8]))

#testing store & restore
print "-Testing store & restore"
fs.store_module(tree,"tmp/tree_878_store_test.dat")
mod = fs.restore_module("tmp/tree_878_store_test.dat")
print "restored tree : " + str(mod.tree)
print "restored tree predict : " + str(mod.predict([8,8,7]))

#batch dtree lense
lense_mat_train, lense_label_train, lense_mat_test, lense_label_test = \
                            fs.f2mat("sample/lense/lense.txt", 0.4)
dtree_lense = basic_dtree(lense_mat_train,lense_label_train)
dtree_lense.fit()
error_rate = batch.eval_predict(dtree_lense, lense_mat_test, lense_label_test)

#tesing nbayes
print "-Tesing Nbayes"
sample_text_mat = [\
        ['hello','this','is','virus','mail'],\
        ['hi','this','is','from', 'friend'],\
        ['how','about','buy','this','virus'],\
        ['facebook','friend','contact','to','you'],\
        ['I','love','you','baby','virus'],\
        ['what','nice','day','how','about','you']
        ]
sample_text_label = ['spam','real','spam','real','spam','real']
sample_mat_3 = [[]]
sample_label_3 = []
nbayes = basic_nbayes(sample_mat_3,sample_label_3)
print "nbayes fit : " + str(nbayes.fit())
print "nbayes predict : " + str(nbayes.predict([]))

#batch nbayes --TODO
