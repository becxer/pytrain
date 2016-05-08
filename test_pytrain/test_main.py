#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com#
# --------------------------------------------

from test_lib import *
from test_KNN import *
from test_DecisionTreeID3 import *
from test_NaiveBayes import *
from test_GaussianNaiveBayes import *
from test_LinearRegression import *
from test_LogisticRegression import *
from test_SVM import *
from test_Apriori import *
from test_Kmeans import *
from test_DBSCAN import *
from test_HierarchicalClustering import *
from test_HMM import *

# test fs lib
test_fs(logging = False).process()

# test normalize lib
test_normalize(logging = False).process()

# test batch lib
test_batch(logging = False).process()

# test nlp lib
test_nlp(logging = False).process()

# test KNN
test_KNN(logging = False).process()

# <Take too much time>
# test_KNN_digit(logging = False).process()

# test DecisionTree
test_DecisionTreeID3(logging = False).process()
test_DecisionTreeID3_store(logging = False).process()
test_DecisionTreeID3_lense(logging = False).process()

# Test NaiveBayes
test_NaiveBayes(logging = False).process()
test_NaiveBayes_email(logging = False).process()

# Test GaussianNaiveBayes
test_GaussianNaiveBayes(logging = False).process()
test_GaussianNaiveBayes_rssi(logging = False).process()

# Test Apriori
test_Apriori(logging = False).process()
# test_Apriori_mushroom(logging = False).process()

# Test LinearRegression
test_LinearRegression(logging = False).process()
test_LinearRegression_horse(logging = False).process()

# Test LogicticRegression
test_LogisticRegression(logging = False).process()
test_LogisticRegression_horse(logging = False).process()

# Test Kmeans
test_Kmeans(logging = False).process()

# Test DBSCAN
test_DBSCAN(logging = False).process()

# Test HierarchicalClustering
test_HierarchicalClustering(logging = False).process()

# Test SimpleSVM
test_SimpleSVM(logging = False).process()

# Test HMM
test_HMM(logging = False).process()

