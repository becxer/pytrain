#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com#
# --------------------------------------------

FULL_TEST = False # Toggle for long test job

from test_lib import *
from test_KNN import *
from test_DecisionTreeID3 import *
from test_NaiveBayes import *
from test_GaussianNaiveBayes import *
from test_Apriori import *
from test_LinearRegression import *
from test_LogisticRegression import *
from test_NeuralNetwork import *
from test_Kmeans import *
from test_DBSCAN import *
from test_HierarchicalClustering import *
from test_SVM import *

# test lib modules
if FULL_TEST:
    test_dataset(logging = False).process()

test_fs(logging = False).process()
test_normalize(logging = False).process()
test_autotest(logging = False).process()
test_nlp(logging = False).process()

# test KNN
test_KNN_iris(logging = False).process()

if FULL_TEST:
    test_KNN_mnist(logging = False).process()

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

if FULL_TEST:
    test_Apriori_mushroom(logging = False).process()

# Test LinearRegression
test_LinearRegression(logging = False).process()
test_LinearRegression_iris(logging = False).process()
test_LinearRegression_mnist(logging = True).process()

# Test LogicticRegression
test_LogisticRegression(logging = False).process()

if FULL_TEST:
    test_LogisticRegression_iris(logging = False).process()

test_LogisticRegression_mnist(logging = True).process()

# Test FNN
test_FNN(logging = False).process()

if FULL_TEST:
    test_FNN_iris(logging = False).process()

test_FNN_mnist(logging = True).process()

# Test Kmeans
test_Kmeans(logging = False).process()

# Test DBSCAN
test_DBSCAN(logging = False).process()

# Test HierarchicalClustering
test_HierarchicalClustering(logging = False).process()

# Test SVM
test_BinarySVM(logging = True).process()

'''
# TODO --

test_SVM(logging = True).process()

#from test_HMM import *
#from test_VersionSpace import *
#from test_CRF import *

# Test HMM
test_HMM(logging = True).process()
'''
