#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com#
# --------------------------------------------

# Toggle dataset
IRIS = True # Toggle for IRIS dataset testing
MNIST = False # Toggle for MNIST dataset testing

import sys
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

# Test lib modules
test_dataset(logging = False).process()
test_fs(logging = False).process()
test_normalize(logging = False).process()
test_autotest(logging = False).process()
test_nlp(logging = False).process()

# Test DecisionTree
test_DecisionTreeID3(logging = False).process()
test_DecisionTreeID3_lense(logging = False).process()

# Test module storing
test_DecisionTreeID3_store(logging = False).process()

# Test NaiveBayes
test_NaiveBayes(logging = False).process()
test_NaiveBayes_email(logging = False).process()

# Test GaussianNaiveBayes
test_GaussianNaiveBayes(logging = False).process()
test_GaussianNaiveBayes_rssi(logging = False).process()

# test KNN
if IRIS :
    test_KNN_iris(logging = False).process()
if MNIST :
    test_KNN_mnist(logging = True).process()

# Test LinearRegression
test_LinearRegression(logging = False).process()
if IRIS :
    test_LinearRegression_iris(logging = False).process()
if MNIST :
    test_LinearRegression_mnist(logging = False).process()
    
# Test LogicticRegression
test_LogisticRegression(logging = False).process()
if IRIS :
    test_LogisticRegression_iris(logging = False).process()
if MNIST :
    test_LogisticRegression_mnist(logging = False).process()
    
# Test FNN
test_FNN(logging = False).process()
if IRIS :
    test_FNN_iris(logging = False).process()
if MNIST:
    test_FNN_mnist(logging = False).process()

# Test Apriori
test_Apriori(logging = False).process()
test_Apriori_mushroom(logging = False).process()

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
