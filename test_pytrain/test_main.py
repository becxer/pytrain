#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com#
# --------------------------------------------

from test_lib import *
from test_KNN import *
from test_Tree import *
from test_NaiveBayes import *

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
# test_KNN_digit(logging = False).process()

# test DecisionTree
test_DecisionTree(logging = False).process()
test_DecisionTree_store(logging = False).process()
test_DecisionTree_lense(logging = False).process()

# Test NaiveBayes
test_NaiveBayes(logging = False).process()
test_NaiveBayes_email(logging = False).process()

