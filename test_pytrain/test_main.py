#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com#
#--------------------------------------------

from test_lib import *
from test_knn import *
from test_dtree import *
from test_nbayes import *

# test fs
test_fs(logging = False).process()

# test normalize
test_normalize(logging = False).process()

# test batch test
test_batch(logging = False).process()

# test nlp test
test_nlp(logging = True).process()

# test knn
test_basic_knn(logging = False).process()
#test_basic_knn_digit(logging = False).process()

# test dtree
test_basic_dtree(logging = False).process()
test_basic_dtree_store(logging = False).process()
test_basic_dtree_lense(logging = False).process()

# test nbayes
test_basic_nbayes(logging = True).process()

