#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com#
#--------------------------------------------

from test_lib import *
from test_knn import *
from test_dtree import *

#test fs
test_fs(logging = True).process()

#test normalize
test_convert(logging = True).process()

#test batch test
test_batch(logging = True).process()

#test knn
test_basic_knn(logging = True).process()
#test_basic_knn_digit(logging = True).process()

#test dtree
test_basic_dtree(logging = True).process()
test_basic_dtree_store(logging = True).process()
