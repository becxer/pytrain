#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com
#

#--------------------------------------------
from test_lib import *

#test fs
test_fs(logging = True).process()
#test normalize
test_convert(logging = True).process()
#test knn
from test_knn import *
test_basic_knn(logging = True).process()
#test batch test
test_batch(logging = True).process()
