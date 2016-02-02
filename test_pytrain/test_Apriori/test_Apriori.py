#
# test Apriori 
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.Apriori import Apriori

class test_Apriori(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        data = [[1,3,4], [2,3,5],\
                [1,2,3,5], [2,5]]

        ap = Apriori(data)
        cl,spd = ap.cluster(0.7)
        self.tlog("cluster :: "+str(cl))
        self.tlog("support data :: " + str(spd))

