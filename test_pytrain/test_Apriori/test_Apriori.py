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
        ap.fit(min_support = 0.5 , min_confidence = 0.5)
        
        itemsets = ap.get_itemsets()
        support_data = ap.get_support_data()
        rules = ap.get_rules()
        
        self.tlog("itemsets : "+str(itemsets))
        self.tlog("support data : " + str(support_data))
        self.tlog("rules : "+str(rules))

        rec = ap.recommend([2])
        self.tlog("recommend with 2 : " + str(rec))

