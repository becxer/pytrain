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
        data = [[1,3,4], [2,3,5],[1,2,3,5], [2,5]]

        ap = Apriori(data)
        ap.fit(min_support = 0.5 , min_confidence = 0.7)
        
        itemsets_list = ap.get_itemsets()
        support_data = ap.get_support_data()
        rules = ap.get_rules()

        self.tlog("itemsets list : " + str(itemsets_list))
        self.tlog("support_data : " + str(support_data))
        self.tlog("rules : " + str(rules))
        
        rec = ap.recommend([2])
        self.tlog("recommend with 2 : " + str(rec))
        assert rec == frozenset([3,5])

class test_Apriori_mushroom(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        mushroom_file = open("sample_data/mushroom/mushroom.tsv")
        data = map(lambda x : x.strip().split(), mushroom_file.read().split("\n"))[:-1]
        ap = Apriori(data)
        ap.fit(min_support = 0.9, min_confidence = 0.8)
        itemsets_list = ap.get_itemsets()
        support_data = ap.get_support_data()
        rules = ap.get_rules()  

        self.tlog("itemsets list : " + str(itemsets_list))
        self.tlog("support_data : " + str(support_data))
        self.tlog("rules : " + str(rules))
