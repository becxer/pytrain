#
# test Decision Tree
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.DecisionTreeID3 import DecisionTreeID3
from pytrain.lib import fs
from pytrain.lib import autotest

class test_DecisionTreeID3(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        
        sample_mat = [['sunny','cloudy','rain'],['cloudy','sunny','rain'],['cloudy','cloudy','rain'],\
                ['cloudy','cloudy','rain'],['cloudy','sunny','sunny'],['sunny','sunny','rain'],\
                ['sunny','sunny','sunny'],['sunny','cloudy','sunny'],['cloudy','cloudy','rain']]
        
        sample_label = ['rain',  'rain',  'rain',\
                'sunny',  'sunny',  'rain',\
                'sunny',  'sunny', 'sunny']
                
        tree = DecisionTreeID3(sample_mat, sample_label)
        tree_structure = tree.build()
        self.tlog("Tree structure : " + str(tree_structure))
        
        r1 = autotest.eval_predict_one(tree, ['cloudy','cloudy','rain'] , 'sunny', self.logging)
        assert r1 == True 

        self.set_global_value('Stored_ID3_tree', tree)


class test_DecisionTreeID3_store(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        tree = self.get_global_value("Stored_ID3_tree")
        tmp_store_name = "tmp/tree_aba_store_test.dat"

        self.tlog("store tree to " + tmp_store_name)
        fs.store_module(tree,tmp_store_name)
        mod = fs.restore_module(tmp_store_name)
        
        self.tlog("restored tree : " + str(mod.tree))
        mod_r1 = autotest.eval_predict_one(mod, ['cloudy','cloudy','sunny'], 'sunny', self.logging)
        
        assert mod_r1 == True


class test_DecisionTreeID3_lense(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        lense_mat_train, lense_label_train, lense_mat_test, lense_label_test=\
          fs.csv_loader("sample_data/lense/lense.csv", 0.3)
        dtree_lense = DecisionTreeID3(lense_mat_train,lense_label_train)
        tree_structure = dtree_lense.build()
        self.tlog("Tree structure : " + str(tree_structure))
        error_rate = autotest.eval_predict(dtree_lense, lense_mat_test, lense_label_test, self.logging)
        self.tlog("lense predict (with decision tree) error rate : " +str(error_rate))
        
