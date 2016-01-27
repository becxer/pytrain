#
# test Decision Tree
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.DecisionTree import DecisionTree
from pytrain.lib import fs
from pytrain.lib import batch


class test_DecisionTree(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        
        sample_mat = [['a','b','b'],['b','a','b'],['b','b','b'],\
                ['b','b','b'],['b','a','a'],['a','a','b'],\
                ['a','a','a'],['a','b','a'],['b','b','b']]
        
        sample_label = ['yes',  'yes',  'yes',\
                'no',  'no',  'yes',\
                'no',   'no', 'no']
        
        tree = DecisionTree(sample_mat, sample_label)
        self.tlog("tree fit : " + str(tree.fit()))
        
        r1 = batch.eval_predict_one(tree, ['b','b','b'], 'no', self.logging)
        assert r1 == True 

        self.set_global_value('DecisionTree', tree)


class test_DecisionTree_store(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        tree = self.get_global_value("DecisionTree")
        tmp_store_name = "tmp/tree_aba_store_test.dat"

        self.tlog("store tree to " + tmp_store_name)
        fs.store_module(tree,tmp_store_name)
        mod = fs.restore_module(tmp_store_name)
        
        self.tlog("restored tree : " + str(mod.tree))
        mod_r1 = batch.eval_predict_one(mod, ['b','b','a'], 'no', self.logging)
        
        assert mod_r1 == True


class test_DecisionTree_lense(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        lense_mat_train, lense_label_train, lense_mat_test, lense_label_test=\
                                    fs.f2mat("sample_data/lense/lense.txt", 0.3)
        dtree_lense = DecisionTree(lense_mat_train,lense_label_train)
        dtree_lense.fit()
        error_rate = batch.eval_predict(dtree_lense, lense_mat_test, lense_label_test, self.logging)
        self.tlog("lense predict (with decision tree) error rate : " +str(error_rate))
        
