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
        sample_mat = [[7,8,8],[8,7,8],[8,8,8],[8,8,8],[8,7,7],[7,7,8],[7,7,7],[7,8,7],[8,8,8]]
        sample_label = ['yes',  'yes',  'yes', 'no',  'no',  'yes',   'no',   'no', 'no']
        
        tree = DecisionTree(sample_mat, sample_label)
        self.tlog("tree fit : " + str(tree.fit()))
        
        f1 = tree.predict([8,8,8])
        self.tlog("tree predict : " + str(f1))

        assert f1 == 'no' 

        self.set_global_value('DecisionTree', tree)


class test_DecisionTree_store(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        tree = self.get_global_value("DecisionTree")
        tmp_store_name = "tmp/tree_878_store_test.dat"

        self.tlog("store tree to " + tmp_store_name)
        fs.store_module(tree,tmp_store_name)
        mod = fs.restore_module(tmp_store_name)
        
        self.tlog("restored tree : " + str(mod.tree))
        mod_res = mod.predict([8,8,7])
        self.tlog("restored tree predict : " + str(mod_res))
        
        assert mod_res == 'no'


class test_DecisionTree_lense(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        lense_mat_train, lense_label_train, lense_mat_test, lense_label_test=\
                                    fs.f2mat("sample_data/lense/lense.txt", 0.3)
        dtree_lense = DecisionTree(lense_mat_train,lense_label_train)
        dtree_lense.fit()
        error_rate = batch.eval_predict(dtree_lense, lense_mat_test, lense_label_test, False)
        self.tlog("lense predict (with decision tree) error rate : " +str(error_rate))
        
        assert error_rate <= 0.3
        
