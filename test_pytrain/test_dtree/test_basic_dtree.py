#
# test template
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite
from pytrain.dtree import basic_dtree
from pytrain.lib import fs
from pytrain.lib import batch

class test_basic_dtree(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        sample_mat = [[7,8,8],[8,7,8],[8,8,8],[8,8,8],[8,7,7],[7,7,8],[7,7,7],[7,8,7],[8,8,8]]
        sample_label = ['yes',  'yes',  'yes', 'no',  'no',  'yes',   'no',   'no', 'no']
        
        tree = basic_dtree(sample_mat, sample_label)
        self.tlog("tree fit : " + str(tree.fit()))
        
        f1 = tree.predict([8,8,8])
        self.tlog("tree predict : " + str(f1))

        assert f1 == 'no' 

        self.set_gvalue('basic_dtree', tree)


class test_basic_dtree_store(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        tree = self.get_gvalue("basic_dtree")
        tmp_store_name = "tmp/tree_878_store_test.dat"

        self.tlog("store tree to " + tmp_store_name)
        fs.store_module(tree,tmp_store_name)
        mod = fs.restore_module(tmp_store_name)
        
        self.tlog("restored tree : " + str(mod.tree))
        mod_res = mod.predict([8,8,7])
        self.tlog("restored tree predict : " + str(mod_res))
        
        assert mod_res == 'no'


class test_basic_dtree_lense(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        lense_mat_train, lense_label_train, lense_mat_test, lense_label_test=\
                                    fs.f2mat("test_data/lense/lense.txt", 0.4)
        dtree_lense = basic_dtree(lense_mat_train,lense_label_train)
        dtree_lense.fit()
        error_rate = batch.eval_predict(dtree_lense, lense_mat_test, lense_label_test, False)
        self.tlog("lense predict (with dtree) error rate : " +str(error_rate))
        
        assert error_rate == 0.2
        
