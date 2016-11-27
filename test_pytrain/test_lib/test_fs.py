#
# test lib.fs
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.lib import fs
from pytrain.lib import nlp

class test_fs(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_fs_csv_loader(self):
        sample_data = "sample_data/iris/iris.csv"
        self.tlog("loading matrix => " + sample_data)

        dmat_train, dlabel_train, dmat_test, dlabel_test \
            = fs.csv_loader(sample_data, 0.2)

        self.tlog('iris train data size : ' + str(len(dmat_train)))
        self.tlog('iris test data size : ' + str(len(dmat_test)))

        self.set_global_value('iris_mat_train', dmat_train)
        self.set_global_value('iris_label_train', dlabel_train)
        self.set_global_value('iris_mat_test', dmat_test)
        self.set_global_value('iris_label_test', dlabel_test)

    def test_fs_tsv_loader_with_nlp(self):
        sample_words = "sample_data/email/email.tsv"
        self.tlog("loading words => " + sample_words)

        nlp_eng = nlp("eng")
        wordmat_train, wordlabel_train, voca, wordmat_test, wordlabel_test \
          = fs.tsv_loader_with_nlp(sample_words, 0.1, nlp_eng)

        self.tlog('email data voca size : ' + str(len(voca)))
        self.tlog('voca sample : ' + str(voca[:5]))

    def test_process(self):
        self.test_fs_csv_loader()
        self.test_fs_tsv_loader_with_nlp()
        # To see test of storing module, check test_decision_tree

