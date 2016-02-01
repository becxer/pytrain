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

    def test_fs_f2mat(self):
        sample_data = "sample_data/dating/date_info.txt"
        self.tlog("loading matrix => " + sample_data)

        dmat_train, dlabel_train, dmat_test, dlabel_test \
            = fs.f2mat(sample_data, 0.1)
        assert len(dmat_train) == 900
        assert len(dlabel_train) == 900
        assert len(dmat_test) == 100
        assert len(dlabel_test) == 100

        self.set_global_value('dmat_train',dmat_train)
        self.set_global_value('dlabel_train',dlabel_train)
        self.set_global_value('dmat_test',dmat_test)
        self.set_global_value('dlabel_test',dlabel_test)


    def test_fs_f2wordmat(self):
        sample_words = "sample_data/email/email_word_small.txt"
        self.tlog("loading words => " + sample_words)


        nlp_eng = nlp("eng")
        wordmat_train, wordlabel_train, voca, wordmat_test, wordlabel_test \
                = fs.f2wordmat(sample_words, 0.1, nlp_eng)

        assert len(voca) == 7
        assert len(wordmat_train) == 4
        assert len(wordlabel_train) == 4


    def test_process(self):
        self.test_fs_f2mat()
        self.test_fs_f2wordmat()
        # To see test of storing module, check test_decision_tree

