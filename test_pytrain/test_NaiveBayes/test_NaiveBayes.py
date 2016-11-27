#
# test Naive Bayes
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.NaiveBayes import NaiveBayes
from pytrain.lib import nlp
from pytrain.lib import fs
from pytrain.lib import autotest

class test_NaiveBayes(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        sample_docs = [\
                "hello this is virus mail",\
                "hi this is from friend",\
                "how about buy this virus",\
                "facebook friend contact to you",\
                "I love you baby virus",\
                "what a nice day how about you"\
            ]

        docs_label =\
                ['spam','real','spam','real','spam','real']

        nlp_eng = nlp("eng")

        # extract vocabulary from docs
        voca = nlp_eng.extract_vocabulary(sample_docs)
        self.tlog(voca)
        assert len(voca) == 12
       
        # convert docs to bag of word vector using vocabulary
        docs_vector = []
        for doc in sample_docs:
            docs_vector.append(nlp_eng.bag_of_word2vector(voca, doc))
        self.tlog(docs_vector)

        # training NaiveBayes
        nbayes = NaiveBayes(docs_vector, docs_label)
        nbayes.fit()

        # test case 1
        tc1 = "this is virus mail"
        tc1_vec = nlp_eng.bag_of_word2vector(voca, tc1)
        
        self.tlog(tc1)
        self.tlog(tc1_vec)
        
        r1 = autotest.eval_predict_one(nbayes,tc1_vec,'spam',self.logging)
        assert r1 == True

        # test case 2
        tc2 = "I love you love"
        tc2_vec = nlp_eng.bag_of_word2vector(voca, tc2)
        
        self.tlog(tc2)
        self.tlog(tc2_vec)

        r2 = autotest.eval_predict_one(nbayes,tc2_vec,'spam',self.logging)
        assert r2 == True


class test_NaiveBayes_email(test_Suite):

    def __init__(self, logging =  True):
        test_Suite.__init__(self,logging)

    def test_process(self):

        nlp_eng = nlp("eng")

        email_data_file = "sample_data/email/email.tsv"
        emailmat_train, emaillabel_train, voca, emailmat_test, emaillabel_test \
                = fs.tsv_loader_with_nlp(email_data_file, 0.3, nlp_eng)
        self.tlog(voca)
        
        email_nbayes = NaiveBayes(emailmat_train, emaillabel_train)
        email_nbayes.fit()

        error_rate = autotest.eval_predict(email_nbayes, emailmat_test, emaillabel_test, self.logging)
        self.tlog("spam-mail predict (with NaiveBayes) error rate : " +str(error_rate))

        assert error_rate <= 0.1


