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
from pytrain.lib import batch


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

        voca = nlp_eng.extract_vocabulary(sample_docs)
        docs_vector = []
        for doc in sample_docs:
            docs_vector.append(nlp_eng.set_of_words2vector(voca, doc))

        self.tlog(voca)
        self.tlog(docs_vector)

        assert len(voca) == 12

        nbayes = NaiveBayes(docs_vector, docs_label)
        nbayes.fit()

        trg = "this is virus mail"
        self.tlog(trg)
        trg_vec = nlp_eng.set_of_words2vector(voca, trg)
        self.tlog(trg_vec)
        
        result = nbayes.predict(trg_vec)
        self.tlog("NaiveBayes predict : " + str(result))

        assert result == 'spam'

        trg2 = "I love you love"
        self.tlog(trg2)
        trg2_vec = nlp_eng.bag_of_words2vector(voca, trg2)
        self.tlog(trg2_vec)

        result2 = nbayes.predict(trg2_vec)
        self.tlog("NaiveBayes predict : " + str(result2))

        assert result2 == 'spam'


class test_NaiveBayes_email(test_Suite):

    def __init__(self, logging =  True):
        test_Suite.__init__(self,logging)

    def test_process(self):

        nlp_eng = nlp("eng")

        email_data_file = "sample_data/email/email_word_real.txt"
        emailmat_train, emaillabel_train, voca, emailmat_test, emaillabel_test \
                = fs.f2wordmat(email_data_file, 0.3, nlp_eng)

        self.tlog(voca)
        
        email_nbayes = NaiveBayes(emailmat_train, emaillabel_train)
        email_nbayes.fit()

        error_rate = batch.eval_predict(email_nbayes, emailmat_test, emaillabel_test, False)

        self.tlog("spam-mail predict (with NaiveBayes) error rate : " +str(error_rate))

        assert error_rate <= 0.1


