#
# test NaiveBayes
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.NaiveBayes import NaiveBayes
from pytrain.lib import nlp


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

        voca = nlp.extract_vocabulary(sample_docs)
        docs_vector = []
        for doc in sample_docs:
            docs_vector.append(nlp.sentence2vector(voca, doc))

        self.tlog(voca)
        self.tlog(docs_vector)

        assert len(voca) == 22

        nbayes = NaiveBayes(docs_vector, docs_label)
        nbayes.fit()

        trg = "this is virus mail"
        trg_vec = nlp.sentence2vector(voca, trg)
        
        self.tlog(trg_vec)
        result = nbayes.predict(trg_vec)
        self.tlog("NaiveBayes predict : " + str(result))

        assert result == 'spam'
  
