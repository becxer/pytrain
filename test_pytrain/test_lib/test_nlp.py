#
# test lib.nlp
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.lib import nlp

class test_nlp(test_Suite):

    voca = []

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_nlp_split(self):

        nlp_eng = nlp("eng")

        sentence = "hello this is virus mail"
        text = "one sentence\ntwo sentence\nthree sentence"

        words = nlp_eng.split2words(sentence)
        words_text = nlp_eng.split2words(text)
        split_sentence = nlp_eng.split2sentence(text)

        self.tlog(words)
        self.tlog(split_sentence)

        assert words[2] == 'mail'
        assert len(words) == 3
        assert split_sentence[1] == "two sentence"
        assert len(split_sentence) == 3
        assert words_text[2] == "two"
        assert len(words_text) == 6

    def test_nlp_extract_vocabulary(self):

        nlp_eng = nlp("eng")
        docs = [\
            "Just try to enjoy it :).",\
            "It's very important for me!",\
            "What is your problem? you look so bad."\
        ]
        self.voca = nlp_eng.extract_vocabulary(docs)
        self.tlog(self.voca)
        assert len(self.voca) == 7

    def test_word2vector(self):
        nlp_eng = nlp("eng")

        input_txt = "try to do this one"
        set_vector = nlp_eng.set_of_word2vector(self.voca, input_txt)
        self.tlog(set_vector)
        
        input_txt2 = "It's your problem. big problem. let's try"
        bag_vector = nlp_eng.bag_of_word2vector(self.voca, input_txt2)
        self.tlog(bag_vector)

    def test_wordseq2matrix(self):
        word_list_array = [\
            ['I','a','m','a','b','o','y'],\
            ['Y','o','u','a','r','e','a','g','i','r','l'],\
            ['I','a','m','a','g','o','o','d','b','o','y'],\
            ['Y','o','u','a','r','e','a','g','o','o','d','g','i','r','l'],\
        ]
        nlp_common = nlp()
        voca = nlp_common.extract_vocabulary(word_list_array)
        word_mat_array = []
        for word_list in word_list_array :
            word_mat = nlp_common.set_of_wordseq2matrix(voca, word_list)
            word_mat_array.append(word_mat)
        self.tlog(word_mat_array)

    def test_process(self):
        self.test_nlp_split()
        self.test_nlp_extract_vocabulary()
        self.test_word2vector()
        self.test_wordseq2matrix()
