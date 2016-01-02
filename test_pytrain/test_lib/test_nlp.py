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
        sentence = "hello this is virus mail"
        text = "one sentence\ntwo sentence\nthree sentence"
        stopwords = ["this", "is", "a", "the"]

        words = nlp.split2words(sentence, stopwords)
        words_text = nlp.split2words(text, stopwords)
        split_sentence = nlp.split2sentence(text)

        self.tlog(words)
        self.tlog(split_sentence)

        assert words[3] == 'virus'
        assert len(words) == 5
        assert split_sentence[1] == "two sentence"
        assert len(split_sentence) == 3
        assert words_text[2] == "two"
        assert len(words_text) == 6

    def test_nlp_extract_vocabulary(self):
        docs = [\
            "Just try to enjoy it :).",\
            "It's very important for me!",\
            "What is your problem? you look so bad."\
        ]

        self.voca = nlp.extract_vocabulary(docs, nlp.ENG_STOPWORDS)
        self.tlog(self.voca)
        assert len(self.voca) == 18

    def test_words2vector(self):
        input_txt = "try to do this one"
        set_vector = nlp.set_of_words2vector(self.voca, input_txt, nlp.ENG_STOPWORDS)
        self.tlog(set_vector)
        
        input_txt2 = "It's your problem"
        bag_vector = nlp.bag_of_words2vector(self.voca, input_txt2, nlp.ENG_STOPWORDS)
        self.tlog(bag_vector)

    def test_process(self):
        self.test_nlp_split()
        self.test_nlp_extract_vocabulary()
        self.test_words2vector()
