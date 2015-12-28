#
# test lib.nlp
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.lib import nlp

class test_nlp(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        sentence = "hello this is virus mail"
        text = "one sentence\ntwo sentence\nthree sentence"

        words = nlp.split2words(sentence)
        words_text = nlp.split2words(text)
        split_sentence = nlp.split2sentence(text)

        self.tlog(words)
        self.tlog(split_sentence)

        assert words[3] == 'virus'
        assert len(words) == 5
        assert split_sentence[1] == "two sentence"
        assert len(split_sentence) == 3
        assert words_text[2] == "two"
        assert len(words_text) == 6

        docs = [\
            "a b c d e f g h i j",\
            "a1 b1 c1 a b abcd a b c a2",\
            "1 2 3 4 5 6 a b c d e f g"\
        ]

        voca = nlp.extract_vocabulary(docs)
        self.tlog(voca)
        assert len(voca) == 21

        # testing set of words2vector

        input_txt = "a1 g e f aa"
        set_vector = nlp.set_of_words2vector(voca, input_txt)
        self.tlog(set_vector)
        assert set_vector[1] == 1

        # testing bag of words2vector

        input_txt2 = "a1 g e f aa a1"
        bag_vector = nlp.bag_of_words2vector(voca, input_txt2)
        self.tlog(bag_vector)
        assert bag_vector[1] == 2


