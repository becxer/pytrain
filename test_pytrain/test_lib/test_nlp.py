#
# test lib.nlp
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite
from pytrain.lib import nlp

class test_nlp(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        sentence = "hello this is virus mail"
        text = "one sentence\ntwo sentence\nthree sentence"

        words = nlp.split2word(sentence)
        words_text = nlp.split2word(text)
        splitted_sentence = nlp.split2sentence(text)

        self.tlog(words)
        self.tlog(splitted_sentence)

        assert words[3] == 'virus'
        assert len(words) == 5
        assert splitted_sentence[1] == "two sentence"
        assert len(splitted_sentence) == 3
        assert words_text[2] == "two"
        assert len(words_text) == 6

        docs = [\
            "a b c d e f g e f g",\
            "e1 23 f asdf 1a dd adsf f g d aa",\
            "1 2 3 4 5 6 a b c d e f g"\
        ]

        voca = nlp.extract_vocabulary(docs)
        self.tlog(voca)
        assert len(voca) == 20

        input_txt = "a 1 g e f aa"
        voca_vector = nlp.sentence2vector(voca, input_txt)
        self.tlog(voca_vector)
        assert voca_vector[0] == 1

