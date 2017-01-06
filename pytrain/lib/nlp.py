# library for nlp
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#
import re

import os
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

class nlp:

    lang = "NOLANG"
    stopwords = []
    lower = False

    def set_with_eng(self, lower = False):
        global path
        self.lower = lower
        sw_f = open(path+"/nlp_stopwords.eng")
        self.stopwords = map(lambda x : x.strip(), sw_f.readlines())

    def set_with_kor(self):
        pass

    def __init__(self, lang_ = "NOLANG"):
        self.lang = lang_
        if self.lang == "eng":
            self.set_with_eng()
        if self.lang == "eng_lower":
            self.set_with_eng(lower = True)
        elif self.lang == "kor":
            self.set_with_kor()

    def switch_split2words_str(self, arg):
        splitwords = re.compile("([\w][\w]*'?\w?)").findall(arg)
        splitlowerwords = [ x.lower() for x in splitwords ]
        res = []
        for spword in splitlowerwords:
            if spword not in self.stopwords:
                res.append(spword)
        return res

    def switch_split2words_list(self, arg):
        res = []
        for item in arg:
            if str(type(item).__name__) == 'str':
                res.append(self.split2words(item))
        return res

    def split2words(self, arg):
        switch = {\
            'str':self.switch_split2words_str,\
            'list':self.switch_split2words_list\
        }
        return switch[str(type(arg).__name__)](arg)

    def split2sentence(self, text):
        # Need to improve sentence split algorithm
        return text.split('\n')

    def extract_vocabulary(self, documents):
        vocabulary = set([])
        for doc in documents:
            if str(type(doc).__name__) == 'str':
                doc = self.split2words(doc)
            ndoc = []
            for w in doc:
                if w not in self.stopwords:
                    if self.lower : w = w.lower()
                    ndoc.append(w)
            vocabulary = vocabulary | set(ndoc)
        return list(vocabulary)

    def set_of_word2vector(self, vocabulary, sentence):
        voca_vector = [0] * len(vocabulary)
        if str(type(sentence).__name__) == 'str':
            sentence = self.split2words(sentence)
        for word in sentence:
            if self.lower : word = word.lower()
            if word in vocabulary:
                voca_vector[vocabulary.index(word)] = 1
        return voca_vector

    def bag_of_word2vector(self, vocabulary, sentence):
        voca_vector = [0] * len(vocabulary)
        if str(type(sentence).__name__) == 'str':
            sentence = self.split2words(sentence)
        for word in sentence:
            if self.lower : word = word.lower()
            if word in vocabulary:
                voca_vector[vocabulary.index(word)] += 1
        return voca_vector

    def set_of_wordseq2matrix(self, vocabulary, wordlist):
        word_mat = []
        for word in wordlist:
            word_vector = [0] * len(vocabulary)
            if self.lower : word = word.lower()
            if word in vocabulary:
                word_vector[vocabulary.index(word)] = 1
            word_mat.append(word_vector)
        return word_mat

