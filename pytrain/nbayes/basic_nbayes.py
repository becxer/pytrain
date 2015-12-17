from numpy import *

class basic_nbayes:

    def __init__(self, mat_data, label_data):
        self.word_data = mat_data
        self.num_word = 0
        
        self.cate_data = label_data
        self.cate_set = {}
        self.num_cate = 0
        
        self.cate_word = {}
        self.cate_word_sum = {}

    def fit(self):
        self.num_word = len(self.word_data[0])
        for i, cate in enumerate(self.cate_data):
            self.cate_word[cate] = self.cate_word.get(cate, \
                    zeros(self.num_word)) + self.word_data[i]
            self.cate_set[cate] = self.cate_set.get(cate, 0) + 1
        for cate in self.cate_word:
            self.cate_word_sum[cate] = self.cate_word[cate].sum(axis=0)
        self.num_cate = len(self.cate_set)

    def predict(self, array_input):

        pass

