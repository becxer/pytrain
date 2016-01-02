# library for nlp
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#
import re

ENG_STOPWORDS = ["a","the","this"]

def switch_split2words_str(arg, stopwords):
    res = re.compile("([\w][\w]*'?\w?)").findall(arg)
    return res

def switch_split2words_list(arg, stopwords):
    res = []
    for item in arg:
        if str(type(item).__name__) == 'str':
            res.append(split2words(item, stopwords))
    return res

def split2words(arg, stopwords):
    switch = {\
        'str':switch_split2words_str,\
        'list':switch_split2words_list\
    }
    return switch[str(type(arg).__name__)](arg, stopwords)

def split2sentence(text):
    # Need to improve sentence split algorithm
    return text.split('\n')

def extract_vocabulary(documents, stopwords):
    vocabulary = set([])
    for doc in documents:
        if str(type(doc).__name__) == 'str':
            doc = split2words(doc,stopwords)
        vocabulary = vocabulary | set(doc)
    return list(vocabulary)

def set_of_words2vector(vocabulary, sentence, stopwords):
    voca_vector = [0] * len(vocabulary)
    if str(type(sentence).__name__) == 'str':
        sentence = split2words(sentence, stopwords)
    for word in sentence:
        if word in vocabulary:
            voca_vector[vocabulary.index(word)] = 1
    return voca_vector

def bag_of_words2vector(vocabulary, sentence, stopwords):
    voca_vector = [0] * len(vocabulary)
    if str(type(sentence).__name__) == 'str':
        sentence = split2words(sentence, stopwords)
    for word in sentence:
        if word in vocabulary:
            voca_vector[vocabulary.index(word)] += 1
    return voca_vector
