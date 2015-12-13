# library for nlp
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#

def split_word(sentence):
    return sentence.split()

def split_sentence(text):
    return text.split('\n')

def extract_vocabulary(documents):
    vocabulary = set([])
    for doc in documents:
        vocabulary = vocabulary | set(split_word(doc))
    return list(vocabulary)

def sentence2vector(vocabulary, sentence):
    voca_vector = [0] * len(vocabulary)
    for word in split_word(sentence):
        if word in vocabulary:
            voca_vector[vocabulary.index(word)] = 1
    return voca_vector
