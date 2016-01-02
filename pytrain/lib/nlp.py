# library for nlp
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#



def switch_split2words_str(arg):
    return arg.split()


def switch_split2words_list(arg):
    res = []
    for item in arg:
        res.append(item.split())
    return res


def split2words(arg):
    switch = {\
        'str':switch_split2words_str,\
        'list':switch_split2words_list\
    }
    return switch[str(type(arg).__name__)](arg)


def split2sentence(text):
    return text.split('\n')

def extract_vocabulary(documents):
    vocabulary = set([])
    for doc in documents:
        if str(type(doc).__name__) == 'str':
            doc = split2words(doc)
        vocabulary = vocabulary | set(doc)
    return list(vocabulary)

def set_of_words2vector(vocabulary, sentence):
    voca_vector = [0] * len(vocabulary)
    if str(type(sentence).__name__) == 'str':
        sentence = split2words(sentence)
    for word in sentence:
        if word in vocabulary:
            voca_vector[vocabulary.index(word)] = 1
    return voca_vector

def bag_of_words2vector(vocabulary, sentence):
    voca_vector = [0] * len(vocabulary)
    if str(type(sentence).__name__) == 'str':
        sentence = split2words(sentence)
    for word in sentence:
        if word in vocabulary:
            voca_vector[vocabulary.index(word)] += 1
    return voca_vector
