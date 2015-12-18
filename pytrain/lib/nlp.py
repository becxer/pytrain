# library for nlp
#
# @ author becxer
# @ e-mail becxer87@gmail.com
#


def do_split2word_str(arg):
    return arg.split()


def do_split2word_list(arg):
    res = []
    for item in arg:
        res.append(item.split())
    return res


def split2word(arg):
    switch = {\
        'str':do_split2word_str,\
        'list':do_split2word_list\
    }
    return switch[str(type(arg).__name__)](arg)


def split2sentence(text):
    return text.split('\n')


def extract_vocabulary(documents):
    vocabulary = set([])
    for doc in documents:
        vocabulary = vocabulary | set(split2word(doc))
    return list(vocabulary)


def sentence2vector(vocabulary, sentence):
    voca_vector = [0] * len(vocabulary)
    for word in split2word(sentence):
        if word in vocabulary:
            voca_vector[vocabulary.index(word)] = 1
    return voca_vector
