#
# test HMM
#
# @ author becxer
# @ e-mail becxer87@gmail.com
# 

from test_pytrain import test_Suite
from pytrain.HMM import HMM
from pytrain.lib import autotest
from pytrain.lib import nlp
from numpy import *

class test_HMM(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):

        train_mat = [\
                     # Sequence of characters with no space
                     ['<s>','I','a','m','a','b','o','y'],\
                     ['<s>','Y','o','u','a','r','e','a','g','i','r','l'],\
                     ['<s>','I','a','m','a','g','o','o','d','b','o','y'],\
                     ['<s>','Y','o','u','a','r','e','a','g','o','o','d','g','i','r','l'],\
                     ]

        train_label = [\
                     # Sequence of label tagged to space
                     # [1 == space, 0 == no-space]
                     [0,1,0,1,1,0,0,1],\
                     [0,0,0,1,0,0,1,1,0,0,0,1],\
                     [0,1,0,1,1,0,0,0,1,0,0,1],\
                     [0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1],\
                    ]

        nlp_common = nlp()
        voca = nlp_common.extract_vocabulary(train_mat)
        train_wordseq_mat = []
        for wordseq in train_mat:
            wordseq_mat = nlp_common.set_of_wordseq2matrix(voca, wordseq)
            train_wordseq_mat.append(wordseq_mat)

        hmm = HMM(train_wordseq_mat, train_label, hidden_state_labeled = True, hidden_state = 2)
        hmm.fit(toler = 0.001, epoch= 30)
        
        ti1 = nlp_common.set_of_wordseq2matrix(voca,['<s>','I','a','m','g','o','o','d'])
        r1 = autotest.eval_predict_one(hmm, ti1, [0,1,0,1,0,0,0,1], self.logging)

        ti2 = nlp_common.set_of_wordseq2matrix(voca,['<s>','Y','o','u','a','r','e','a','b','o','y'])
        r2 = autotest.eval_predict_one(hmm, ti2, [0,0,0,1,0,0,1,1,0,0,1], self.logging)

        ti3 = nlp_common.set_of_wordseq2matrix(voca,['<s>','Y','o','u','a','r','e','g','i','r','l'])
        r3 = autotest.eval_predict_one(hmm, ti3, [0,0,0,1,0,0,1,0,0,0,1], self.logging)

        ti4 = nlp_common.set_of_wordseq2matrix(voca,['<s>','I','a','m','g','i','r','l'])
        r4 = autotest.eval_predict_one(hmm, ti4, [0,1,0,1,0,0,0,1], self.logging)

class test_HMM_BaumWelch(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):

        train_mat = [\
                     # Sequence of characters with no space
                     ['<s>','I','a','m','a','b','o','y'],\
                     ['<s>','Y','o','u','a','r','e','a','g','i','r','l'],\
                     ['<s>','I','a','m','a','g','o','o','d','b','o','y'],\
                     ['<s>','Y','o','u','a','r','e','a','g','o','o','d','g','i','r','l'],\
                     ]
                     
        nlp_common = nlp()
        voca = nlp_common.extract_vocabulary(train_mat)
        train_wordseq_mat = []
        for wordseq in train_mat:
            wordseq_mat = nlp_common.set_of_wordseq2matrix(voca, wordseq)
            train_wordseq_mat.append(wordseq_mat)

        hmm = HMM(train_wordseq_mat, label_data =  None, hidden_state_labeled = False, hidden_state = 2)
        hmm.fit(toler = 0.001, epoch = 30)
        
        ti1 = nlp_common.set_of_wordseq2matrix(voca,['<s>','I','a','m','g','o','o','d'])
        r1 = autotest.eval_predict_one(hmm, ti1, [0,1,0,1,0,0,0,1], self.logging)
        
        ti2 = nlp_common.set_of_wordseq2matrix(voca,['<s>','Y','o','u','a','r','e','a','b','o','y'])
        r2 = autotest.eval_predict_one(hmm, ti2, [0,0,0,1,0,0,1,1,0,0,1], self.logging)

        ti3 = nlp_common.set_of_wordseq2matrix(voca,['<s>','Y','o','u','a','r','e','g','i','r','l'])
        r3 = autotest.eval_predict_one(hmm, ti3, [0,0,0,1,0,0,1,0,0,0,1], self.logging)

        ti4 = nlp_common.set_of_wordseq2matrix(voca,['<s>','I','a','m','g','i','r','l'])
        r4 = autotest.eval_predict_one(hmm, ti4, [0,1,0,1,0,0,0,1], self.logging)
