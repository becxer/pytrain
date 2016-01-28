#
# test Gaussian Naive Bayes
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.Bayes import GaussianNaiveBayes
from pytrain.lib import nlp
from pytrain.lib import fs
from pytrain.lib import batch

class test_GaussianNaiveBayes(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        train_mat = [\
                [-65,-55,-42],[-20,-59,-71],[-43,-49,-69],\
                [-61,-30,-74],[-79,-81,-40],[-71,-57,-24],\
                [-67,-19,-58],[-57,-73,-83],[-68,-74,-59]
            ]
        train_label = ['B','A','A','A','B','B','A','C','C']
        
        test_mat = [\
                [-45,-47,-74],[-77,-69,-25],[-64,-71,-59]
            ]
        test_label = ['A','B','C']

        gnb = GaussianNaiveBayes(train_mat,train_label)
        gnb.fit()
        error_rate = batch.eval_predict(gnb, test_mat, test_label, self.logging)
        self.tlog("strength of signal predict error rate : " + str(error_rate))


