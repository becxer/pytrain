#
# test Gaussian Naive Bayes
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.NaiveBayes import GaussianNaiveBayes
from pytrain.lib import nlp
from pytrain.lib import fs
from pytrain.lib import batch


class test_GaussianNaiveBayes(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        pass
