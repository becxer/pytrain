#
# test Logistic Regression
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.Regression import LogisticRegression
from pytrain.lib import fs
from pytrain.lib import batch

class test_LogisticRegression(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        train_mat = [\
                     [0.12, 0.25],\
                     [3.24, 4.33],\
                     [0.14, 0.45],\
                     [7.30, 4.23],\
                     ]
        train_label = [[0],[1],[0],[1]] # out bit is 1
        
        logis_reg =\
            LogisticRegression(train_mat, train_label)
        logis_reg.fit(lr = 0.01, epoch = 1000)
        
        r1 = batch.eval_predict_one(logis_reg,[0.10,0.33],[0.0],self.logging)
        r2 = batch.eval_predict_one(logis_reg,[4.40,4.37],[1.0],self.logging)

        assert r1 == True
        assert r2 == True
        
