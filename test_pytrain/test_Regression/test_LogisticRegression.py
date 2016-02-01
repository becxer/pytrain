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
                     [0.12, 3.25],\
                     [0.24, 4.33],\
                     [0.14, 2.45],\
                     [0.30, 4.23],\
                     ]
        train_label = [0,1,0,1]
        
        logis_reg = LogisticRegression(train_mat, train_label)
        logis_reg.fit(0.1, 300)
        
        batch.eval_predict_one(logis_reg,[0.10,2.33],0,self.logging)
        batch.eval_predict_one(logis_reg,[0.40,4.37],1,self.logging)
        
