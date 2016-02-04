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
        
class test_LogisticRegression_horse(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        horse_mat_train, horse_label_train = fs.f2mat("sample_data/horse/horseColicTraining_1.txt",0)
        horse_label_train = [[float(x)] for x in horse_label_train]
        horse_mat_test, horse_label_test = fs.f2mat("sample_data/horse/horseColicTest_1.txt",0)
        horse_label_test = [[float(x)] for x in horse_label_test]
        logis_reg =\
            LogisticRegression(horse_mat_train, horse_label_train)
        logis_reg.fit(lr = 0.004, epoch = 100)
        error_rate = batch.eval_predict(logis_reg,horse_mat_test,horse_label_test,self.logging)
        self.tlog("horse predict (with logistic regression) error rate :" + str(error_rate))
        assert False
        
