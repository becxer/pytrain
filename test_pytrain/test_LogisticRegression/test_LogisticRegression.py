#
# test Logistic Regression
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.LogisticRegression import LogisticRegression
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
        train_label = [[0,1],[1,0],[0,1],[1,0]] # out bit is 1
        
        logistic_reg =\
            LogisticRegression(train_mat, train_label)
        logistic_reg.fit(lr = 0.001, epoch = 2000, stoc = 4)
        
        r1 = batch.eval_predict_one(logistic_reg,[0.10,0.33],[0, 1],self.logging)
        r2 = batch.eval_predict_one(logistic_reg,[4.40,4.37],[1, 0],self.logging)
        
class test_LogisticRegression_horse(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        horse_mat_train, horse_label_train = fs.f2mat("sample_data/horse/horseColicTraining_1.txt",0)
        horse_label_train = [[float(x)] for x in horse_label_train]
        horse_mat_test, horse_label_test = fs.f2mat("sample_data/horse/horseColicTest_1.txt",0)
        horse_label_test = [[float(x)] for x in horse_label_test]
        logistic_reg =\
            LogisticRegression(horse_mat_train, horse_label_train)
        logistic_reg.fit(lr = 0.0001, epoch = 1000, stoc = 400)
        error_rate = batch.eval_predict(logistic_reg,horse_mat_test,horse_label_test,self.logging)
        self.tlog("horse predict (with logistic regression) error rate :" + str(error_rate))
        
