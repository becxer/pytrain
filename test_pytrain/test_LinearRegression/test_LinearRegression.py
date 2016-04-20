#
# test Linear Regression
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite
from pytrain.LinearRegression import LinearRegression
from pytrain.lib import fs
from pytrain.lib import batch

class test_LinearRegression(test_Suite):

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
        
        linear_reg =\
            LinearRegression(train_mat, train_label)
        linear_reg.fit(lr = 0.001, epoch = 1000, stoc = 4)
        
        r1 = batch.eval_predict_one(linear_reg,[0.10,0.33],[0, 1],self.logging)
        r2 = batch.eval_predict_one(linear_reg,[4.40,4.37],[1, 0],self.logging)
        
class test_LinearRegression_horse(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        horse_mat_train, horse_label_train = fs.f2mat("sample_data/horse/horseColicTraining_1.txt",0)
        horse_label_train = [[float(x)] for x in horse_label_train]
        horse_mat_test, horse_label_test = fs.f2mat("sample_data/horse/horseColicTest_1.txt",0)
        horse_label_test = [[float(x)] for x in horse_label_test]
        linear_reg =\
            LinearRegression(horse_mat_train, horse_label_train)
        linear_reg.fit(lr = 0.0000001, epoch = 1000, stoc = 400)
        error_rate = batch.eval_predict(linear_reg,horse_mat_test,horse_label_test,self.logging)
        self.tlog("horse predict (with linear regression) error rate :" + str(error_rate))
        
