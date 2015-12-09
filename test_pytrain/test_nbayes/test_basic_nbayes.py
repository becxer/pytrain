#
# test basic nbayes
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite
from pytrain.nbayes import basic_nbayes

class test_basic_nbayes(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        sample_text_mat = [\
            ['hello','this','is','virus','mail'],\
            ['hi','this','is','from', 'friend'],\
            ['how','about','buy','this','virus'],\
            ['facebook','friend','contact','to','you'],\
            ['I','love','you','baby','virus'],\
            ['what','nice','day','how','about','you']\
        ]
        sample_text_label = ['spam','real','spam','real','spam','real']
        sample_mat_3 = [[]]
        sample_label_3 = []

        assert len(sample_label_3) == 6
        
        nbayes = basic_nbayes(sample_mat_3,sample_label_3)

        self.tlog("nbayes fit : " + str(nbayes.fit()))
        self.tlog("nbayes predict : " + str(nbayes.predict([])))
  
