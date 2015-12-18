#
# test template
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_Suite


class test_Template(test_Suite):

    def __init__(self, logging = True):
        test_Suite.__init__(self, logging)

    def test_process(self):
        global_value = self.get_global_value('some_key')
        self.set_global_value('another_key', 'new_value')
        
        self.tlog('logging somthing')
        
        assert 1 == 1 
