#
# test template
#
# @ author becxer
# @ email becxer87@gmail.com
#
from test_pytrain import test_suite

class test_template(test_suite):

    def __init__(self, logging = True):
        test_suite.__init__(self, logging)

    def test_process(self):
        global_value = self.get_gvalue('some_key')
        self.set_gvalue('another_key', 'new_value')
        
        self.tlog('logging somthing')
        
        assert 1 == 1 
