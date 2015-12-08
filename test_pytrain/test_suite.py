#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com
#
import traceback
import sys

class test_suite:

    gvalue_set = {}

    def __init__(self, logging = True):
        self.logging = logging
        pass

    def tlog(self, log_str) : 
        if self.logging : 
            print "*Log from '" + self.__class__.__name__ +\
                    "' Module : \n",log_str

    def set_gvalue(self, key, value):
        self.__class__.gvalue_set[key] = value

    def get_gvalue(self, key):
        return self.__class__.gvalue_set[key]

    def test_process(self):
        pass

    def process(self):
        tag = "Module '" + self.__class__.__name__ + "' "
        print "-"
        try:
            print tag + "is now testing ..."
            self.test_process()
        except:
            traceback.print_exception(*sys.exc_info())
            print tag + "result ... Error"
        else:
            print tag + "result ... Ok"
