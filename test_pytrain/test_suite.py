#
# Dev test for pytrain library
#
# @ author becxer
# @ email becxer87@gmail.com
#

class test_suite:

    def __init__(self):
        pass

    def test_process(self):
        pass

    def process(self):
        print "__________________________________________"
        print "TEST MODULE NAME : " + self.__class__.__name__
        print "now testing... "
        self.test_process()
        print "[Success] " + self.__class__.__name__
