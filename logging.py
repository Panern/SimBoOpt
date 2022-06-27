import sys
from datetime import datetime
Run_day = datetime.today().date()



'''
    writting log for observation or reproduct
'''
class Logger(object) :
    def __init__(self, filename='./Logging/main/Run_recording_{}'.format(Run_day), stream=sys.stdout) :
        self.terminal = stream
        self.log = open(filename +'.log', 'a+')
        self.run = 1

    def write(self, message) :
        message = message
        self.terminal.write(message)
        self.log.write(message)


    def flush(self) :
        pass