from log.Log import Log
#
#     'white':    "\033[1,37m",
#     'yellow':   "\033[1,33m",
#     'green':    "\033[1,32m",
#     'blue':     "\033[1,34m",
#     'cyan':     "\033[1,36m",
#     'red':      "\033[1,31m",
#     'magenta':  "\033[1,35m",
#     'black':    "\033[1,30m",
#     'darkwhite':  "\033[0,37m",
#     'darkyellow': "\033[0,33m",
#     'darkgreen':  "\033[0,32m",
#     'darkblue':   "\033[0,34m",
#     'darkcyan':   "\033[0,36m",
#     'darkred':    "\033[0,31m",
#     'darkmagenta':"\033[0,35m",
#     'darkblack':  "\033[0,30m",
#     'off':        "\033[0,0m"

COMMENT = '\033[90m'
HEADER = '\033[35m'
OKBLUE = '\033[34m'
OKGREEN = '\033[32m'
WARNING = '\033[33m'
FAIL = '\033[31m'
NORMAL = '\033[0m'

BOLD = '\033[1m'
UNDERLINE = '\033[4m'

class LogSingleton:
    log = Log()

    @staticmethod
    def get_singleton():
        return LogSingleton.log


def print(text, status=NORMAL):
    LogSingleton.get_singleton().print(status + text + NORMAL)