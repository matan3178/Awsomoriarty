# http://misc.flogisoft.com/bash/tip_colors_and_formatting - colors
from code.log.Log import Log

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


def print(msg, status=NORMAL):
    LogSingleton.get_singleton().print(status + str(msg) + NORMAL)


def blank_line():
    print("")

