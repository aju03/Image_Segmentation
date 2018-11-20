import logging
import sys
import time
import os

def init_logging():
    rootLogger = logging.getLogger('my_logger')

    LOG_DIR = os.getcwd() + '/' + 'logs'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    fileHandler = logging.FileHandler("{0}/{1}.log".format(LOG_DIR, str(int(time.time()))))
    rootLogger.addHandler(fileHandler)

    rootLogger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)

    return rootLogger
