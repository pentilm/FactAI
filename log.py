import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')
import pathlib
import logging
import logging.handlers as loghandler

from datetime import datetime

import trace

loggers = {}
LOG_LEVEL = logging.DEBUG

def setup_custom_logger(name, log_level=LOG_LEVEL):
    if(not hasattr(logging,name.lower())):
        trace.addLoggingLevel(name, LOG_LEVEL - 5)
    if loggers.get(name.lower()):
        return loggers[name.lower()]

    logger = logging.getLogger(name.lower())
    loggers[name] = logger

    file =False # True if we want things to be printed to the file or files

    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - line %(lineno)d - %(message)s')
    path = pathlib.Path(__file__).parent.absolute()
    timestamp = datetime.utcnow().isoformat()
    if file:
        handler = loghandler.TimedRotatingFileHandler(                
                    filename="factai.log",
                    when='D', interval=1, backupCount=7,
                    encoding="utf-8")
    else:
        handler = logging.StreamHandler() 
    handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = True
    return logger

