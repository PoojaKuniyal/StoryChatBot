import logging
import os
from datetime import datetime

LOGS_DIR ='logs'
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")
# configure login
logging.basicConfig(
    filename= LOG_FILE,
    format= '%(asctime)s - %(levelname)s - %(message)s', # time-info-msg. levelname - info, warning, error
   level = logging.INFO, # means only info, error, warning msgs will be shown no other msgs will be shown
) 

# logger function to initalize logger in different files
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
