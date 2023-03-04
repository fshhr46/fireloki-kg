
import logging
import sys


def get_default_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(log_level)
    return logger