import logging
import sys

APP_LOGGER_NAME = 'app'

def setup_logging(logger_name=APP_LOGGER_NAME, file_name=None):
    """
    Setup logging for the application.

    Args:
        logger_name (str): The name of the logger. Defaults to APP_LOGGER_NAME.
        file_name (str): The name of the log file. Defaults to None.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_logger(module_name, logger_name=None):
    """
    Get a logger for a module.
    """
    return logging.getLogger(logger_name or APP_LOGGER_NAME).getChild(module_name)
