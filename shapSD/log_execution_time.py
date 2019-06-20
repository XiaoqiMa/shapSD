import time
import functools
import logging


def init_logging(logfile):
    logger = logging.getLogger(logfile)
    if not len(logging.getLogger(logfile).handlers):
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                      datefmt='%m/%d/%Y %H:%M:%S')
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
    return logger

# def get_logger(logfile):
#     logger = logging.getLogger(logfile)
#     return logger


def func_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'logfile' in kwargs.keys():
            logfile = kwargs['logfile']
            logfile = '../logs/{}'.format(logfile)
        else:
            logfile = '../logs/execution_time.log'

        logger = init_logging(logfile)
        # print('handlers: ', logger.handlers)
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        msg = '{} running time: {} ms'.format(func.__name__, (end - start) * 1000)
        logger.info(msg)
        print('logging to file: ', logfile)
        return res

    return wrapper
