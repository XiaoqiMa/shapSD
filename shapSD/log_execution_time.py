import os
import time
import functools
import logging


def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    fh = logging.FileHandler(logfile)
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)


def get_logger(logfile):
    logger = logging.getLogger(logfile)
    return logger


def func_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            logfile = args[0]
        else:
            logfile = 'execution_time'
        if os.path.exists(logfile):
            pass
            # print('logging file exists')
        else:
            init_logging(logfile)
        logger = get_logger(logfile)

        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        msg = '{} running time: {} ms'.format(func.__name__, (end - start) * 1000)
        print(logger.__class__)
        logger.info(msg)
        return res

    return wrapper
