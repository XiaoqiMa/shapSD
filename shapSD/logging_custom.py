import time
import functools
import logging


def init_logging(logfile, level=logging.INFO):
    logger = logging.getLogger(logfile)
    if not len(logging.getLogger(logfile).handlers):
        log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                          datefmt='%m/%d/%Y %H:%M:%S')
        fh = logging.FileHandler(logfile)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
        logger.setLevel(level)
    return logger

# def get_logger(logfile):
#     logger = logging.getLogger(logfile)
#     return logger


def execution_time_logging(func):
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
        running_time_ms = (end - start) * 1000
        rest, t_ms = divmod(running_time_ms, 1000)
        t_minute, t_s = divmod(rest, 60)
        msg = '{} running time: {}M:{}s:{}ms'.format(func.__name__, int(t_minute), int(t_s), int(t_ms))
        logger.info(msg)
        # print('logging to file: ', logfile)
        return res

    return wrapper

def err_logging(msg):
    level = logging.ERROR
    logfile = '../logs/err_msg.log'
    logger = init_logging(logfile, level)
    logger.error(msg)