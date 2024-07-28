import logging
from datetime import datetime, timedelta

from .common import FileUtils


def beijing_time():
    _time = datetime.now() + timedelta(hours=8)
    return _time.timetuple()


logging.Formatter.converter = beijing_time


def create_logger(save_path, mode='Train'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    current_time = (datetime.now() + timedelta(hours=8)).strftime("%Y%m%d%H")
    log_file = f'{save_path}/logs/{mode}_log_{current_time}.log'
    try:
        FileUtils().make_updir(file_name=log_file)
    except:
        pass
    fileinfo = logging.FileHandler(log_file)

    controshow = logging.StreamHandler()
    controshow.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controshow)
    return logger
