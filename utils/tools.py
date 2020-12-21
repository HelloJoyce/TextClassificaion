import logging
from logging import handlers
import re
from utils import config
# import config
import jieba
import json
from tqdm import tqdm
# jieba.enable_parallel(4)
import pandas as pd
import time
from functools import partial, wraps
from datetime import timedelta

tqdm.pandas()


def clean_symbols(text):
    """
    对特殊符号做一些处理
    """
    text = re.sub('[0-9]+', " NUM ", str(text))
    text = re.sub('[!！]+', "!", text)
    #     text = re.sub('!', '', text)
    text = re.sub('[?？]+', "?", text)
    text = re.sub("[a-zA-Z#$%&\'()*+,-./:;：<=>@，。★、…【】《》“”‘’'!'[\\]^_`{|}~]+", " OOV ", text)
    return re.sub("\s+", " ", text)


def query_cut(query):
    return jieba.cut(query)


def create_logger(log_path):
    """
    日志的创建
    :param log_path:
    :return:
    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(filename=log_path,
                                           when='D',
                                           backupCount=3,
                                           encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger
