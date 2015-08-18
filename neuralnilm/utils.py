from __future__ import print_function, division
import logging
from sys import stdout


def none_to_dict(data):
    return {} if data is None else data


def configure_logger(output_filename=None):
    logger = logging.getLogger("neuralnilm")
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(message)s')
        if output_filename:
            fh = logging.FileHandler(output_filename)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(stream=stdout))
    logger.setLevel(logging.DEBUG)


def flatten(_list):
    """Flatten a 2D list to 1D"""
    # From http://stackoverflow.com/a/952952
    return [item for sublist in _list for item in sublist]
