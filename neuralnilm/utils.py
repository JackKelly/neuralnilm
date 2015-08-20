from __future__ import print_function, division
import logging
from sys import stdout
import numpy as np
import theano
import theano.tensor as T


def none_to_dict(data):
    return {} if data is None else data


def none_to_list(data):
    return [] if data is None else data


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


def sfloatX(data):
    """Convert scalar to floatX"""
    return getattr(np, theano.config.floatX)(data)


def ndim_tensor(name, ndim, dtype=theano.config.floatX):
    tensor_type = T.TensorType(dtype=dtype, broadcastable=((False,) * ndim))
    return tensor_type(name=name)
