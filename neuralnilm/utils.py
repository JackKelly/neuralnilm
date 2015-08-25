from __future__ import print_function, division
import logging
from sys import stdout
import csv
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
        if not is_running_in_ipython_notebook():
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


class ANSI:
    # from dnouri/nolearn/nolearn/lasagne.py
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


def is_running_in_ipython_notebook():
    """Returns True if code is running in an IPython notebook."""
    # adapted from http://stackoverflow.com/a/24937408
    try:
        cfg = get_ipython().config
        return 'jupyter' in cfg['IPKernelApp']['connection_file']
    except (NameError, TypeError):
        return False


def write_csv_row(filename, row, mode='a'):
    with open(filename, mode=mode) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)


def downsample(array, factor):
    """
    Parameters
    ----------
    array : 1D np.ndarray
    factor : int
    """
    assert isinstance(factor, int)
    if array.size % factor:
        num_zeros_to_append = factor - (array.size % factor)
        array = np.pad(
            array, pad_width=(0, num_zeros_to_append), mode='reflect')
    array = array.reshape(-1, factor)
    downsampled = array.mean(axis=1)
    return downsampled
