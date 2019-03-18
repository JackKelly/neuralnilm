from __future__ import print_function, division
import logging
from sys import stdout
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from neuralnilm.consts import DATA_FOLD_NAMES


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


def check_windows(windows):
    if set(windows.keys()) != set(DATA_FOLD_NAMES):
        raise ValueError(
            "`windows` must have these exact keys: '{}'.  Not '{}'."
            .format(DATA_FOLD_NAMES, windows.keys()))
    if (set(windows['train'].keys()) !=
            set(windows['unseen_activations_of_seen_appliances'].keys())):
        raise ValueError(
            "`train` and `unseen_activations_of_seen_appliances` must refer"
            " to exactly the same buildings.")


def sanitise_value_for_mogno(value):
    if isinstance(value, dict):
        value = sanitise_dict_for_mongo(value)
    elif isinstance(value, list) or isinstance(value, np.ndarray):
        value = [sanitise_value_for_mogno(item) for item in value]
    elif isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        value = int(value)
    return value


def sanitise_dict_for_mongo(dictionary):
    """Convert dict keys to strings (Mongo doesn't like numeric keys)"""
    new_dict = {}
    for key, value in dictionary.items():
        new_dict[str(key)] = sanitise_value_for_mogno(value)

    return new_dict


def two_level_dict_to_series(dictionary):
    index = []
    values = []
    for k0, v0 in dictionary.items():
        for k1, v1 in v0.items():
            index.append((k0, k1))
            values.append(v1)
    return pd.Series(values, pd.MultiIndex.from_tuples(index))


def two_level_series_to_dict(series):
    dictionary = {}
    for (k0, k1), value in series.items():
        dictionary.setdefault(k0, {})[k1] = value
    return dictionary


def get_colors(n, cmap_name='jet'):
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    return colors


def select_windows(train_buildings, unseen_buildings, original_windows):
    windows = {fold: {} for fold in DATA_FOLD_NAMES}

    def copy_window(fold, i):
        windows[fold][i] = original_windows[fold][i]

    for i in train_buildings:
        copy_window('train', i)
        copy_window('unseen_activations_of_seen_appliances', i)
    for i in unseen_buildings:
        copy_window('unseen_appliances', i)
    return windows


def filter_activations(windows, activations, appliances):
    new_activations = {
        fold: {appliance: {} for appliance in appliances}
        for fold in DATA_FOLD_NAMES}
    for fold, appliances in activations.items():
        for appliance, buildings in appliances.items():
            required_building_ids = windows[fold].keys()
            required_building_names = [
                'UK-DALE_building_{}'.format(i) for i in required_building_ids]
            for building_name in required_building_names:
                try:
                    new_activations[fold][appliance][building_name] = (
                        activations[fold][appliance][building_name])
                except KeyError:
                    pass
    return activations
