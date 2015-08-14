from __future__ import print_function, division
import numpy as np
from theano.config import floatX


class Batch(object):
    """
    Attributes
    ----------
    all_appliances : list of pd.DataFrames
        One entry for each seq in batch
    metadata : dict
    input_after_processing : np.ndarray
    target_after_processing : np.ndarray
    input_before_processing : np.ndarray
    target_before_processing : np.ndarray
    """
    def __init__(self,
                 input_shape_after_processing,
                 target_shape_after_processing,
                 input_shape_before_processing,
                 target_shape_before_processing):
        self.all_appliances = []
        self.metadata = {}
        self.input_after_processing = np.zeros(
            input_shape_after_processing, dtype=floatX)
        self.target_after_processing = np.zeros(
            target_shape_after_processing, dtype=floatX)
        self.input_before_processing = np.zeros(
            input_shape_before_processing, dtype=floatX)
        self.target_before_processing = np.zeros(
            target_shape_after_processing, dtype=floatX)
