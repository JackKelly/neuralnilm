from __future__ import print_function, division
import numpy as np


class Sequence(object):
    """
    Attributes
    ----------
    input : np.ndarray
    target : np.ndarray
    all_appliances : pd.DataFrame
        Column names are the appliance names.
    """
    def __init__(self):
        self.input = None
        self.target = None
        self.all_appliances = None


class Source(object):
    def __init__(self, rng_seed=None):
        self.rng = np.random.RandomState(rng_seed)

    def get_sequence(self, validation=False):
        """
        Returns
        -------
        sequence : Sequence
        """
        raise NotImplementedError()
