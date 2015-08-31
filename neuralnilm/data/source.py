from __future__ import print_function, division
from copy import copy
import numpy as np
import pandas as pd

from .batch import Batch


class Sequence(object):
    """
    Attributes
    ----------
    input : np.ndarray
    target : np.ndarray
    all_appliances : pd.DataFrame
        Column names are the appliance names.
    """
    def __init__(self, shape):
        self.input = np.zeros(shape, dtype=np.float32)
        self.target = np.zeros(shape, dtype=np.float32)
        self.all_appliances = pd.DataFrame()


class Source(object):
    def __init__(self, rng_seed=None):
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(rng_seed)

    def get_sequence(self, validation=False):
        """
        Returns
        -------
        sequence : Sequence
        """
        raise NotImplementedError()

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return ['activations', 'rng']

    def report(self):
        report = copy(self.__dict__)
        for attr in self._attrs_to_remove_for_report():
            report.pop(attr, None)
        return {self.__class__.__name__: report}

    def get_batch(self, num_seq_per_batch, fold='train',
                  enable_all_appliances=False):
        input_sequences = []
        target_sequences = []
        all_appliances = {}
        for i in range(num_seq_per_batch):
            seq = self.get_sequence(
                fold=fold,
                enable_all_appliances=enable_all_appliances)
            if enable_all_appliances:
                all_appliances[i] = seq.all_appliances
            input_sequences.append(seq.input[np.newaxis, :])
            target_sequences.append(seq.target[np.newaxis, :])

        batch = Batch()
        batch.metadata['fold'] = fold
        batch.metadata['source_name'] = self.__class__.__name__
        batch.before_processing.input = np.concatenate(input_sequences)
        del input_sequences
        batch.before_processing.target = np.concatenate(target_sequences)
        del target_sequences
        if enable_all_appliances:
            batch.all_appliances = pd.concat(
                all_appliances, axis=1, names=['sequence', 'appliance'])
        return batch
