from __future__ import print_function, division
import numpy as np
import pandas as pd

from .batch import Batch
from neuralnilm.utils import none_to_list


class DataPipeline(object):
    def __init__(self, source, num_seq_per_batch,
                 input_processing=None, target_processing=None):
        self.source = source
        self.num_seq_per_batch = num_seq_per_batch
        self.input_processing = none_to_list(input_processing)
        self.target_processing = none_to_list(target_processing)

    def get_batch(self, validation=False):
        batch = Batch()
        input_sequences = []
        target_sequences = []
        all_appliances = {}
        for i in range(self.num_seq_per_batch):
            seq = self.source.get_sequence(validation=validation)
            all_appliances[i] = seq.all_appliances
            input_sequences.append(seq.input[np.newaxis, :])
            target_sequences.append(seq.target[np.newaxis, :])

        batch.before_processing.input = np.concatenate(input_sequences)
        batch.before_processing.target = np.concatenate(target_sequences)
        del input_sequences
        del target_sequences

        batch.after_processing.input = self.apply_processing(
            batch.before_processing.input, 'input')
        batch.after_processing.target = self.apply_processing(
            batch.before_processing.target, 'target')

        batch.all_appliances = pd.concat(
            all_appliances, axis=1, names=['sequence', 'appliance'])
        return batch

    def apply_processing(self, data, net_input_or_target):
        """Applies `<input, target>_processing` to `data`.

        Parameters
        ----------
        data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        net_input_or_target : {'target', 'input}

        Returns
        -------
        processed_data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        """
        processing_steps = self._get_processing_steps(net_input_or_target)
        for step in processing_steps:
            data = step(data)
        return data

    def apply_inverse_processing(self, data, net_input_or_target):
        """Applies the inverse of `<input, target>_processing` to `data`.

        Parameters
        ----------
        data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        net_input_or_target : {'target', 'input}

        Returns
        -------
        processed_data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        """
        processing_steps = self._get_processing_steps(net_input_or_target)
        reversed_processing_steps = processing_steps[::-1]
        for step in reversed_processing_steps:
            try:
                data = step.inverse(data)
            except AttributeError:
                pass
        return data

    def _get_processing_steps(self, net_input_or_target):
        assert net_input_or_target in ['input', 'target']
        attribute = net_input_or_target + '_processing'
        processing_steps = getattr(self, attribute)
        assert isinstance(processing_steps, list)
        return processing_steps
