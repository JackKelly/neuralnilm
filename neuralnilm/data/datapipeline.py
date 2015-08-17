from __future__ import print_function, division
import numpy as np
from .batch import Batch


class DataPipeline(object):
    def __init__(self, source, num_seq_per_batch,
                 input_processing=None, target_processing=None):
        self.source = source
        self.num_seq_per_batch = num_seq_per_batch
        self.input_processing = input_processing
        self.target_processing = target_processing
        self._shapes = None

    def get_batch(self, validation=False):
        batch = Batch(**self.shapes)
        for i in range(self.num_seq_per_batch):
            seq = self.source.get_sequence(validation=validation)
            batch.all_appliances.append(seq.all_appliances)
            batch.input_before_processing[i] = seq.input
            batch.target_before_processing[i] = seq.target

        batch.input_after_processing = self.apply_processing(
            batch.input_before_processing, 'input')
        batch.target_after_processing = self.apply_processing(
            batch.target_before_processing, 'target')

        return batch

    @property
    def shapes(self):
        if self._shapes is None:
            seq = self.source.get_seq()
            processed_input = self.apply_processing(
                np.expand_dims(seq.input, axis=0), 'input')
            processed_target = self.apply_processing(
                np.expand_dims(seq.target, axis=0), 'target')
            self._shapes = {
                'input_shape_before_processing':
                    (self.num_seq_per_batch,) + seq.input.shape,
                'target_shape_before_processing':
                    (self.num_seq_per_batch,) + seq.target.shape,
                'input_shape_after_processing':
                    (self.num_seq_per_batch,) + processed_input.shape[1:],
                'target_shape_after_processing':
                    (self.num_seq_per_batch,) + processed_target.shape[1:]
            }
        return self._shapes

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
