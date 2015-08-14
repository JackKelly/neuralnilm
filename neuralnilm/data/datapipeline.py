from __future__ import print_function, division
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
            batch.input_after_processing[i] = self.apply_processing(
                seq.input, 'input')
            batch.target_after_processing[i] = self.apply_processing(
                seq.target, 'target')
        return batch

    @property
    def shapes(self):
        if self._shapes is None:
            seq = self.source.get_seq()
            processed_input = self.apply_processing(seq.input, 'input')
            processed_target = self.apply_processing(seq.target, 'target')
            self._shapes = {
                'input_shape_before_processing':
                    (self.num_seq_per_batch,) + seq.input.shape,
                'target_shape_before_processing':
                    (self.num_seq_per_batch,) + seq.target.shape,
                'input_shape_after_processing':
                    (self.num_seq_per_batch,) + processed_input.shape,
                'target_shape_after_processing':
                    (self.num_seq_per_batch,) + processed_target.shape
            }
        return self._shapes

    def apply_processing(self, data, net_input_or_target):
        """Applies `<input, target>_processing` to `data`.

        Parameters
        ----------
        data
        net_input_or_target : {'target', 'input}

        Returns
        -------
        processed_data
        """
        pass

    def apply_inverse_processing(self, data, net_input_or_target):
        """Applies the inverse of `<input, target>_processing` to `data`."""
        pass
