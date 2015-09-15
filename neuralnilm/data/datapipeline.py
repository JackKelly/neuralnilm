from __future__ import print_function, division
from copy import copy
import numpy as np

from neuralnilm.utils import none_to_list


class DataPipeline(object):
    def __init__(self, sources, num_seq_per_batch,
                 input_processing=None,
                 target_processing=None,
                 source_probabilities=None,
                 rng_seed=None):
        self.sources = sources
        self.num_seq_per_batch = num_seq_per_batch
        self.input_processing = none_to_list(input_processing)
        self.target_processing = none_to_list(target_processing)
        num_sources = len(self.sources)
        if source_probabilities is None:
            self.source_probabilities = [1 / num_sources] * num_sources
        else:
            self.source_probabilities = source_probabilities
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(self.rng_seed)
        self._source_iterators = [None] * num_sources

    def get_batch(self, fold='train', enable_all_appliances=False,
                  source_id=None, reset_iterator=False,
                  validation=False):
        """
        Returns
        -------
        A Batch object or None if source iterator has hit a StopIteration.
        """
        if source_id is None:
            n = len(self.sources)
            source_id = self.rng.choice(n, p=self.source_probabilities)
        if reset_iterator or self._source_iterators[source_id] is None:
            self._source_iterators[source_id] = (
                self.sources[source_id].get_batch(
                    num_seq_per_batch=self.num_seq_per_batch,
                    fold=fold,
                    enable_all_appliances=enable_all_appliances,
                    validation=validation))
        try:
            batch = self._source_iterators[source_id].next()
        except StopIteration:
            self._source_iterators[source_id] = None
            return None
        else:
            batch.after_processing.input, i_metadata = self.apply_processing(
                batch.before_processing.input, 'input')
            batch.after_processing.target, t_metadata = self.apply_processing(
                batch.before_processing.target, 'target')
            batch.metadata.update({
                'source_id': source_id,
                'processing': {
                    'input': i_metadata,
                    'target': t_metadata
                }
            })
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
        processed_data, metadata
        processed_data : np.ndarray
            shape = (num_seq_per_batch, seq_length, num_features)
        metadata : dict
        """
        processing_steps = self._get_processing_steps(net_input_or_target)
        metadata = {}
        for step in processing_steps:
            data = step(data)
            if hasattr(step, 'metadata'):
                metadata.update(step.metadata)
        return data, metadata

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

    def report(self):
        report = copy(self.__dict__)
        for attr in ['sources', 'rng', '_source_iterators']:
            report.pop(attr)
        report['sources'] = {
            i: source.report() for i, source in enumerate(self.sources)}
        report['input_processing'] = [
            processor.report() for processor in self.input_processing]
        report['target_processing'] = [
            processor.report() for processor in self.target_processing]
        return {'pipeline': report}
