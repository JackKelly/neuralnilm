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
        if source_probabilities is None:
            n = len(self.sources)
            self.source_probabilities = [1 / n] * n
        else:
            self.source_probabilities = source_probabilities
        self.rng_seed = rng_seed
        self.rng = np.random.RandomState(self.rng_seed)

    def get_batch(self, fold='train', enable_all_appliances=False,
                  source_id=None):
        if source_id is None:
            n = len(self.sources)
            source_id = self.rng.choice(n, p=self.source_probabilities)
        batch = self.sources[source_id].get_batch(
            num_seq_per_batch=self.num_seq_per_batch,
            fold=fold,
            enable_all_appliances=enable_all_appliances)
        batch.after_processing.input = self.apply_processing(
            batch.before_processing.input, 'input')
        batch.after_processing.target = self.apply_processing(
            batch.before_processing.target, 'target')
        batch.metadata['source_id'] = source_id
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

    def report(self):
        report = copy(self.__dict__)
        report.pop('sources')
        report.pop('rng')
        report['sources'] = {
            i: source.report() for i, source in enumerate(self.sources)}
        report['input_processing'] = [
            processor.report() for processor in self.input_processing]
        report['target_processing'] = [
            processor.report() for processor in self.target_processing]
        return {'pipeline': report}
