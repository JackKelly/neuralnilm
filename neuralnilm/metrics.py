from __future__ import print_function, division
from copy import copy
import numpy as np
import sklearn.metrics as metrics


def relative_error_in_total_energy(output, target):
    """Negative means under-estimates."""
    sum_output = np.sum(output)
    sum_target = np.sum(target)
    return float(
        (sum_output - sum_target) / max(sum_output, sum_target))


METRICS = {
    'classification': [
        metrics.accuracy_score,
        metrics.f1_score,
        metrics.precision_score,
        metrics.recall_score
    ],
    'regression': [
        metrics.mean_absolute_error,
        metrics.mean_squared_error,
        relative_error_in_total_energy
    ]
}


class Metrics(object):
    def __init__(self, state_boundaries, clip_to_zero=False):
        """
        Parameters
        ----------
        state_boundaries : list or tuple of numbers
            Define the state (power) boundaries.  For a two-state (on/off)
            appliance, this will be a list of [<on_power_threshold>].
        clip_to_zero : bool
            If True then clip all values in both `target` and `output`
            less than state_boundaries[0] to 0.
        """
        if not isinstance(state_boundaries, (list, tuple)):
            raise ValueError("`state_boundaries` must be a list or tuple.")
        if len(state_boundaries) == 0:
            raise ValueError("state_boundaries must have >= 1 element.")
        self.state_boundaries = state_boundaries
        self.clip_to_zero = clip_to_zero

    def compute_metrics(self, output, target):
        """
        Parameters
        ----------
        output, target : np.ndarray

        Returns
        -------
        scores : dict
            {
                'regression': {
                    'mean_absolute_error': 0.5
                },
                'classification_2_state': {
                    'f1_score': 0.5,
                    'precision_score': 0.8
                }
            }
        """
        if output.shape != target.shape:
            raise ValueError("`output.shape` != `target.shape`")

        flat_output = output.flatten()
        flat_target = target.flatten()

        if self.clip_to_zero:
            flat_output[flat_output < self.state_boundaries[0]] = 0
            flat_target[flat_target < self.state_boundaries[0]] = 0

        all_scores = {}

        # Classification
        for num_states in range(2, len(self.state_boundaries)+2):
            metric_type = 'classification_{:d}_state'.format(num_states)
            all_scores[metric_type] = self._get_classification_scores(
                flat_output, flat_target, num_states)

        # Regression
        regression_scores = {}
        for metric in METRICS['regression']:
            regression_scores[metric.__name__] = float(
                metric(flat_target, flat_output))
        all_scores['regression'] = regression_scores

        return all_scores

    def _get_classification_scores(self, flat_output, flat_target, num_states):
        # Get class labels
        output_class = np.zeros(flat_output.shape)
        target_class = np.zeros(flat_target.shape)
        state_boundaries = self.state_boundaries[:num_states-1]
        for i, power_threshold in enumerate(state_boundaries):
            class_label = i + 1
            output_class[flat_output >= power_threshold] = class_label
            target_class[flat_target >= power_threshold] = class_label

        # Get scores
        scores = {}
        for metric in METRICS['classification']:
            scores[metric.__name__] = float(metric(target_class, output_class))

        return scores

    def report(self):
        report = copy(self.__dict__)
        report['name'] = self.__class__.__name__
        return report
