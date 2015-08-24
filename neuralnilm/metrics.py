from __future__ import print_function, division
import numpy as np
import sklearn.metrics as metrics

METRICS = {
    'classification': [
        'accuracy_score',
        'f1_score',
        'precision_score',
        'recall_score'
    ],
    'regression': [
        'mean_absolute_error'
    ]
}


class Metrics(object):
    def __init__(self, state_boundaries, clip_to_zero=True):
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
            {'regression': {'mse': <>},
             '2_state_classification': {'f1': <>, ...},
             '3_state_classification': {'f1': <>, ...}
            }
        """
        if output.shape != target.shape:
            raise ValueError("`output.shape` != `target.shape`")

        flat_output = output.flatten()
        flat_target = target.flatten()

        if self.clip_to_zero:
            flat_output[flat_output < self.state_boundaries[0]] = 0
            flat_target[flat_target < self.state_boundaries[0]] = 0

        # Get class labels
        output_class = np.zeros(flat_output.shape)
        target_class = np.zeros(flat_target.shape)
        for i, power_threshold in enumerate(self.state_boundaries):
            class_label = i + 1
            output_class[flat_output >= power_threshold] = class_label
            target_class[flat_target >= power_threshold] = class_label

        # Compute metrics
        ARGS = {
            'classification': '(target_class, output_class)',
            'regression': '(flat_target, flat_output)'
        }

        scores = {}
        n_states = len(self.state_boundaries) + 1
        for metric_type, metric_list in METRICS.iteritems():
            if metric_type == 'classification':
                metric_type_name = '{:d}_state_classification'.format(n_states)
            else:
                metric_type_name = metric_type
            args = ARGS[metric_type]
            scores[metric_type_name] = {}
            for metric in metric_list:
                score = eval('metrics.' + metric + args)
                scores[metric_type_name][metric] = float(score)

        scores['regression']['relative_error_in_total_energy'] = (
            relative_error_in_total_energy(flat_output, flat_target))

        return scores


def relative_error_in_total_energy(output, target):
    """Negative means under-estimates."""
    sum_output = np.sum(output)
    sum_target = np.sum(target)
    return float(
        (sum_output - sum_target) / max(sum_output, sum_target))
