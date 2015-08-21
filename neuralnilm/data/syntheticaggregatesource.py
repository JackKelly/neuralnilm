from __future__ import print_function, division
from copy import copy
import numpy as np
import pandas as pd
from neuralnilm.data.source import Source, Sequence
from neuralnilm.utils import flatten
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


class SyntheticAggregateSource(Source):
    def __init__(self, activations, target_appliance, seq_length,
                 distractor_inclusion_prob=1.0,
                 target_inclusion_prob=1.0,
                 uniform_prob_of_selecting_each_model=True,
                 allow_incomplete_target=True,
                 allow_incomplete_distractors=True,
                 include_incomplete_target_in_output=True,
                 rng_seed=None):
        self.activations = activations
        self.target_appliance = target_appliance
        self.seq_length = seq_length
        self.distractor_inclusion_prob = distractor_inclusion_prob
        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_model = (
            uniform_prob_of_selecting_each_model)
        self.allow_incomplete_target = allow_incomplete_target
        self.allow_incomplete_distractors = allow_incomplete_distractors
        self.include_incomplete_target_in_output = (
            include_incomplete_target_in_output)
        super(SyntheticAggregateSource, self).__init__(rng_seed=rng_seed)

    def get_sequence(self, fold='train', enable_all_appliances=False):
        seq = Sequence(self.seq_length)
        all_appliances = {}

        # Target appliance
        if self.rng.binomial(n=1, p=self.target_inclusion_prob):
            activation = self._select_activation(fold, self.target_appliance)
            positioned_activation, is_complete = self._position_activation(
                activation.values, is_target_appliance=True)
            seq.input += positioned_activation
            if enable_all_appliances:
                all_appliances[self.target_appliance] = positioned_activation
            if is_complete or self.include_incomplete_target_in_output:
                seq.target += positioned_activation

        # Distractor appliances
        distractor_appliances = [
            appliance for appliance in self._distractor_appliances(fold)
            if self.rng.binomial(n=1, p=self.distractor_inclusion_prob)]

        for appliance in distractor_appliances:
            activation = self._select_activation(fold, appliance)
            positioned_activation, is_complete = self._position_activation(
                activation.values, is_target_appliance=False)
            seq.input += positioned_activation
            if enable_all_appliances:
                all_appliances[appliance] = positioned_activation

        seq.input = seq.input[:, np.newaxis]
        seq.target = seq.target[:, np.newaxis]
        if enable_all_appliances:
            seq.all_appliances = pd.DataFrame(all_appliances)
        return seq

    def _distractor_appliances(self, fold):
        all_appliances = set(self.activations[fold].keys())
        distractor_appliances = all_appliances - set([self.target_appliance])
        return list(distractor_appliances)

    def _select_activation(self, fold, appliance):
        if fold not in DATA_FOLD_NAMES:
            raise ValueError("`fold` must be one of '{}' not '{}'."
                             .format(DATA_FOLD_NAMES, fold))

        activations_per_model = self.activations[fold][appliance]
        if self.uniform_prob_of_selecting_each_model:
            n_models = len(activations_per_model)
            if n_models == 0:
                raise RuntimeError("No appliance models for " + appliance)
            model_i = self.rng.randint(low=0, high=n_models)
            activations = activations_per_model.values()[model_i]
        else:
            activations = flatten(activations_per_model.values())
        n_activations = len(activations)
        if n_activations == 0:
            raise RuntimeError("No appliance activations for " + appliance)
        activation_i = self.rng.randint(low=0, high=n_activations)
        activation = activations[activation_i]
        return activation

    def _position_activation(self, activation, is_target_appliance):
        if is_target_appliance:
            allow_incomplete = self.allow_incomplete_target
        else:
            allow_incomplete = self.allow_incomplete_distractors

        # Select a start index
        if allow_incomplete:
            earliest_start_i = -len(activation)
            latest_start_i = self.seq_length
        else:
            if len(activation) > self.seq_length:
                raise RuntimeError("Activation too long to fit into sequence"
                                   " and incomplete activations not allowed.")
            earliest_start_i = 0
            latest_start_i = self.seq_length - len(activation)

        start_i = self.rng.randint(low=earliest_start_i, high=latest_start_i)

        # Clip or pad head of sequence
        if start_i < 0:
            positioned_activation = activation[-start_i:]
        else:
            positioned_activation = np.pad(
                activation, pad_width=(start_i, 0), mode='constant')

        # Clip or pad tail to produce a sequence which is seq_length long
        if len(positioned_activation) <= self.seq_length:
            n_zeros_to_pad = self.seq_length - len(positioned_activation)
            positioned_activation = np.pad(
                positioned_activation, pad_width=(0, n_zeros_to_pad),
                mode='constant')
        else:
            positioned_activation = positioned_activation[:self.seq_length]

        if len(activation) > self.seq_length:
            is_complete = False
        else:
            space_after_activation = self.seq_length - len(activation)
            is_complete = 0 <= start_i <= space_after_activation
        return positioned_activation, is_complete

    def report(self):
        report = copy(self.__dict__)
        report.pop('activations')
        report.pop('rng')
        return {self.__class__.__name__: report}
