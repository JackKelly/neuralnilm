from __future__ import print_function, division
import numpy as np
import pandas as pd
from neuralnilm.data.source import Sequence
from neuralnilm.data.activationssource import ActivationsSource

import logging
logger = logging.getLogger(__name__)


class SyntheticAggregateSource(ActivationsSource):
    def __init__(self, activations, target_appliance, seq_length,
                 sample_period,
                 distractor_inclusion_prob=0.25,
                 target_inclusion_prob=0.5,
                 uniform_prob_of_selecting_each_building=True,
                 allow_incomplete_target=True,
                 allow_incomplete_distractors=True,
                 include_incomplete_target_in_output=True,
                 rng_seed=None):
        self.activations = activations
        self.target_appliance = target_appliance
        self.seq_length = seq_length
        self.sample_period = sample_period
        self.distractor_inclusion_prob = distractor_inclusion_prob
        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_building = (
            uniform_prob_of_selecting_each_building)
        self.allow_incomplete_target = allow_incomplete_target
        self.allow_incomplete_distractors = allow_incomplete_distractors
        self.include_incomplete_target_in_output = (
            include_incomplete_target_in_output)
        super(SyntheticAggregateSource, self).__init__(rng_seed=rng_seed)

    def _get_sequence(self, fold='train', enable_all_appliances=False):
        seq = Sequence(self.seq_length)
        all_appliances = {}

        # Target appliance
        if self.rng.binomial(n=1, p=self.target_inclusion_prob):
            building_name = self._select_building(fold, self.target_appliance)
            activations = (
                self.activations[fold][self.target_appliance][building_name])
            activation_i = self._select_activation(activations)
            activation = activations[activation_i]
            positioned_activation, is_complete = self._position_activation(
                activation, is_target_appliance=True)
            positioned_activation = positioned_activation.values
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
            building_name = self._select_building(fold, appliance)
            activations = self.activations[fold][appliance][building_name]
            activation_i = self._select_activation(activations)
            activation = activations[activation_i]
            positioned_activation, is_complete = self._position_activation(
                activation, is_target_appliance=False)
            positioned_activation = positioned_activation.values
            seq.input += positioned_activation
            if enable_all_appliances:
                all_appliances[appliance] = positioned_activation

        seq.input = seq.input[:, np.newaxis]
        seq.target = seq.target[:, np.newaxis]
        assert len(seq.input) == self.seq_length
        assert len(seq.target) == self.seq_length        
        if enable_all_appliances:
            seq.all_appliances = pd.DataFrame(all_appliances)
        return seq
