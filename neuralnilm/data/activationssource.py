from __future__ import print_function, division
from copy import copy
from datetime import timedelta
import numpy as np
import pandas as pd
from neuralnilm.data.source import Source
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


class ActivationsSource(Source):
    """Abstract base class for holding common behaviour across subclasses.

    Attributes
    ----------
    activations: dict:
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <appliance>: {
                 <building_name>: [<activations>]
        }}}
        Each activation is a pd.Series with DatetimeIndex and the following
        metadata attributes: building, appliance, fold.
    """

    def __init__(self, **kwargs):
        if not self.allow_incomplete_target:
            self._remove_over_long_activations(self.target_appliance)
        super(ActivationsSource, self).__init__(**kwargs)

    def report(self):
        report = super(ActivationsSource, self).report()
        report['num_activations'] = self.get_num_activations()
        return report

    def get_num_activations(self):
        num_activations = {}
        for fold, appliances in self.activations.iteritems():
            for appliance, buildings in appliances.iteritems():
                for building_name, activations in buildings.iteritems():
                    num_activations.setdefault(fold, {}).setdefault(
                        appliance, {})[building_name] = len(activations)
        return num_activations

    def get_sequence(self, fold='train', enable_all_appliances=False):
        while True:
            yield self._get_sequence(
                fold=fold, enable_all_appliances=enable_all_appliances)

    def _distractor_appliances(self, fold):
        all_appliances = set(self.activations[fold].keys())
        distractor_appliances = all_appliances - set([self.target_appliance])
        return list(distractor_appliances)

    def _remove_over_long_activations(self, appliance_to_filter):
        new_activations = {}
        for fold, appliances in self.activations.iteritems():
            new_activations[fold] = {}
            for appliance, buildings in appliances.iteritems():
                new_activations[fold][appliance] = {}
                if appliance == appliance_to_filter:
                    for building, activations in buildings.iteritems():
                        new_activations[fold][appliance][building] = [
                            activation for activation in activations
                            if len(activation) < self.seq_length]
                else:
                    new_activations[fold][appliance] = buildings
        self.activations = new_activations

    def _select_building(self, fold, appliance):
        """
        Parameters
        ----------
        fold : str
        appliance : str

        Returns
        -------
        building_name : str
        """
        if fold not in DATA_FOLD_NAMES:
            raise ValueError("`fold` must be one of '{}' not '{}'."
                             .format(DATA_FOLD_NAMES, fold))

        activations_per_building = self.activations[fold][appliance]

        # select `p` for np.random.choice
        if self.uniform_prob_of_selecting_each_building:
            p = None  # uniform distribution over all buildings
        else:
            num_activations_per_building = np.array([
                len(activations) for activations in
                activations_per_building.values()])
            p = (num_activations_per_building /
                 num_activations_per_building.sum())

        num_buildings = len(activations_per_building)
        building_i = self.rng.choice(num_buildings, p=p)
        building_name = activations_per_building.keys()[building_i]
        return building_name

    def _select_activation(self, activations):
        num_activations = len(activations)
        if num_activations == 0:
            raise RuntimeError("No appliance activations.")
        activation_i = self.rng.randint(low=0, high=num_activations)
        return activation_i

    def _position_activation(self, activation, is_target_appliance):
        """
        Parameters
        ----------
        activation : pd.Series
        is_target_appliance : bool

        Returns
        -------
        pd.Series
        """
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
            positioned_activation = activation.values[-start_i:]
        else:
            positioned_activation = np.pad(
                activation.values, pad_width=(start_i, 0), mode='constant')

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

        seq_start_time = activation.index[0] - timedelta(
            seconds=start_i * self.sample_period)
        index = pd.date_range(seq_start_time, periods=self.seq_length,
                              freq="{:d}S".format(self.sample_period))
        positioned_activation_series = pd.Series(
            positioned_activation, index=index)
        return positioned_activation_series, is_complete
