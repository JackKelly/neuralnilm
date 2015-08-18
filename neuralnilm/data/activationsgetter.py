from __future__ import print_function, division
import numpy as np


class ActivationsGetter(object):
    def __init__(self, appliances, sample_period):
        self.appliances = appliances
        self.sample_period = sample_period

    def load_activations(self):
        """
        Returns
        -------
        activations : dict
            Structure example:
            {'kettle': {'UK-DALE_building_1': [<activations>]}}
        """
        raise NotImplementedError("Subclass must implement this method.")

    def _get_empty_activations_dict(self):
        return {appliance: {} for appliance in self.appliances}

    def _process_activations(self, activations):
        for i, activation in enumerate(activations):
            # tz_convert('UTC') is a workaround for Pandas bug #10117
            tz = activation.index.tz.zone
            activation = activation.tz_convert('UTC')
            freq = "{:d}S".format(self.sample_period)
            activation = activation.resample(freq)
            activation.fillna(method='ffill', inplace=True)
            activation.fillna(method='bfill', inplace=True)
            activation = activation.tz_convert(tz)
            activations[i] = activation.astype(np.float32)
        return activations

    def report(self):
        return self.__dict__
