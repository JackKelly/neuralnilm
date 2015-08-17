from __future__ import print_function, division
import numpy as np

import nilmtk
from neuralnilm.utils import none_to_dict

import logging
logger = logging.getLogger(__name__)


class NILMTKActivations(object):
    def __init__(self, appliances, filename, buildings, sample_period,
                 window_per_building=None):

        self.appliances = appliances
        self.filename = filename
        self.buildings = buildings
        self.sample_period = sample_period
        self.window_per_building = none_to_dict(window_per_building)
        self.number_of_activations_loaded = {}

    def _get_empty_activations_dict(self):
        return {appliance: {} for appliance in self.appliances}

    def load_activations(self):
        dataset = nilmtk.DataSet(self.filename)
        activations = self._get_empty_activations_dict()
        self.number_of_activations_loaded = self._get_empty_activations_dict()
        for building_i in self.buildings:
            window = self.window_per_building.get(building_i, (None, None))
            dataset.set_window(*window)
            elec = dataset.buildings[building_i].elec
            building_name = (
                dataset.metadata['name'] + '_building_{}'.format(building_i))
            for appliance in self.appliances:
                logger.info(
                    "Loading {} for {}...".format(appliance, building_name))
                try:
                    meter = elec[appliance]
                except KeyError as exception:
                    logger.info(building_name + " has no " + appliance +
                                ". Full exception: {}".format(exception))
                    continue
                meter_activations = meter.get_activations()
                meter_activations = self._process_activations(
                    meter_activations)
                activations[appliance][building_name] = meter_activations
                self.number_of_activations_loaded[appliance][building_name] = (
                    len(meter_activations))
                logger.info(
                    "Loaded {} {} activations from {}."
                    .format(len(meter_activations), appliance, building_name))

        dataset.store.close()
        return activations

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
