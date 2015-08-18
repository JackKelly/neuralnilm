from __future__ import print_function, division
import numpy as np

import nilmtk
from neuralnilm.utils import none_to_dict
from .activationsgetter import ActivationsGetter

import logging
logger = logging.getLogger(__name__)


class NILMTKActivationsGetter(ActivationsGetter):
    def __init__(self, appliances, filename, buildings, sample_period,
                 window_per_building=None):
        self.filename = filename
        self.buildings = buildings
        self.window_per_building = none_to_dict(window_per_building)
        self.number_of_activations_loaded = {}
        super(NILMTKActivationsGetter, self).__init__(
            appliances=appliances, sample_period=sample_period)

    def load_activations(self):
        """
        Returns
        -------
        activations : dict
            Structure example:
            {'kettle': {'UK-DALE_building_1': [<activations>]}}
        """
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
