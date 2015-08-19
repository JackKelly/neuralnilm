from __future__ import print_function, division
import numpy as np

import nilmtk
from neuralnilm.utils import none_to_dict

import logging
logger = logging.getLogger(__name__)


def load_nilmtk_activations(appliances, filename, buildings, sample_period,
                            window_per_building=None, window=None):
    """
    Returns
    -------
    activations : dict
        Structure example:
        {'kettle': {'UK-DALE_building_1': [<activations>]}}
        Each activation is a pd.Series with DatetimeIndex and
        name == <appliance type>.
    """
    assert not (window and window_per_building)
    if window is None:
        window_per_building = none_to_dict(window_per_building)
    else:
        window_per_building = {building_i: window for building_i in buildings}
    dataset = nilmtk.DataSet(filename)
    activations = {appliance: {} for appliance in appliances}
    for building_i in buildings:
        window = window_per_building.get(building_i, (None, None))
        dataset.set_window(*window)
        elec = dataset.buildings[building_i].elec
        building_name = (
            dataset.metadata['name'] + '_building_{}'.format(building_i))
        for appliance in appliances:
            logger.info(
                "Loading {} for {}...".format(appliance, building_name))
            try:
                meter = elec[appliance]
            except KeyError as exception:
                logger.info(building_name + " has no " + appliance +
                            ". Full exception: {}".format(exception))
                continue
            meter_activations = meter.get_activations()
            meter_activations = _process_activations(
                meter_activations, sample_period)
            if meter_activations:
                activations[appliance][building_name] = meter_activations
            logger.info(
                "Loaded {} {} activations from {}."
                .format(len(meter_activations), appliance, building_name))

    dataset.store.close()
    activations = {appliance: data
                   for appliance, data in activations.iteritems()
                   if data}
    return activations


def _process_activations(activations, sample_period):
    for i, activation in enumerate(activations):
        # tz_convert('UTC') is a workaround for Pandas bug #10117
        tz = activation.index.tz.zone
        activation = activation.tz_convert('UTC')
        freq = "{:d}S".format(sample_period)
        activation = activation.resample(freq)
        activation.fillna(method='ffill', inplace=True)
        activation.fillna(method='bfill', inplace=True)
        activation = activation.tz_convert(tz)
        activation = activation.astype(np.float32)
        activations[i] = activation
    return activations
