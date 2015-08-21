from __future__ import print_function, division
import numpy as np

import nilmtk
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


def load_nilmtk_activations(appliances, filename, sample_period, windows):
    """
    Parameters
    ----------
    appliances : list of strings
    filename : string
    sample_period : int
    windows : dict
        Structure example:
        {
            'train': {<building_i>: <window>},
            'unseen_activations_of_seen_appliances': {<building_i>: <window>},
            'unseen_appliances': {<building_i>: <window>}
        }

    Returns
    -------
    all_activations : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <appliance>: {
                 <building_name>: [<activations>]
        }}}
        Each activation is a pd.Series with DatetimeIndex.
    """
    logger.info("Loading NILMTK activations...")

    # Sanity check
    if set(windows.keys()) != set(DATA_FOLD_NAMES):
        raise ValueError(
            "`windows` must have these exact keys: '{}'.  Not '{}'."
            .format(DATA_FOLD_NAMES, windows.keys()))
    if (set(windows['train'].keys()) !=
            set(windows['unseen_activations_of_seen_appliances'].keys())):
        raise ValueError(
            "`train` and `unseen_activations_of_seen_appliances` must refer"
            " to exactly the same buildings.")

    # Load dataset
    dataset = nilmtk.DataSet(filename)

    all_activations = {}
    for fold, buildings_and_windows in windows.iteritems():
        activations = {appliance: {} for appliance in appliances}
        for building_i, window in buildings_and_windows.iteritems():
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
        activations = {appliance: data
                       for appliance, data in activations.iteritems() if data}
        all_activations[fold] = activations

    dataset.store.close()
    logger.info("Done loading NILMTK activations.")
    return all_activations


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
