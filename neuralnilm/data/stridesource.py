from __future__ import print_function, division
from copy import copy
from datetime import timedelta
import numpy as np
import pandas as pd
import nilmtk
from nilmtk.timeframegroup import TimeFrameGroup
from nilmtk.timeframe import TimeFrame
from neuralnilm.data.source import Sequence
from neuralnilm.utils import check_windows
from neuralnilm.data.source import Source
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


class StrideSource(Source):
    """
    Attributes
    ----------
    data : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <building_name>: pd.DataFrame of with 2 cols: mains, target
        }}
    _seq_i : dict of ints:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>:
           int}
        How many sequences have been emitted so far.  Loops back to 0 when
        all the data has been gone through.
    _num_seqs : pd.Series with 2-level hierarchical index
        L0 : train, unseen_appliances, unseen_activations_of_seen_appliances
        L1 : building_names
    """
    def __init__(self, target_appliance,
                 seq_length, filename, windows, sample_period,
                 stride=None,
                 rng_seed=None):
        self.target_appliance = target_appliance
        self.seq_length = seq_length
        self.filename = filename
        check_windows(windows)
        self.windows = windows
        self.sample_period = sample_period
        self.stride = self.seq_length if stride is None else stride
        self._reset()
        super(StrideSource, self).__init__(rng_seed=rng_seed)

        self._load_data_into_memory()
        self._compute_num_sequences_per_building()

    def _reset(self):
        self._seq_i = {fold: 0 for fold in DATA_FOLD_NAMES}
        self.data = {}
        self._num_seqs = pd.Series()

    def _load_data_into_memory(self):
        logger.info("Loading NILMTK data...")

        # Load dataset
        dataset = nilmtk.DataSet(self.filename)

        for fold, buildings_and_windows in self.windows.iteritems():
            for building_i, window in buildings_and_windows.iteritems():
                dataset.set_window(*window)
                elec = dataset.buildings[building_i].elec
                building_name = (
                    dataset.metadata['name'] +
                    '_building_{}'.format(building_i))

                # Mains
                logger.info(
                    "Loading data for {}...".format(building_name))

                mains_meter = elec.mains()
                mains_good_sections = mains_meter.good_sections()

                appliance_meter = elec[self.target_appliance]
                good_sections = appliance_meter.good_sections(
                    sections=mains_good_sections)

                def load_data(meter):
                    return meter.power_series_all_data(
                        sample_period=self.sample_period,
                        sections=good_sections).dropna()

                mains_data = load_data(mains_meter)
                appliance_data = load_data(appliance_meter)
                df = pd.DataFrame(
                    {'mains': mains_data, 'target': appliance_data},
                    dtype=np.float32).dropna()
                del mains_data
                del appliance_data

                if not df.empty:
                    self.data.setdefault(fold, {})[building_name] = df

                logger.info(
                    "Loaded data from building {} for fold {}"
                    " from {} to {}."
                    .format(building_name, fold, df.index[0], df.index[-1]))

        dataset.store.close()
        logger.info("Done loading NILMTK mains data.")

    def _compute_num_sequences_per_building(self):
        index = []
        all_num_seqs = []
        for fold, buildings in self.data.iteritems():
            for building_name, df in buildings.iteritems():
                remainder = len(df) - self.seq_length
                num_seqs = np.ceil(remainder / self.stride) + 1
                num_seqs = max(0 if df.empty else 1, int(num_seqs))
                index.append((fold, building_name))
                all_num_seqs.append(num_seqs)
        multi_index = pd.MultiIndex.from_tuples(
            index, names=["fold", "building_name"])
        self._num_seqs = pd.Series(all_num_seqs, multi_index)

    def get_sequence(self, fold='train', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for StrideSource!")

        # select building
        seq_i = self._seq_i[fold]
        building_divisions = self._num_seqs[fold].cumsum()
        prev_seq_cumsum = 0
        for row_building, row_seq_cumsum in building_divisions.iteritems():
            if prev_seq_cumsum <= seq_i < row_seq_cumsum:
                building_name = row_building
                seq_i_for_building = seq_i - prev_seq_cumsum
                break
            else:
                prev_seq_cumsum = row_seq_cumsum
        else:
            # Wrap round!
            seq_i = 0
            building_name = building_divisions.index[0]
            seq_i_for_building = 0

        start_i = seq_i_for_building * self.stride
        end_i = start_i + self.seq_length
        data_for_seq = self.data[fold][building_name].iloc[start_i:end_i]

        def get_data(col):
            data = data_for_seq[col].values
            n_zeros_to_pad = self.seq_length - len(data)
            data = np.pad(data, pad_width=(0, n_zeros_to_pad), mode='constant')
            return data[:, np.newaxis]

        seq = Sequence(self.seq_length)
        seq.input = get_data('mains')
        seq.target = get_data('target')
        seq.metadata = {
            'seq_i': seq_i,
            'building_name': building_name,
            'total_num_sequences': self._num_seqs[fold].sum(),
            'start_date': data_for_seq.index[0],
            'end_date': data_for_seq.index[-1]
        }

        self._seq_i[fold] = seq_i + 1
        return seq

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return ['data']
