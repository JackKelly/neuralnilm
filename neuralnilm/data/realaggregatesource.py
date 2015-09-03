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
from neuralnilm.data.activationssource import ActivationsSource
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


class RealAggregateSource(ActivationsSource):
    """
    Attributes
    ----------
    mains : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <building_name>: pd.Series of raw data
        }}
    mains_good_sections : dict
        Same structure as `mains`.
    sections_with_no_target : dict
        Same structure as `mains`.
    all_gaps : dict of pd.DataFrames
        Each key is a fold name.
        Each DF has columns:
        building, gap, duration, p (short for 'probability')
        p is used by _get_sequence_without_target().
    """
    def __init__(self, activations, target_appliance,
                 seq_length, filename, windows, sample_period,
                 target_inclusion_prob=0.5,
                 uniform_prob_of_selecting_each_building=True,
                 allow_incomplete_target=True,
                 include_incomplete_target_in_output=True,
                 allow_multiple_target_activations_in_aggregate=False,
                 include_multiple_targets_in_output=False,
                 rng_seed=None):
        self.activations = copy(activations)
        self.target_appliance = target_appliance
        self.seq_length = seq_length
        self.filename = filename
        check_windows(windows)
        self.windows = windows
        self.sample_period = sample_period
        self.target_inclusion_prob = target_inclusion_prob
        self.uniform_prob_of_selecting_each_building = (
            uniform_prob_of_selecting_each_building)
        self.allow_incomplete_target = allow_incomplete_target
        self.include_incomplete_target_in_output = (
            include_incomplete_target_in_output)
        self.allow_multiple_target_activations_in_aggregate = (
            allow_multiple_target_activations_in_aggregate)
        self.include_multiple_targets_in_output = (
            include_multiple_targets_in_output)
        super(RealAggregateSource, self).__init__(rng_seed=rng_seed)

        self._load_mains_into_memory()
        self._remove_activations_with_no_mains()
        self._find_sections_with_no_target()
        self._compute_gap_probabilities()

    def _load_mains_into_memory(self):
        logger.info("Loading NILMTK mains...")

        # Load dataset
        dataset = nilmtk.DataSet(self.filename)

        self.mains = {}
        self.mains_good_sections = {}
        for fold, buildings_and_windows in self.windows.iteritems():
            for building_i, window in buildings_and_windows.iteritems():
                dataset.set_window(*window)
                elec = dataset.buildings[building_i].elec
                building_name = (
                    dataset.metadata['name'] +
                    '_building_{}'.format(building_i))

                logger.info(
                    "Loading mains for {}...".format(building_name))

                mains_meter = elec.mains()
                good_sections = mains_meter.good_sections()
                mains_data = mains_meter.power_series_all_data(
                    sample_period=self.sample_period,
                    sections=good_sections).dropna()

                def set_mains_data(dictionary, data):
                    dictionary.setdefault(fold, {})[building_name] = data

                if not mains_data.empty:
                    set_mains_data(self.mains, mains_data)
                    set_mains_data(self.mains_good_sections, good_sections)

                logger.info(
                    "Loaded mains data from building {} for fold {}"
                    " from {} to {}."
                    .format(building_name, fold,
                            mains_data.index[0], mains_data.index[-1]))

        dataset.store.close()
        logger.info("Done loading NILMTK mains data.")

    def _find_sections_with_no_target(self):
        """Finds the intersections of the mains good sections with the gaps
        between target appliance activations.
        """
        self.sections_with_no_target = {}
        seq_length_secs = self.seq_length * self.sample_period
        for fold, sects_per_building in self.mains_good_sections.iteritems():
            for building, good_sections in sects_per_building.iteritems():
                activations = (
                    self.activations[fold][self.target_appliance][building])
                mains = self.mains[fold][building]
                mains_good_sections = self.mains_good_sections[fold][building]
                gaps_between_activations = TimeFrameGroup()
                prev_end = mains.index[0]
                for activation in activations:
                    gap = TimeFrame(prev_end, activation.index[0])
                    gaps_between_activations.append(gap)
                    prev_end = activation.index[-1]
                gap = TimeFrame(prev_end, mains.index[-1])
                gaps_between_activations.append(gap)
                intersection = (
                    gaps_between_activations.intersection(mains_good_sections))
                intersection = intersection.remove_shorter_than(
                    seq_length_secs)
                self.sections_with_no_target.setdefault(fold, {})[building] = (
                    intersection)
                logger.info("Found {} sections without target for {} {}."
                            .format(len(intersection), fold, building))

    def _compute_gap_probabilities(self):
        # Choose a building and a gap
        self.all_gaps = {}
        for fold in DATA_FOLD_NAMES:
            all_gaps_for_fold = []
            for building, gaps in self.sections_with_no_target[fold].iteritems():
                gaps_for_building = [
                    (building, gap, gap.timedelta.total_seconds())
                    for gap in gaps]
                all_gaps_for_fold.extend(gaps_for_building)
            gaps_df = pd.DataFrame(
                all_gaps_for_fold, columns=['building', 'gap', 'duration'])
            gaps_df['p'] = gaps_df['duration'] / gaps_df['duration'].sum()
            self.all_gaps[fold] = gaps_df

    def _get_sequence_without_target(self, fold):
        # Choose a building and a gap
        all_gaps_for_fold = self.all_gaps[fold]
        n = len(all_gaps_for_fold)
        gap_i = self.rng.choice(n, p=all_gaps_for_fold['p'])
        row = all_gaps_for_fold.iloc[gap_i]
        building, gap = row['building'], row['gap']

        # Choose a start point in the gap
        latest_start_time = gap.end - timedelta(
            seconds=self.seq_length * self.sample_period)
        max_offset_seconds = (latest_start_time - gap.start).total_seconds()
        if max_offset_seconds <= 0:
            offset = 0
        else:
            offset = self.rng.randint(max_offset_seconds)
        start_time = gap.start + timedelta(seconds=offset)
        end_time = start_time + timedelta(
            seconds=(self.seq_length + 1) * self.sample_period)
        mains = self.mains[fold][building][start_time:end_time]
        seq = Sequence(self.seq_length)
        seq.input = mains.values[:self.seq_length]
        return seq

    def _has_sufficient_samples(self, data, start, end, threshold=0.8):
        if len(data) < 2:
            return False
        num_expected_samples = (
            (end - start).total_seconds() / self.sample_period)
        hit_rate = len(data) / num_expected_samples
        return (hit_rate >= threshold)

    def _remove_activations_with_no_mains(self):
        # First remove any activations where there is no mains data at all
        for fold, activations_for_appliance in self.activations.iteritems():
            activations_for_buildings = activations_for_appliance[
                self.target_appliance]
            buildings_to_remove = []
            for building in activations_for_buildings:
                mains_for_fold = self.mains[fold]
                if (building not in mains_for_fold and
                        building not in buildings_to_remove):
                    buildings_to_remove.append(building)
            for building in buildings_to_remove:
                self.activations[fold][self.target_appliance].pop(building)

        # Now check for places where mains has insufficient samples,
        # for example because the mains series has a break in it.
        for fold, activations_for_appliance in self.activations.iteritems():
            activations_for_buildings = activations_for_appliance[
                self.target_appliance]
            for building, activations in activations_for_buildings.iteritems():
                mains = self.mains[fold][building]
                activations_to_remove = []
                for i, activation in enumerate(activations):
                    activation_duration = (
                        activation.index[-1] - activation.index[0])
                    start = activation.index[0] - activation_duration
                    end = activation.index[-1] + activation_duration
                    mains_for_activ = mains[start:end]
                    if (start < mains.index[0] or
                            end > mains.index[-1] or not
                            self._has_sufficient_samples(
                                mains_for_activ, start, end)):
                        activations_to_remove.append(i)
                if activations_to_remove:
                    logger.info(
                        "Removing {} activations from fold '{}' building '{}'"
                        " because there was not enough mains data for"
                        " these activations. This leaves {} activations."
                        .format(
                            len(activations_to_remove), fold, building,
                            len(activations) - len(activations_to_remove)))
                activations_to_remove.reverse()
                for i in activations_to_remove:
                    activations.pop(i)
                self.activations[fold][self.target_appliance][building] = (
                    activations)

    def _get_sequence(self, fold='train', enable_all_appliances=False):
        if enable_all_appliances:
            raise ValueError("`enable_all_appliances` is not implemented yet"
                             " for RealAggregateSource!")

        if self.rng.binomial(n=1, p=self.target_inclusion_prob):
            _seq_getter_func = self._get_sequence_which_includes_target
        else:
            _seq_getter_func = self._get_sequence_without_target

        MAX_RETRIES = 50
        for retry_i in range(MAX_RETRIES):
            seq = _seq_getter_func(fold=fold)
            if seq is None:
                continue
            if len(seq.input) != self.seq_length:
                continue
            if len(seq.target) != self.seq_length:
                continue
            break
        else:
            raise RuntimeError("No valid sequences found after {} retries!"
                               .format(MAX_RETRIES))

        seq.input = seq.input[:, np.newaxis]
        seq.target = seq.target[:, np.newaxis]
        assert len(seq.input) == self.seq_length
        assert len(seq.target) == self.seq_length
        return seq

    def _get_sequence_which_includes_target(self, fold):
        seq = Sequence(self.seq_length)
        building_name = self._select_building(fold, self.target_appliance)
        activations = (
            self.activations[fold][self.target_appliance][building_name])
        activation_i = self._select_activation(activations)
        activation = activations[activation_i]
        positioned_activation, is_complete = (
            self._position_activation(
                activation, is_target_appliance=True))
        if is_complete or self.include_incomplete_target_in_output:
            seq.target = positioned_activation
        else:
            seq.target = pd.Series(0, index=positioned_activation.index)

        # Check neighbouring activations
        mains_start = positioned_activation.index[0]
        mains_end = positioned_activation.index[-1]

        def neighbours_ok(neighbour_indicies):
            for i in neighbour_indicies:
                activation = activations[i]
                activation_duration = (
                    activation.index[-1] - activation.index[0])
                neighbouring_activation_is_inside_mains_window = (
                    activation.index[0] >
                    (mains_start - activation_duration)
                    and activation.index[0] < mains_end)

                if neighbouring_activation_is_inside_mains_window:
                    if self.allow_multiple_target_activations_in_aggregate:
                        if self.include_multiple_targets_in_output:
                            sum_target = seq.target.add(
                                activation, fill_value=0)
                            is_complete = (
                                sum_target.index == seq.target.index)
                            if self.allow_incomplete_target or is_complete:
                                seq.target = sum_target[seq.target.index]
                    else:
                        return False  # need to retry
                else:
                    return True  # everything checks out OK so far
            return True

        # Check forwards
        if not neighbours_ok(range(activation_i+1, len(activations))):
            return
        # Check backwards
        if not neighbours_ok(range(activation_i-1, -1, -1)):
            return

        # Get mains
        mains_for_building = self.mains[fold][activation.building]
        # load some additional data to make sure we have enough samples
        mains_end_extended = mains_end + timedelta(
            seconds=self.sample_period * 2)
        mains = mains_for_building[mains_start:mains_end_extended]
        seq.input = mains.values[:self.seq_length]
        return seq

    @classmethod
    def _attrs_to_remove_for_report(cls):
        return [
            'activations', 'rng', 'mains', 'mains_good_sections',
            'sections_with_no_target', 'all_gaps']
