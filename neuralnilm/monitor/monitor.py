from __future__ import print_function, division
from time import sleep
import pymongo
from monary import Monary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralnilm.consts import DATA_FOLD_NAMES


class Monitor(object):
    def __init__(self, experiment_id, output_path='.',
                 update_period=1, max_num_lines=1000,
                 mongo_db='neuralnilm'):
        """
        Parameters
        ----------
        max_num_lines : int
            Number of pixels.
        """
        self.experiment_id = experiment_id
        self.output_path = output_path
        self.update_period = update_period
        self.max_num_lines = max_num_lines
        self._last_iteration_processed = {'train': 0, 'validation': 0}
        self.mongo_client = pymongo.MongoClient()
        self.db = self.mongo_client[mongo_db]
        self.mongo_db = mongo_db
        self._validation_metric_names = None

    def start(self):
        while True:
            if self._new_scores_available('train'):
                self._plot_train_scores()
            if self._new_scores_available('validation'):
                self._plot_validation_scores()
            sleep(self.update_period)

    def _new_costs_available(self, train_or_validation):
        """Returns True if new training costs are available from DB.

        Parameters
        ----------
        train_or_validation : str, {'train', 'validation'}
        """
        collection = self.db[train_or_validation + '_scores']
        document = collection.find_one(
            filter={
                'experiment_id': self.experiment_id,
                'iteration': {
                    '$gt': self._last_iteration_processed[train_or_validation]}
            }
        )
        return bool(document)

    def _get_validation_mse(self):
        monary = Monary()

        def get_mse_for_fold(fold):
            iterations, loss, source_id = monary.query(
                db=self.mongo_db,
                coll='validation_scores',
                query={'experiment_id': self.experiment_id, 'fold': fold},
                fields=['iteration', 'scores.regression.mean_squared_error',
                        'source_id'],
                types=['int32', 'float32', 'int8']
            )

            scores_df = pd.DataFrame(
                {'loss': loss, 'source_id': source_id}, index=iterations)

            return scores_df

        FOLDS = ['unseen_appliances', 'unseen_activations_of_seen_appliances']
        scores = {}
        for fold in FOLDS:
            scores[fold] = get_mse_for_fold(fold)

        return scores

    def _get_train_costs(self):
        # Get train scores
        monary = Monary()
        iterations, loss, source_id = monary.query(
            db=self.mongo_db,
            coll='train_scores',
            query={'experiment_id': self.experiment_id},
            fields=['iteration', 'loss', 'source_id'],
            types=['int32', 'float32', 'int8']
        )

        scores_df = pd.DataFrame(
            {'loss': loss, 'source_id': source_id}, index=iterations)

        return scores_df

    def _plot_train_scores(self):
        train_scores_df = self._get_train_costs()
        all_scores = self._get_validation_mse()
        all_scores.update({'train': train_scores_df})

        fig, ax = plt.subplots(1)
        for fold, scores_df in all_scores.iteritems():
            sources = scores_df['source_id'].unique()
            for source_i in sources:
                # Get losses for just this source
                mask = scores_df['source_id'] == source_i
                loss = scores_df[mask]['loss']

                # Downsample if necessary
                loss_for_source = self._downsample(loss)

                # Plot
                ax.plot(loss_for_source.index, loss_for_source.values,
                        label='{:s} (source {:d})'.format(fold, source_i))

        ax.legend()
        plt.title('Training costs')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean squared error')
        plt.show()
        try:
            self._last_iteration_processed['train'] = train_scores_df.index[-1]
        except IndexError:
            # No data loaded
            pass

    @property
    def validation_metric_names(self):
        """
        Returns
        -------
        metric_names : list
            e.g. ['regression.mean_squared_error',
                  'classification_2_state.f1_score']
        """
        if self._validation_metric_names is None:
            scores = self.db.validation_scores.find_one(
                filter={'experiment_id': self.experiment_id})['scores']
            self._validation_metric_names = []
            for metric_type, metrics in scores.iteritems():
                for metric_name in metrics:
                    self._validation_metric_names.append(
                        metric_type + '.' + metric_name)
        return self._validation_metric_names

    def _plot_validation_scores(self):
        validation_sources = self.db.validation_scores.distinct(
            key='source_id', filter={'experiment_id': self.experiment_id})
        num_cols = len(validation_sources)
        fig, axes = plt.subplots(
            nrows=3, ncols=num_cols, sharex="col", sharey="row",
            squeeze=False)
        # Make some space on the right side for the extra y-axes.
        fig.subplots_adjust(right=0.65)
        for col, source_id in enumerate(validation_sources):
            for row, fold in enumerate(DATA_FOLD_NAMES):
                ax = axes[row, col]
                self._plot_validation_scores_for_source_and_fold(
                    ax=ax, source_id=source_id, fold=fold,
                    show_axes_labels=(row == 0),
                    show_scales=(col == num_cols-1))
                if row == 0:
                    title = "source_id = {:d}\n{}".format(source_id, fold)
                else:
                    title = fold
                ax.set_title(title)
                if row == 2:
                    ax.set_xlabel('Iteration')
        plt.show()

    def _plot_validation_scores_for_source_and_fold(self, ax, source_id, fold,
                                                    show_axes_labels,
                                                    show_scales):
        fields = ['iteration'] + ['scores.' + metric_name for metric_name in
                                  self.validation_metric_names]
        monary = Monary()
        result = monary.query(
            db=self.mongo_db,
            coll='validation_scores',
            query={
                'experiment_id': self.experiment_id,
                'source_id': source_id,
                'fold': fold
            },
            fields=fields,
            types=['int32'] + ['float32'] * len(self.validation_metric_names)
        )

        index = result[0]
        data = {metric_name: result[i+1] for i, metric_name in
                enumerate(self.validation_metric_names)}
        df = pd.DataFrame(data, index=index)
        df = self._downsample(df)

        # Create multiple independent axes.  Adapted from Joe Kington's answer:
        # http://stackoverflow.com/a/7734614

        # Colors
        cmap = plt.get_cmap('gnuplot')
        n = len(self.validation_metric_names)
        colors = [cmap(i) for i in np.linspace(0, 1, n)]

        # Twin the x-axis to make independent y-axes.
        axes = [ax]
        for metric_name in self.validation_metric_names[1:]:
            axes.append(ax.twinx())

        if show_scales:
            for i, axis in enumerate(axes[2:]):
                # To make the border of the right-most axis visible, we need to
                # turn the frame on. This hides the other plots, however, so we
                # need to turn its fill off.
                axis.set_frame_on(True)
                axis.patch.set_visible(False)

                # Move the last y-axes spines over to the right by 20% of the
                # width of the axes
                axis.spines['right'].set_position(('axes', 1.1 + (0.1 * i)))
        else:
            for axis in axes:
                axis.tick_params(labelright=False)
                for spine in ['top', 'right']:
                    axis.spines[spine].set_visible(False)

        lines = []
        for i, (axis, metric_name, color) in enumerate(
                zip(axes, self.validation_metric_names, colors)):
            axis.tick_params(axis='y', colors=color, direction='out')
            label = metric_name.replace("regression.", "")
            label = label.replace("classification_", "")
            label = label.replace("_", " ")
            label = label.replace(".", " ")
            label = label.replace(" ", "\n")
            line, = axis.plot(
                df.index, df[metric_name].values, color=color, label=label)
            if show_axes_labels and show_scales:
                axis.set_ylabel(label, color=color, rotation=0, fontsize=8)
                if i == 0:
                    coords = (0.0, 1.1)
                else:
                    coords = (0.95 + (0.1 * i), 1.3)
                axis.yaxis.set_label_coords(*coords)
            lines.append(line)

        self._last_iteration_processed['validation'] = index[-1]
        return lines

    def _downsample(self, data):
        """Downsample `data` if necessary."""
        if len(data) > self.max_num_lines:
            divisor = int(np.ceil(len(data) / self.max_num_lines))
            data = data.groupby(lambda x: x // divisor).mean()
            data.index *= divisor
        return data
