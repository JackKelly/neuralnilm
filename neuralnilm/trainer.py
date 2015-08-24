from __future__ import print_function, division
from functools import partial
from os.path import join
import numpy as np
import pandas as pd
import theano
from time import time
from lasagne.updates import nesterov_momentum
from lasagne.objectives import aggregate, squared_error
from lasagne.layers.helper import get_all_params
from .utils import sfloatX, ndim_tensor
from neuralnilm.data.datathread import DataThread
from neuralnilm.utils import ANSI, write_csv_row
from neuralnilm.plot.costplotter import CostPlotter
from neuralnilm.consts import DATA_FOLD_NAMES

import logging
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, net, data_pipeline, db, experiment_id,
                 loss_func=squared_error,
                 loss_aggregation_mode='mean',
                 updates_func=partial(nesterov_momentum, momentum=0.9),
                 learning_rate=1e-2,
                 callbacks=None,
                 repeat_callbacks=None,
                 display=None,
                 metrics=None):
        """
        Parameters
        ----------
        db : PyMongo database object for this experiment.
        callbacks : list of 2-tuples (<iteration>, <function>)
            Function must accept a single argument: this Trainer object.
        repeat_callbacks : list of 2-tuples (<iteration>, <function>)
            Function must accept a single argument: this Trainer object.
        display : neuralnilm.Display object
            For displaying and saving plots and data during training.
        metrics : neuralnilm.Metrics object
        """
        # Training and validation state
        self.db = db
        self.experiment_id = experiment_id
        self._train_func = None
        self.iteration = 0
        self.display = display
        self.metrics = metrics

        self.net = net
        self.data_pipeline = data_pipeline

        def _loss_func(prediction, target):
            loss = loss_func(prediction, target)
            return aggregate(loss, mode=loss_aggregation_mode)
        self.loss_func = _loss_func
        self.updates_func = updates_func
        # Set _learning_rate to -1 so when we set self.learning_rate
        # with the second line, Trainer logs the initial LR.
        self._learning_rate = theano.shared(sfloatX(-1), name='learning_rate')
        self.learning_rate = learning_rate

        # Callbacks
        def callbacks_dataframe(lst):
            return pd.DataFrame(lst, columns=['iteration', 'function'])
        self.callbacks = callbacks_dataframe(callbacks)
        self.repeat_callbacks = callbacks_dataframe(repeat_callbacks)

    @property
    def learning_rate(self):
        return self._learning_rate.get_value()

    @learning_rate.setter
    def learning_rate(self, rate):
        rate = sfloatX(rate)
        if rate == self.learning_rate:
            logger.info(
                "Iteration {:d}: Attempted to change learning rate to {:.1E}"
                " but that is already the value!"
                .format(self.iteration, rate))
        else:
            logger.info(
                "Iteration {:d}: Change learning rate to {:.1E}"
                .format(self.iteration, rate))
            self.db.experiments.find_one_and_update(
                filter={'_id': self.experiment_id},
                update={'$set':
                        {'trainer.learning_rates.{:d}'.format(self.iteration):
                         float(rate)}},
                upsert=True
            )
            self._learning_rate.set_value(rate)

    def fit(self, num_iterations=None):
        logger.info(
            "Starting training for {} iterations.".format(num_iterations))
        self.data_thread = DataThread(self.data_pipeline)
        self.data_thread.start()
        if self.display is not None:
            self.display.start()
        self.time = time()
        while self.iteration != num_iterations:
            try:
                self._run_train_iteration()
            except Exception as exception:
                logger.exception(exception)
                break
            self.iteration += 1
        self.data_thread.stop()
        if self.display is not None:
            self.display.stop()
        logger.info("Stopped training. Completed {} iterations."
                    .format(self.iteration))

    def _run_train_iteration(self):
        # Training
        batch = self.data_thread.get_batch()
        train_cost = self._get_train_func()(
            batch.after_processing.input,
            batch.after_processing.target)
        train_cost = float(train_cost.flatten()[0])

        # Save training costs
        score = {
            'experiment_id': self.experiment_id,
            'iteration': self.iteration,
            'loss': train_cost
        }

        if 'source_id' in batch.metadata:
            score['source_id'] = batch.metadata['source_id']
        self.db.train_scores.insert_one(score)

        if np.isnan(train_cost):
            msg = "training cost is NaN at iteration {}!".format(
                self.iteration)
            logger.error(msg)
            raise TrainingError(msg)

        # Callbacks
        def run_callbacks(df):
            for callback in df['function']:
                callback(self)
        repeat_callbacks = self.repeat_callbacks[
            (self.iteration % self.repeat_callbacks['iteration']) == 0]
        run_callbacks(repeat_callbacks)
        callbacks = self.callbacks[
            self.callbacks['iteration'] == self.iteration]
        run_callbacks(callbacks)

    def validate(self):
        sources = self.data_thread.data_pipeline.sources
        output_func = self.net.deterministic_output_func
        all_scores = {
            'experiment_id': self.experiment_id, 'iteration': self.iteration}
        for source_id, source in enumerate(sources):
            scores_for_source = {}
            for fold in DATA_FOLD_NAMES:
                batch = self.data_thread.data_pipeline.get_batch(
                    fold=fold, source_id=source_id)
                output = output_func(batch.after_processing.input)
                metrics = self.metrics.compute_metrics(
                    output, batch.after_processing.target)
                scores_for_source[fold] = metrics
            all_scores[batch.metadata['source_name']] = scores_for_source
        self.db.validation_scores.insert_one(all_scores)

    def _get_train_func(self):
        if self._train_func is None:
            self._train_func = self._compile_cost_func(validation=False)
        return self._train_func

    def _get_validation_cost_func(self):
        if self._validation_cost_func is None:
            self._validation_cost_func = self._compile_cost_func(
                validation=True)
        return self._validation_cost_func

    def _compile_cost_func(self, validation):
        logger.info("Compiling " + ("validation" if validation else "train") +
                    " cost function...")
        network_input = self.net.symbolic_input()
        network_output = self.net.symbolic_output(deterministic=validation)
        target_var = ndim_tensor(name='target', ndim=network_output.ndim)
        loss = self.loss_func(network_output, target_var)
        if validation:
            updates = None
        else:
            # Training
            all_params = get_all_params(self.net.layers[-1], trainable=True)
            updates = self.updates_func(
                loss, all_params, learning_rate=self._learning_rate)
        train_func = theano.function(
            inputs=[network_input, target_var],
            outputs=loss,
            updates=updates,
            on_unused_input='warn',
            allow_input_downcast=True)
        logger.info("Done compiling cost function.")
        return train_func


class TrainingError(Exception):
    pass
