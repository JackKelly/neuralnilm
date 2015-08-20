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

import logging
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, net, data_pipeline,
                 path='.',
                 loss_func=squared_error,
                 loss_aggregation_mode='mean',
                 updates_func=partial(nesterov_momentum, momentum=0.9),
                 learning_rate=1e-2,
                 callbacks=None,
                 repeat_callbacks=None):
        self.net = net
        self.data_pipeline = data_pipeline
        self.validation_batch = self.data_pipeline.get_batch(validation=True)
        self.path = path

        def _loss_func(prediction, target):
            loss = loss_func(prediction, target)
            return aggregate(loss, mode=loss_aggregation_mode)
        self.loss_func = _loss_func
        self.updates_func = updates_func
        self._learning_rate = theano.shared(
            sfloatX(learning_rate), name='learning_rate')

        # Callbacks
        def callbacks_dataframe(lst):
            return pd.DataFrame(lst, columns=['iteration', 'function'])
        self.callbacks = callbacks_dataframe(callbacks)
        self.repeat_callbacks = callbacks_dataframe(repeat_callbacks)

        # Training and validation state
        self._train_func = None
        self._validation_cost_func = None
        self.training_costs = []
        self.validation_costs = []
        self.iteration = 0

    @property
    def learning_rate(self):
        return self._learning_rate.get_value()

    @learning_rate.setter
    def learning_rate(self, rate):
        rate = sfloatX(rate)
        self.logger.info(
            "Iteration {:d}: Change learning rate to {:.1E}"
            .format(self.iteration, rate))
        self._learning_rate.set_value(rate)

    def fit(self, num_iterations=None):
        logger.info(
            "Starting training for {} iterations.".format(num_iterations))
        self._print_costs_header()
        self.data_thread = DataThread(self.data_pipeline)
        self.data_thread.start()
        self.cost_plotter = CostPlotter(path=self.path)
        self.cost_plotter.start()
        self.time = time()
        while self.iteration != num_iterations:
            self.iteration = len(self.training_costs)
            try:
                self._run_train_iteration()
            except Exception as exception:
                logger.exception(exception)
                break
        self.data_thread.stop()
        self.cost_plotter.stop()
        logger.info("Stopped training. Completed {} iterations."
                    .format(self.iteration))

    def _run_train_iteration(self):
        # Training
        self.train_batch = self.data_thread.get_batch()
        train_cost = self._get_train_func()(
            self.train_batch.after_processing.input,
            self.train_batch.after_processing.target)
        train_cost = train_cost.flatten()[0]
        self.training_costs.append(train_cost)
        write_csv_row(join(self.path, 'training_costs.csv'),
                      ["{:.6E}".format(train_cost)])
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

        # Print costs to screen
        self._print_costs()

    def plot_costs(self):
        self.cost_plotter.plot()

    def _print_costs_header(self):
        print("""
 Update |  Train cost  |  Valid cost  |  Train / Val  | Secs per update
--------|--------------|--------------|---------------|----------------\
""")

    def _print_costs(self):
        time_now = time()
        duration = time_now - self.time
        train_cost = self.training_costs[-1]
        validation_cost = (
            self.validation_costs[-1] if self.validation_costs else np.NaN)

        def is_best(costs, cost):
            if costs:
                best_cost = min(costs)
                is_best = cost == best_cost
            else:
                is_best = True
            return is_best

        is_best_train = is_best(self.training_costs, train_cost)
        is_best_valid = is_best(self.validation_costs, validation_cost)

        # print bests to screen
        print("  {:>5} |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  |"
              "  {:>11.6f}  |  {:>.3f}s".format(
                  self.iteration,
                  ANSI.BLUE if is_best_train else "",
                  train_cost,
                  ANSI.ENDC if is_best_train else "",
                  ANSI.GREEN if is_best_valid else "",
                  validation_cost,
                  ANSI.ENDC if is_best_valid else "",
                  train_cost / validation_cost,
                  duration
              ))
        self.time = time_now

    def validate(self):
        validation_cost = self._get_validation_cost_func()(
            self.validation_batch.after_processing.input,
            self.validation_batch.after_processing.target)
        validation_cost = validation_cost.flatten()[0]
        self.validation_costs.append(validation_cost)
        write_csv_row(join(self.path, 'validation_costs.csv'),
                      ["{:.6E}".format(validation_cost)])

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
