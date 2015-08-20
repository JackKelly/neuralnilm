from __future__ import print_function, division
from functools import partial
import pandas as pd
import theano
from lasagne.updates import nesterov_momentum
from lasagne.objectives import aggregate, squared_error
from lasagne.layers.helper import get_all_params
from .utils import sfloatX, ndim_tensor
from neuralnilm.data.datathread import DataThread

import logging
logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, net, data_pipeline,
                 loss_func=squared_error,
                 loss_aggregation_mode='mean',
                 updates_func=partial(nesterov_momentum, momentum=0.9),
                 learning_rate=1e-2,
                 callbacks=None,
                 repeat_callbacks=None):
        self.net = net
        self.data_pipeline = data_pipeline
        self.validation_batch = self.data_pipeline.get_batch() #validation=True)

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
    def train_func(self):
        if self._train_func is None:
            self._train_func = self._compile_train_func()
        return self._train_func

    @property
    def validation_cost_func(self):
        if self._validation_cost_func is None:
            self._validation_cost_func = self._compile_validation_cost_func()
        return self._validation_cost_func

    def _compile_train_func(self):
        logger.info("Compiling train function...")
        network_input = self.net.symbolic_input()
        train_output = self.net.symbolic_output(deterministic=False)
        target_var = ndim_tensor(name='target', ndim=train_output.ndim)
        train_loss = self.loss_func(train_output, target_var)
        all_params = get_all_params(self.net.layers[-1], trainable=True)
        updates = self.updates_func(
            train_loss, all_params, learning_rate=self._learning_rate)
        train_func = theano.function(
            inputs=[network_input, target_var],
            outputs=train_loss,
            updates=updates,
            on_unused_input='warn',
            allow_input_downcast=True)
        logger.info("Done compiling train function.")
        return train_func

    def _compile_validation_cost_func(self):
        logger.info("Compiling validation cost function...")
        network_input = self.net.symbolic_input()
        deterministic_output = self.net.symbolic_output(deterministic=True)
        target_var = ndim_tensor(name='target', ndim=deterministic_output.ndim)
        validation_loss = self.loss_func(deterministic_output, target_var)
        validation_cost_func = theano.function(
            inputs=[network_input, target_var],
            outputs=[validation_loss, deterministic_output],
            on_unused_input='warn',
            allow_input_downcast=True)
        logger.info("Done compiling validation cost function.")
        return validation_cost_func

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
        self.data_thread = DataThread(self.data_pipeline)
        self.data_thread.start()
        while self.iteration != num_iterations:
            self.iteration = len(self.training_costs)
            try:
                self._run_train_iteration()
            except Exception as exception:
                logger.exception(exception)
                break
        self.data_thread.stop()

    def _run_train_iteration(self):
        # Training
        self.train_batch = self.data_thread.get_batch()
        train_cost = self.train_func(
            self.train_batch.after_processing.input,
            self.train_batch.after_processing.target)
        self.training_costs.append(train_cost.flatten()[0])

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
        pass
