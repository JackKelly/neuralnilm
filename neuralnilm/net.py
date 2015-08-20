from __future__ import print_function, division
import theano
from lasagne.layers.helper import get_all_layers, get_output

import logging
logger = logging.getLogger(__name__)


class Net(object):
    def __init__(self, output_layer):
        self.layers = get_all_layers(output_layer)
        self._deterministic_output_func = None

    @property
    def deterministic_output_func(self):
        if self._deterministic_output_func is None:
            self._deterministic_output_func = (
                self._compile_deterministic_output_func())
        return self._deterministic_output_func

    def _compile_deterministic_output_func(self):
        logger.info("Compiling deterministic output function...")
        network_input = self.symbolic_input()
        deterministic_output = self.symbolic_output(deterministic=True)
        net_output_func = theano.function(
            inputs=[network_input],
            outputs=deterministic_output,
            on_unused_input='warn',
            allow_input_downcast=True
        )
        logger.info("Done compiling deterministic output function.")
        return net_output_func

    def symbolic_input(self):
        network_input = self.layers[0].input_var
        return network_input

    def symbolic_output(self, deterministic=False):
        network_input = self.symbolic_input()
        network_output = get_output(
            self.layers[-1], network_input, deterministic=deterministic)
        return network_output
