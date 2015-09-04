from __future__ import print_function, division
from copy import copy
import theano
import h5py
from lasagne.layers.helper import get_all_layers, get_output, get_all_params
from lasagne.layers import InputLayer
from neuralnilm.utils import none_to_list

import logging
logger = logging.getLogger(__name__)


VALID_TAGS = ['AE', 'RNN', 'Conv1D', 'tied weights', 'rectangles']


class Net(object):
    """
    Attributes
    ----------
    layers : list
    train_iterations : int
    description : string
    tags : list of strings
       from controlled vocab (see VALID_TAGS)
    predecessor_experiment : string
    """
    def __init__(self, output_layer, description="", tags=None,
                 predecessor_experiment=""):
        self.layers = get_all_layers(output_layer)
        self._deterministic_output_func = None
        self.train_iterations = 0
        self.description = description
        self.tags = none_to_list(tags)
        for tag in self.tags:
            if tag not in VALID_TAGS:
                raise ValueError("{} is not a valid tag!".format(tag))
        self.predecessor_experiment = predecessor_experiment

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

    def save_params(self, filename):
        """
        Save params to HDF in the following format:
            /epoch<N>/L<I>_<type>/P<I>_<name>
        """
        mode = 'w' if self.train_iterations == 0 else 'a'
        f = h5py.File(filename, mode=mode)
        epoch_name = 'epoch{:06d}'.format(self.train_iterations)
        try:
            epoch_group = f.create_group(epoch_name)
        except ValueError:
            logger.exception("Cannot save params!")
            f.close()
            return

        for layer_i, layer in enumerate(self.layers):
            params = layer.get_params()
            if not params:
                continue
            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            layer_group = epoch_group.create_group(layer_name)
            for param_i, param in enumerate(params):
                param_name = 'P{:02d}'.format(param_i)
                if param.name:
                    param_name += "_" + param.name
                data = param.get_value()
                layer_group.create_dataset(
                    param_name, data=data, compression="gzip")

        f.close()

    def load_params(self, filename, iteration):
        """
        Load params from HDF in the following format:
            /epoch<N>/L<I>_<type>/P<I>_<name>
        """
        # Process function parameters
        logger.info('Loading params from ' + filename + '...')

        f = h5py.File(filename, mode='r')
        epoch_name = 'epoch{:06d}'.format(iteration)
        epoch_group = f[epoch_name]

        for layer_i, layer in enumerate(self.layers):
            params = layer.get_params()
            if not params:
                continue
            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            layer_group = epoch_group[layer_name]
            for param_i, param in enumerate(params):
                param_name = 'P{:02d}'.format(param_i)
                if param.name:
                    param_name += "_" + param.name
                data = layer_group[param_name]
                param.set_value(data.value)
        f.close()
        self.train_iterations = iteration
        logger.info('Done loading params from ' + filename + '.')

    def num_trainable_parameters(self):
        return sum(
            [p.get_value().size for p in get_all_params(self.layers[-1])])

    def description_of_architecture(self):
        layers = []
        for layer in self.layers:
            layer_dict = {
                'type': layer.__class__.__name__,
                'output_shape': layer.output_shape
            }
            for attr in ['num_units', 'concat_axis', 'input_layers']:
                try:
                    value = getattr(layer, attr)
                except AttributeError:
                    pass
                else:
                    layer_dict[attr] = value
            try:
                value = layer.nonlinearity
            except AttributeError:
                pass
            else:
                layer_dict['nonlinearity'] = value.__name__
            layers.append(layer_dict)

        return {
            'layers': layers,
            'num_trainable_parameters': self.num_trainable_parameters()
        }

    def report(self):
        report = copy(self.__dict__)
        for attr in [
                'layers', '_deterministic_output_func', 'train_iterations']:
            report.pop(attr)
        report.setdefault('architecture', {})[0] = (
            self.description_of_architecture())
        return {'net': report}


def build_net(input_shape, layers):
    # Input layer
    layer = InputLayer(shape=input_shape)
    for layer_config in layers:
        layer_type = layer_config.pop('type')
        layer = layer_type(layer, **layer_config)

    return layer
