from __future__ import print_function, division
from lasagne.layers import (LSTMLayer, RecurrentLayer, ElemwiseSumLayer,
                            ConcatLayer)


def BLSTMLayer(*args, **kwargs):
    """Configures forward and backwards LSTM layers to create a
    bidirectional LSTM.
    """
    return BidirectionalLayer(LSTMLayer, *args, **kwargs)


def BidirectionalRecurrentLayer(*args, **kwargs):
    """Configures forward and backwards RecurrentLayers to create a
    bidirectional recurrent layer."""
    return BidirectionalLayer(RecurrentLayer, *args, **kwargs)


def BidirectionalLayer(layer_class, *args, **kwargs):
    """
    Parameters
    ----------
    layer_class : e.g. lasagne.layers.LSTMLayer
    merge_mode : {'sum', 'concatenate'}
    """
    kwargs.pop('backwards', None)
    merge_mode = kwargs.pop('merge_mode', 'sum')
    l_fwd = layer_class(*args, backwards=False, **kwargs)
    l_bck = layer_class(*args, backwards=True, **kwargs)
    layers = [l_fwd, l_bck]
    if merge_mode == 'sum':
        return ElemwiseSumLayer(layers)
    elif merge_mode == 'concatenate':
        return ConcatLayer(layers, axis=2)
    else:
        raise ValueError("'{}' not a valid merge_mode".format(merge_mode))
