from __future__ import print_function, division
import numpy as np


def stride(data, num_seq_per_batch, seq_length, stride=None, pad=True):
    """Distribute 1D array `data` across one or more batches.

    Optionally specify a `stride` to control overlap.

    Parameters
    ----------
    data : 1D np.ndarray
    num_seq_per_batch : int
    seq_length : int
    stride : int or None, optional
        If None then `stride` will be set to `seq_length`.
    pad : bool
        If True then pad `data` with `seq_length` values of `min(data)`
        on each side.

    Returns
    -------
    batches : list of 3D arrays
    """
    assert data.ndim == 1
    if stride is None:
        stride = seq_length

    # need to convert to float64 to avoid this Numpy bug
    # in Numpy version <= 1.9.2:
    # https://github.com/numpy/numpy/issues/6026
    minimum = data.min().astype(np.float64)
    if pad:
        data = np.pad(
            data, pad_width=seq_length, mode='constant',
            constant_values=(minimum, minimum))
    num_samples = len(data)
    input_shape = (num_seq_per_batch, seq_length, 1)

    # Divide data data into batches
    num_batches = np.ceil(
        (num_samples / stride) / num_seq_per_batch).astype(int)

    batches = []
    for batch_i in xrange(num_batches):
        batch = np.zeros(input_shape, dtype=np.float32)
        batch_start = batch_i * num_seq_per_batch * stride
        for seq_i in xrange(num_seq_per_batch):
            mains_start_i = batch_start + (seq_i * stride)
            mains_end_i = mains_start_i + seq_length
            seq = data[mains_start_i:mains_end_i]
            seq = np.pad(
                seq, pad_width=(0, seq_length-len(seq)), mode='constant',
                constant_values=(minimum, minimum))
            batch[seq_i, :len(seq), 0] = seq
        batches.append(batch)

    return batches
