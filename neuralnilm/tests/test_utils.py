from __future__ import print_function, division
import numpy as np


def test_downsample():
    from neuralnilm.utils import downsample
    arr = np.array([
        0.13636507,  0.19106028, -0.30038809, -0.87103187, -0.99128436,
        -0.64778967,  0.9848212, -0.26735439,  1.43959721,  1.90148751])
    downsampled_2x = downsample(arr, 2)
    for i in range(arr.size // 2):
        start_i = i * 2
        end_i = start_i + 2
        assert downsampled_2x[i] == arr[start_i:end_i].mean()

    downsampled_3x = downsample(arr, 3)
    for i in range(arr.size // 3):
        start_i = i * 3
        end_i = start_i + 3
        assert downsampled_3x[i] == arr[start_i:end_i].mean()
