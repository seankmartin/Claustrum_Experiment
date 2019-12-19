"""Assorted helper functions to split timestamp numpy arrays up."""

import numpy as np


def split_into_blocks(array, block_max, num_blocks):
    """
    Split a sorted array into num_blocks.

    Each block has a value of at most block_max
    greater than the previous block.

    Returns
    -------
    List:
        A list of size num_blocks

    """
    blocks = np.arange(
        block_max + 0.001, block_max * num_blocks + 0.001, block_max + 0.001)
    return np.array(np.split(array, np.searchsorted(array, blocks)))


def split_array(array, idxs):
    """
    Split a one d array into two arrays based on idxs.

    Returns
    -------
    (array[idxs], array[not idxs])

    """
    if len(idxs) == 0:
        return [], []
    left = array[idxs]
    ia = np.indices(array.shape)
    not_indices = np.setxor1d(ia, idxs)
    right = array[not_indices]

    return (left, right)


def split_array_with_another(array, split_arr):
    """
    Split a sorted one d array into two.

    First array is the values directly after splt_arr.

    Returns
    -------
    (array values after split_array, other values)

    """
    idxs = np.searchsorted(array, split_arr)
    idxs = idxs[idxs != len(array)]
    return split_array(array, idxs)


def split_array_in_between_two(array, left, right):
    """
    Split a sorted one d array into two.

    One array is the values in between left and right
    left and right should be the same size

    Returns
    -------
    (array values between left and right, other values)

    Raises
    ------
    ValueError
        If 'left' is not the same size as 'right'

    """
    if len(left) != len(right):
        raise(ValueError(
            'left and right must have the same number of elements'))
    good_idxs = []
    for i, val in enumerate(array):
        bigger = (left <= val)
        smaller = (right >= val)
        between = np.logical_and(smaller, bigger)
        if between.any():
            good_idxs.append(i)
    return split_array(array, good_idxs)


if __name__ == "__main__":
    array = np.array([i + 1 for i in range(9)])
    tests = []
    tests.append(split_into_blocks(array, 3, 3))
    tests.append(split_array(array, [0, 1, 2]))
    tests.append(split_array_with_another(array, [0.9, 3.9, 8.9]))
    tests.append(split_array_in_between_two(array, [1, 8], [4, 9]))
    for t in tests:
        print(t)