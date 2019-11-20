import numpy as np


def split_into_blocks(array, block_max, num_blocks):
    """
    Splits a sorted array into num_blocks
    where each block has a value of at most block_max
    greater than the previous block.
    """
    blocks = np.arange(
        block_max + 0.001, block_max * num_blocks + 0.001, block_max + 0.001)
    return np.split(array, np.searchsorted(array, blocks))


def split_array(array, idxs):
    """
    Split a one d array into two arrays based on idxs.

    Returns
    -------
    (array[idxs], array[not idxs])

    """
    left = array[idxs]
    ia = np.indices(array.shape)
    not_indices = np.setxor1d(ia, idxs)
    right = array[not_indices]

    return (left, right)


def split_array_with_another(array, split_arr):
    """
    Split a sorted one darray into two with 
    one array being the values directly after splt_arr.

    Returns
    -------
    (array values after split_array, other values)

    """
    idxs = np.searchsorted(array, split_arr)
    return split_array(array, idxs)


if __name__ == "__main__":
    array = np.array([i + 1 for i in range(9)])
    tests = []
    tests.append(split_into_blocks(array, 3, 3))
    tests.append(split_array(array, [0, 1, 2]))
    tests.append(split_array_with_another(array, [0.9, 3.9, 8.9]))
    for t in tests:
        print(t)
