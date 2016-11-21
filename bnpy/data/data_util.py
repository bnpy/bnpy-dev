import numpy as np

from XData import XData

def fetch_batch(
        data,
        batch_id=0,
        n_batches=10,
        **kwargs):
    ''' Fetch specific batch of dataset

    Returns
    -------
    batch_dataset : dataset object

    Examples
    --------
    >>> X = np.arange(8)[:,np.newaxis]
    >>> fetch_batch(X, batch_id=0, n_batches=2).flatten()
    array([0, 1, 2, 3])
    >>> fetch_batch(X, batch_id=1, n_batches=2).flatten()
    array([4, 5, 6, 7])

    # Try 5 batches
    >>> fetch_batch(X, batch_id=0, n_batches=5).flatten()
    array([0, 1])
    >>> fetch_batch(X, batch_id=4, n_batches=5).flatten()
    array([7])

    '''
    if isinstance(data, np.ndarray):
        list_of_slices = make_list_of_slices_for_dataset(data, n_batches)
        batch_slice = list_of_slices[batch_id]
        return data[batch_slice]
    elif hasattr(data, '__getitem__'):
        return data[batch_id]


def make_list_of_slices_for_dataset(
        data,
        n_parallel_workers=1,
        **kwargs):
    ''' Make list of slice intervals to divide dataset among parallel workers

    Returns
    -------
    list_of_slices : list of tuples

    Examples
    --------
    >>> make_list_of_slices_for_dataset(30, 3)
    [slice(0, 10, None), slice(10, 20, None), slice(20, 30, None)]
    >>> make_list_of_slices_for_dataset(7, 3)
    [slice(0, 3, None), slice(3, 5, None), slice(5, 7, None)]
    >>> make_list_of_slices_for_dataset(27, 1)
    [slice(0, 27, None)]
    '''
    if isinstance(data, int):
        n_examples = data
    elif isinstance(data, np.ndarray):
        n_examples = data.shape[0]
    elif hasattr(data, 'nDoc'):
        n_examples = data.nDoc
    else:
        n_examples = data.nObs
    start = 0
    n_examples_per_slice = int(np.floor(n_examples // n_parallel_workers))
    n_leftover = n_examples - n_parallel_workers * n_examples_per_slice
    list_of_slices = []
    start = 0
    for i in range(n_parallel_workers):
        stop = start + n_examples_per_slice
        if i < n_leftover:
            stop += 1
        list_of_slices.append(slice(start, stop))
        start = stop
    return list_of_slices

def convert_dataset_to_shared_mem_dict(
        data,
        **kwargs):
    ''' Convert provided dataset into proper shared memory dictionary

    Returns
    -------
    shared_mem_dict : dict
    make_dataset_from_shared_mem_dict : function

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> X = prng.randn(100, 2)
    >>> shared_dict, make_dataset_func = convert_dataset_to_shared_mem_dict(X)
    >>> Xcopy = make_dataset_func(shared_dict).X
    >>> np.allclose(X, Xcopy)
    True

    # Verify we can access a slice from the shared memory
    >>> slice_interval = slice(17, 40)
    >>> X_sliced = make_dataset_func(shared_dict, slice_interval).X
    >>> np.allclose(X_sliced, X[slice_interval])
    True
    '''
    if isinstance(data, np.ndarray):
        dataset = XData(X=data)
    assert isinstance(dataset, XData)

    shared_mem_dict = dataset.to_shared_mem_dict()
    make_dataset_from_shared_mem_dict = dataset.\
        get_function_to_make_dataset_from_shared_mem_dict()
    return shared_mem_dict, make_dataset_from_shared_mem_dict