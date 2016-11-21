import numpy as np
import multiprocessing.sharedctypes
import warnings

def convert_numpy_arr_to_shared_mem_arr(X):
    """ Get copy of X accessible as shared memory

    Returns
    --------
    Xsh : RawArray (same size as X)
        Uses separate storage than original array X.
    """
    Xtmp = np.ctypeslib.as_ctypes(X)
    Xsh = multiprocessing.sharedctypes.RawArray(Xtmp._type_, Xtmp)
    return Xsh


def convert_shared_mem_arr_to_numpy_arr(Xsh):
    """ Get view (not copy) of shared memory as numpy array.

    Returns
    -------
    X : ND numpy array (same size as X)
        Any changes to X will also influence data stored in Xsh.
    """
    if isinstance(Xsh, int) or isinstance(Xsh, float):
        return Xsh
    elif isinstance(Xsh, np.ndarray):
        return Xsh
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.ctypeslib.as_array(Xsh)

def make_shared_mem_dict_from_numpy_dict(param_dict):
    ''' Create dict of shared memory arrays from provided dict of numpy arrays

    Returns
    -------
    shared_mem_dict : dict with same keys as param_dict
    '''
    shared_mem_dict = dict()
    for key in param_dict:
        arr = param_dict[key]
        if isinstance(arr, np.ndarray):
            shared_mem_dict[key] = \
                convert_numpy_arr_to_shared_mem_arr(arr)
        else:
            assert isinstance(arr, int) or isinstance(arr, float)
            shared_mem_dict[key] = arr
    return shared_mem_dict

def convert_shared_mem_dict_to_numpy_dict(shared_mem_dict):
    ''' Make views (not copies) of all shared-mem arrays in provided dict

    Returns
    -------
    param_dict : dict
    '''
    param_dict = dict()
    if shared_mem_dict is None:
        return param_dict

    for key, arr in shared_mem_dict.items():
        param_dict[key] = convert_shared_mem_arr_to_numpy_arr(arr)
    return param_dict


def update_shared_mem_arr_inplace(Xsh, Xarr):
    ''' Copy all data from a numpy array into provided shared memory

    Post Condition
    --------------
    Xsh updated in place.
    '''
    Xsh_arr = sharedMemToNumpyArray(Xsh)
    K = Xarr.shape[0]
    assert Xsh_arr.shape[0] >= K
    Xsh_arr[:K] = Xarr
