import numpy as np

def uid2k(cur_uids_K=None, uid=0):
    ''' Convert specific uid to current index value k

    Returns
    -------
    k : int

    Examples
    --------
    >>> uid2k([40, 30, 20, 10], uid=20)
    2
    >>> uid2k([40, 20, 30, 10, 50, 70, 60], uid=50)
    4
    '''
    return np.flatnonzero(np.asarray(cur_uids_K) == uid)[0]

def uidpair2kpair(cur_uids_K=None, uidA=0, uidB=1):
    ''' Convert specific uids to current index value k

    Returns
    -------
    kA : int
    kB : int
        Guarantees that kB > kA

    Examples
    --------
    >>> uidpair2kpair([40, 30, 20, 10], uidA=20, uidB=30)
    (1, 2)
    >>> uidpair2kpair([40, 20, 30, 10, 50, 70, 60], uidA=50, uidB=60)
    (4, 6)
    '''
    kA = uid2k(cur_uids_K, uidA)
    kB = uid2k(cur_uids_K, uidB)
    return np.minimum(kA, kB), np.maximum(kA, kB)