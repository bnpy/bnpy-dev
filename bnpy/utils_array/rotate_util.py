import numpy as np

def rotate_2_by_2_array(a_22, angle_rad=np.pi / 4):
    ''' Rotate provided 2x2 array by given angle (in radians)

    This rotation preserves the underlying eigen structure.

    Parameters
    ----------
    a_22 : 2D array, size 2x2
        symmetric and postiive definite
    angle_rad : float
        specifies rotation angle in radians

    Returns
    -----
    rotated_22 : 2D array, size 2x2
        symmetric and positive definite

    Examples
    --------
    >>> np.set_printoptions(precision=3, suppress=1)
    >>> rotate_2_by_2_array(np.diag([2, 1.]), 0.0)
    array([[ 2.,  0.],
           [ 0.,  1.]])
    >>> rotate_2_by_2_array(np.diag([2, 1]), np.pi / 2)
    array([[ 1.,  0.],
           [ 0.,  2.]])
    >>> rotate_2_by_2_array(np.diag([2, 1]), np.pi / 4)
    array([[ 1.5,  0.5],
           [ 0.5,  1.5]])
    '''
    rotation_22 = [
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]]
    rotation_22 = np.asarray(rotation_22)

    eigval_2, eigvec_22 = np.linalg.eig(a_22)
    diag_eigval_22 = np.diag(eigval_2)
    rotated_eigvec_22 = np.dot(eigvec_22, rotation_22)
    return np.dot(
        rotated_eigvec_22,
        np.dot(diag_eigval_22, rotated_eigvec_22.T))
