from matplotlib import pylab

import numpy as np

default_colors = [(1, 0, 0),
          (1, 0, 1),
          (0, 1, 0),
          (0, 1, 1),
          (0, 0, 1),
          (1, 0.6, 0),
          (1, 0, 0.5),
          (0.5, 0.8, 0.8)]


def plot_gauss_2D_from_common_params(
        proba_K=None,
        mean_KD=None,
        covar_KDD=None,
        proba_thr=0.0001,
        ax_handle=None,
        color_list=default_colors,
        color='k',
        data=None,
        **kwargs):
    ''' Plot 2D contours for components in hmodel in current pylab figure
    '''
    if ax_handle is None:
        ax_handle = pylab.gca()

    if data is not None and hasattr(data, 'X'):
        ax_handle.plot(
            data.X[:, 0], data.X[:, 1], '.', 
            color=(.3,.3,.3),
            alpha=0.5)

    K, D = mean_KD.shape
    nSkip = 0
    nGood = 0
    for ii, k in enumerate(range(K)):
        mean_D = mean_KD[k]
        covar_DD = covar_KDD[k]
        if color_list is not None:
            cur_color = color_list[k % len(color_list)]
        else:
            cur_color = color
        plot_gauss_2D_contour(
            mean_D, covar_DD, color=cur_color, ax_handle=ax_handle)
    ax_handle.set_aspect('equal', 'datalim')

def plot_gauss_2D_contour(
        mean_D=None,
        covar_DD=None,
        color='k',
        ax_handle=None,
        prob_contour_levels=None,
        radius_contour_levels=[1.0, 3.0],
        markersize=3,
        ):
    ''' Plot elliptical contours for provided mean and covariance arrays

    Uses only the first 2 dimensions.

    Post Condition
    --------------
    Plot created on current axes
    '''
    if ax_handle is None:
        ax_handle = pylab.gca()

    mean_2 = np.asarray(mean_D)[:2]
    covar_22 = np.asarray(covar_DD)[:2, :2]

    eigval_2, eigvec_22 = np.linalg.eig(covar_22)
    chol_covar_22 = np.dot(eigvec_22, np.sqrt(np.diag(eigval_2)))

    # Prep for plotting elliptical contours
    # by creating grid of (x,y) points along perfect circle
    ts = np.arange(-np.pi, np.pi, 0.03)
    x = np.sin(ts)
    y = np.cos(ts)
    Z_2M = np.vstack([x, y])

    # Warp circle into ellipse defined by eigenvectors
    P_2M = np.dot(chol_covar_22, Z_2M)

    # plot contour lines across several radius lengths
    for r in radius_contour_levels:
        P_r_2M = r * P_2M + mean_2[:, np.newaxis]
        ax_handle.plot(
            P_r_2M[0], P_r_2M[1], '.',
            markersize=markersize,
            markerfacecolor=color, markeredgecolor=color)
