import PIL
import ot
import numpy as np
from skimage.segmentation import slic  # super pixels
from scipy.sparse import csr_matrix  # sparse matrices for graph weights
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

def pos_mat_projection(P, hist):
    """
    Projection of a coupling matrix P on the set of positive matrices whose rows' sum are given by hist
    """

    def projection(x, exp_sum):
        """Projection of a vector x on the set of postive vectors whose coefficient sum to exp_sum"""
        n = len(x)
        sorted_x = np.sort(x)[::-1]  # sort in descending order
        cumsum = np.cumsum(sorted_x)
        # compute the projection onto the diagonal matrix
        proj = (cumsum - exp_sum) / np.arange(1, n+1)
        i = np.argmax(proj >= sorted_x) - 1
        proj = proj[i] if i >= 0 else (np.sum(x) - exp_sum) / n
        return np.maximum(x - proj, 0)


    projection = np.vectorize(projection, signature='(n),()->(n)')
    proj = projection(P, hist)

    return proj

