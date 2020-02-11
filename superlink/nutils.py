import numpy as np
from numba import njit

@njit
def interpolate_sample(x, xp, fp):
    n = xp.shape[0]
    m = fp.shape[1]
    ix = np.searchsorted(xp, x)
    if (ix == 0):
        result = fp[0]
    elif (ix >= n):
        result = fp[n - 1]
    else:
        dx_0 = x - xp[ix - 1]
        dx_1 = xp[ix] - x
        frac = dx_0 / (dx_0 + dx_1)
        result = (1 - frac) * fp[ix - 1] + (frac) * fp[ix]
    return result
