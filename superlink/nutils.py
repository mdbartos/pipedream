import numpy as np

@njit
def interpolate_sample(x, xp, fp):
    n = len(xp)
    ix = np.searchsorted(xp, x)
    if (ix == 0):
        result = np.zeros(n)
    elif (ix >= n):
        result = np.zeros(n)
    else:
        dx_0 = x - xp[ix - 1]
        dx_1 = xp[ix] - x
        frac = dx_0 / (dx_0 + dx_1)
        result = (1 - frac) * fp[ix - 1] + (frac) * fp[ix]
    return result
