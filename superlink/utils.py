import numpy as np

def interpolate_sample(x, xp, fp):
    n = xp.shape[0]
    m = fp.shape[1]
    ix = np.searchsorted(xp, x)
    if (ix == 0):
        result = np.zeros(m)
    elif (ix >= n):
        result = np.zeros(m)
    else:
        dx_0 = x - xp[ix - 1]
        dx_1 = xp[ix] - x
        frac = dx_0 / (dx_0 + dx_1)
        result = (1 - frac) * fp[ix - 1] + (frac) * fp[ix]
    return result
