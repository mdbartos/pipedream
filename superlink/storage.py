import numpy as np
import scipy.interpolate

class Functional():
    def __init__(self):
        pass

    @classmethod
    def A_sj(self, h, a=1, b=1, c=0, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.
        """
        h[h < 0] = 0
        A = a * h**b + c
        return A

class Tabular():
    def __init__(self, h, A):
        self.h = np.asarray(h)
        self.A = np.asarray(A)
        self.hmax = self.h.max()
        self.hmin = self.h.min()
        self.interpolator = scipy.interpolate.interp1d(self.h, self.A)

    def A_sj(self, h):
        """
        Compute cross-sectional area of flow for link i, superlink k.
        """
        hmax = self.hmax
        hmin = self.hmin
        _h = np.copy(h)
        _h[_h < hmin] = hmin
        _h[_h > hmax] = hmax
        A = self.interpolator(_h)
        return A

