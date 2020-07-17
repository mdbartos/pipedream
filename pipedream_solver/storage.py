import numpy as np
import scipy.interpolate
import scipy.integrate

class Functional():
    def __init__(self):
        pass

    @classmethod
    def A_sj(self, h, a=1, b=1, c=0, **kwargs):
        """
        Compute surface area for superjunction j using functional relation:
        A_sj = a * h**b + c

        Inputs:
        -------
        h : np.ndarray
            Depth of water at superjunction
        a : np.ndarray or float
            Coefficient parameter
        b : np.ndarray or float
            Exponent parameter
        c : np.ndarray or float
            Constant parameter
        """
        h[h < 0] = 0
        A = a * h**b + c
        return A

    @classmethod
    def V_sj(self, h, a=1, b=1, c=0, **kwargs):
        """
        Compute surface area for superjunction j using functional relation:
        V_sj = a * h**(b + 1) / (b + 1) + c * h

        Inputs:
        -------
        h : np.ndarray
            Depth of water at superjunction
        a : np.ndarray or float
            Coefficient parameter
        b : np.ndarray or float
            Exponent parameter
        c : np.ndarray or float
            Constant parameter
        """
        h[h < 0] = 0
        V = a * h**(b + 1) / (b + 1) + c * h
        return V

class Tabular():
    """
    Class for computing tabular area/volume-depth relations at superjunctions.

    Inputs:
    -------
    h : np.ndarray
        Depth points on depth-area profile (meters)
    A : np.ndarray
        Surface areas associated with each depth (square meters)
    """
    def __init__(self, h, A):
        self.h = np.asarray(h)
        self.A = np.asarray(A)
        self.V = scipy.integrate.cumtrapz(h, A, initial=0.)
        self.hmax = self.h.max()
        self.hmin = self.h.min()
        self.Amax = self.A.max()
        self.A_interpolator = scipy.interpolate.interp1d(self.h, self.A)
        self.V_interpolator = scipy.interpolate.interp1d(self.h, self.V)

    def A_sj(self, h):
        """
        Compute surface area for superjunction j

        Inputs:
        -------
        h : np.ndarray
            Depth of water at superjunction
        """
        hmax = self.hmax
        hmin = self.hmin
        _h = np.copy(h)
        _h[_h < hmin] = hmin
        _h[_h > hmax] = hmax
        A = self.A_interpolator(_h)
        return A

    def V_sj(self, h):
        """
        Compute surface area of flow for superjunction j

        Inputs:
        -------
        h : np.ndarray
            Depth of water at superjunction
        """
        hmax = self.hmax
        hmin = self.hmin
        Amax = self.Amax
        _h = np.copy(h)
        _h[_h < hmin] = hmin
        _h[_h > hmax] = hmax
        r = np.maximum(h - _h, 0)
        V = self.V_interpolator(_h) + (r * Amax)
        return V

