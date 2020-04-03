import numpy as np
import scipy.interpolate

# Allowed geometries and their numeric codes
geom_code = {
    'circular' : 1,
    'rect_closed' : 2,
    'rect_open' : 3,
    'triangular' : 4,
    'trapezoidal' : 5,
    'parabolic' : 6,
    'elliptical' : 7,
    'wide' : 8
}

# Machine precision
eps = np.finfo(float).eps

class Circular():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Diameter of channel (meters)
        """
        d = g1
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > d] = d[y > d]
        r = d / 2
        phi = y / r
        phi[phi < 0] = 0
        phi[phi > 2] = 2
        theta = np.arccos(1 - phi)
        A = r**2 * (theta - np.cos(theta) * np.sin(theta))
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Diameter of channel (meters)
        """
        d = g1
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > d] = d[y > d]
        r = d / 2
        phi = y / r
        phi[phi < 0] = 0
        phi[phi > 2] = 2
        theta = np.arccos(1 - phi)
        Pe = 2 * r * theta
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, pslot=0.001, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Diameter of channel (meters)
        pslot: float
            Width of Preissman slot (as a ratio of the diameter)
        """
        d = g1
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        r = d / 2
        phi = y / r
        phi[phi < 0] = 0
        phi[phi > 2] = 2
        theta = np.arccos(1 - phi)
        cond = (y < d)
        B = np.zeros(y.size)
        B[~cond] = pslot * d[~cond]
        B[cond] = 2 * r[cond] * np.sin(theta[cond])
        return B

class Rect_Closed():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = y * b
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        Pe = b + 2 * y
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, g2, pslot=0.001, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        pslot: float
            Width of Preissman slot (as a ratio of the width)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        cond = (y < y_max)
        B = np.zeros(y.size)
        B[~cond] = pslot * b[~cond]
        B[cond] = b[cond]
        return B

class Rect_Open():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = y * b
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        Pe = b + 2 * y
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        return b


class Triangular():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Inverse slope of channel sides (run/rise)
        """
        y_max = g1
        m = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = m * y**2
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Inverse slope of channel sides (run/rise)
        """
        y_max = g1
        m = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        Pe = 2 * y * np.sqrt(1 + m**2)
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Inverse slope of channel sides (run/rise)
        """
        y_max = g1
        m = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        cond = (y < y_max)
        B = np.zeros(y.size)
        B[~cond] = 2 * m[~cond] * y_max[~cond]
        B[cond] = 2 * m[cond] * y[cond]
        return B

class Trapezoidal():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, g2, g3, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        g3: np.ndarray
            Inverse slope of channel sides (run/rise)
        """
        y_max = g1
        b = g2
        m = g3
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = y * (b + m * y)
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, g2, g3, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        g3: np.ndarray
            Inverse slope of channel sides (run/rise)
        """
        y_max = g1
        b = g2
        m = g3
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        Pe = b + 2 * y * np.sqrt(1 + m**2)
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, g2, g3, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        g3: np.ndarray
            Inverse slope of channel sides (run/rise)
        """
        y_max = g1
        b = g2
        m = g3
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        cond = (y < y_max)
        B = np.zeros(y.size)
        # B[~cond] = 0.001 * b[~cond]
        B[~cond] = b[~cond] + 2 * m[~cond] * y_max[~cond]
        B[cond] = b[cond] + 2 * m[cond] * y[cond]
        return B

class Parabolic():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = 2 * b * y / 3
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y <= 0] = eps
        y[y > y_max] = y_max[y > y_max]
        x = 4 * y / b
        Pe = (b / 2) * (np.sqrt(1 + x**2) + (1 / x) * np.log(x + np.sqrt(1 + x**2)))
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        B = b * np.sqrt(y / y_max)
        return B

class Elliptical():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Full height of channel (meters)
        g2: np.ndarray
            Full width of channel (meters)
        """
        y_max = g1
        b = g1 / 2
        a = g2 / 2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        theta = np.arcsin((y - b) / b)
        A_a = a * b * (np.pi / 2 + theta)
        A_b = a * b * np.cos(theta) * np.sin(theta)
        A = A_a + A_b
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Full height of channel (meters)
        g2: np.ndarray
            Full width of channel (meters)
        """
        y_max = g1
        b = g1 / 2
        a = g2 / 2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        theta = np.arcsin((y - b) / b)
        m = (1 - a**2 / b**2)
        Pe_a = 2 * b * scipy.special.ellipe(m)
        Pe_b = 2 * b * scipy.special.ellipeinc(theta, m)
        Pe = Pe_a + Pe_b
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Full height of channel (meters)
        g2: np.ndarray
            Full width of channel (meters)
        """
        y_max = g1
        b = g1 / 2
        a = g2 / 2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        theta = np.arcsin((y - b) / b)
        B = 2 * np.cos(theta) * np.sqrt(a**2 * np.cos(theta)**2
                                        + b**2 * np.sin(theta)**2)
        return B


class Wide():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute cross-sectional area of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = y * b
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute perimeter of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        Pe = b
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.

        Inputs:
        -------
        A_ik: np.ndarray
            Area of cross section (square meters)
        Pe_ik: np.ndarray
            Wetted perimeter of cross section (meters)
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, g1, g2, **kwargs):
        """
        Compute top width of flow for link i, superlink k.

        Inputs:
        -------
        h_Ik: np.ndarray
            Depth at upstream junction (meters)
        h_Ip1k: np.ndarray
            Depth at downstream junction (meters)
        g1: np.ndarray
            Height of channel (meters)
        g2: np.ndarray
            Width of channel (meters)
        """
        y_max = g1
        b = g2
        return b

class Irregular():
    def __init__(self, x, y, horiz_points=100, vert_points=100):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.horiz_points = horiz_points
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.interpolator = scipy.interpolate.interp1d(self.x, self.y)
        self.xx, self.xstep = np.linspace(self.xmin, self.xmax,
                                          horiz_points, retstep=True)
        self.yy = self.interpolator(self.xx)
        self.h = np.linspace(self.ymin, self.ymax, vert_points)
        # Generate lookup tables
        self._A_ik = self.A_ik_lut(self.h)
        self._Pe_ik = self.Pe_ik_lut(self.h)
        self._R_ik = self.R_ik_lut(self._A_ik, self._Pe_ik)
        self._B_ik = self.B_ik_lut(self.h)
        # Create interpolators
        self.A_interpolator = scipy.interpolate.interp1d(self.h, self._A_ik)
        self.Pe_interpolator = scipy.interpolate.interp1d(self.h, self._Pe_ik)
        self.R_interpolator = scipy.interpolate.interp1d(self.h, self._R_ik)
        self.B_interpolator = scipy.interpolate.interp1d(self.h, self._B_ik)

    def A_ik(self, h_Ik, h_Ip1k):
        ymax = self.ymax
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > ymax] = ymax
        A = self.A_interpolator(y)
        return A

    def Pe_ik(self, h_Ik, h_Ip1k):
        ymax = self.ymax
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > ymax] = ymax
        Pe = self.Pe_interpolator(y)
        return Pe

    def R_ik(self, h_Ik, h_Ip1k):
        ymax = self.ymax
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > ymax] = ymax
        R = self.R_interpolator(y)
        return R

    def B_ik(self, h_Ik, h_Ip1k):
        ymax = self.ymax
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > ymax] = ymax
        B = self.B_interpolator(y)
        return B

    def A_ik_lut(self, h):
        """
        Compute cross-sectional area of flow for link i, superlink k.
        """
        xx = self.xx
        yy = self.yy
        if np.isscalar(h):
            hh = np.maximum(h - yy, 0)
        else:
            n = yy.shape[0]
            m = h.size
            h = np.repeat(h, n).reshape(m, n)
            hh = np.maximum(h - yy, 0)
        A = np.trapz(hh, xx)
        return A

    def Pe_ik_lut(self, h):
        """
        Compute perimeter of flow for link i, superlink k.
        """
        xx = self.xx
        yy = self.yy
        xstep = self.xstep
        if np.isscalar(h):
            hh = np.maximum(h - yy, 0)
            dh = np.gradient(hh)
        else:
            n = yy.shape[0]
            m = h.size
            h = np.repeat(h, n).reshape(m, n)
            hh = np.maximum(h - yy, 0)
            dh = np.gradient(hh)[-1]
        dx = np.zeros(dh.shape)
        dx[hh > 0] = xstep
        Pe = np.sqrt(dx**2 + dh**2).sum(axis=-1)
        return Pe

    def R_ik_lut(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    def B_ik_lut(self, h):
        """
        Compute top width of flow for link i, superlink k.
        """
        xx = self.xx
        yy = self.yy
        xstep = self.xstep
        if np.isscalar(h):
            hh = h > yy
        else:
            n = yy.shape[0]
            m = h.size
            h = np.repeat(h, n).reshape(m, n)
            hh = h >= yy
        dx = np.zeros(hh.shape)
        dx[hh] = xstep
        B = dx.sum(axis=1)
        return B
