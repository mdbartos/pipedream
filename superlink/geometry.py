import numpy as np
import scipy.interpolate

class Circular():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, d):
        """
        Compute cross-sectional area of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > d] = d[y > d]
        r = d / 2
        theta = np.arccos(1 - y / r)
        A = r**2 * (theta - np.cos(theta) * np.sin(theta))
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, d):
        """
        Compute perimeter of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > d] = d[y > d]
        r = d / 2
        theta = np.arccos(1 - y / r)
        Pe = 2 * r * theta
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, d):
        """
        Compute top width of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        r = d / 2
        theta = np.arccos(1 - y / r)
        cond = (y < d)
        B = np.zeros(y.size)
        B[~cond] = 0.001 * d[~cond]
        B[cond] = 2 * r[cond] * np.sin(theta[cond])
        return B

class Rect_Closed():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, b, y_max):
        """
        Compute cross-sectional area of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = y * b
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, b, y_max):
        """
        Compute perimeter of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        Pe = b + 2 * y
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, b, y_max):
        """
        Compute top width of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        cond = (y < y_max)
        B = np.zeros(y.size)
        B[~cond] = 0.001 * b[~cond]
        B[cond] = b[cond]
        return B

class Triangular():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, m, y_max):
        """
        Compute cross-sectional area of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = m * y**2
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, m, y_max):
        """
        Compute perimeter of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        Pe = 2 * y * np.sqrt(1 + m**2)
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, b, m, y_max):
        """
        Compute top width of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        cond = (y < y_max)
        B = np.zeros(y.size)
        B[~cond] = 0.001 * b[~cond]
        B[cond] = 2 * m[cond] * y[cond]
        return B

class Trapezoidal():
    def __init__(self):
        pass

    @classmethod
    def A_ik(self, h_Ik, h_Ip1k, b, m, y_max):
        """
        Compute cross-sectional area of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        A = y * (b + m * y)
        return A

    @classmethod
    def Pe_ik(self, h_Ik, h_Ip1k, b, m, y_max):
        """
        Compute perimeter of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > y_max] = y_max[y > y_max]
        Pe = b + 2 * y * np.sqrt(1 + m**2)
        return Pe

    @classmethod
    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    @classmethod
    def B_ik(self, h_Ik, h_Ip1k, b, m, y_max):
        """
        Compute top width of flow for link i, superlink k.
        """
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        cond = (y < y_max)
        B = np.zeros(y.size)
        B[~cond] = 0.001 * b[~cond]
        B[cond] = b[cond] + 2 * m[cond] * y[cond]
        return B

class Irregular():
    def __init__(self, x, y, n=100):
        self.n = n
        self.x = x
        self.y = y
        ymin = y.min()
        ymax = y.max()
        xmin = x.min()
        xmax = x.max()
        self.interpolator = scipy.interpolate.interp1d(x, y)
        self.xx, self.xstep = np.linspace(xmin, xmax, retstep=True)
        self.yy = interpolator(xx)

    def A_ik(h):
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

    def Pe_ik(h):
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

    def R_ik(self, A_ik, Pe_ik):
        """
        Compute hydraulic radius for link i, superlink k.
        """
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    def B_ik(h):
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
