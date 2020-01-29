import numpy as np
import scipy.interpolate
from numba import njit

geom_code = {
    'circular' : 1,
    'rect_closed' : 2,
    'rect_open' : 3,
    'triangular' : 4,
    'trapezoidal' : 5
}

@njit
def Circular_A_ik(h_Ik, h_Ip1k, g1):
    """
    Compute cross-sectional area of flow for link i, superlink k.
    """
    d = g1
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > d] = d[y > d]
    if y < 0:
        y = 0
    if y > d:
        y = d
    r = d / 2
    phi = y / r
    # phi[phi < 0] = 0
    # phi[phi > 2] = 2
    if phi < 0:
        phi = 0
    if phi > 2:
        phi = 2
    theta = np.arccos(1 - phi)
    A = r**2 * (theta - np.cos(theta) * np.sin(theta))
    return A

@njit
def Circular_Pe_ik(h_Ik, h_Ip1k, g1):
    """
    Compute perimeter of flow for link i, superlink k.
    """
    d = g1
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > d] = d[y > d]
    if y < 0:
        y = 0
    if y > d:
        y = d
    r = d / 2
    phi = y / r
    # phi[phi < 0] = 0
    # phi[phi > 2] = 2
    if phi < 0:
        phi = 0
    if phi > 2:
        phi = 2
    theta = np.arccos(1 - phi)
    Pe = 2 * r * theta
    return Pe

@njit
def Circular_R_ik(A_ik, Pe_ik):
    """
    Compute hydraulic radius for link i, superlink k.
    """
    cond = Pe_ik > 0
    # R = np.zeros(A_ik.size)
    # R[cond] = A_ik[cond] / Pe_ik[cond]
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Circular_B_ik(h_Ik, h_Ip1k, g1, pslot=0.001):
    """
    Compute top width of flow for link i, superlink k.
    """
    d = g1
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    if y < 0:
        y = 0
    r = d / 2
    phi = y / r
    # phi[phi < 0] = 0
    # phi[phi > 2] = 2
    if phi < 0:
        phi = 0
    if phi > 2:
        phi = 2
    theta = np.arccos(1 - phi)
    # B = np.zeros(y.size)
    # B[~cond] = pslot * d[~cond]
    # B[cond] = 2 * r[cond] * np.sin(theta[cond])
    cond = (y < d)
    if cond:
        B = 2 * r * np.sin(theta)
    else:
        B = pslot * d
    return B


@njit
def Rect_Closed_A_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute cross-sectional area of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    A = y * b
    return A

@njit
def Rect_Closed_Pe_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute perimeter of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    Pe = b + 2 * y
    return Pe

@njit
def Rect_Closed_R_ik(A_ik, Pe_ik):
    """
    Compute hydraulic radius for link i, superlink k.
    """
    cond = Pe_ik > 0
    # R = np.zeros(A_ik.size)
    # R[cond] = A_ik[cond] / Pe_ik[cond]
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Rect_Closed_B_ik(h_Ik, h_Ip1k, g1, g2, pslot=0.001):
    """
    Compute top width of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    if y < 0:
        y = 0
    cond = (y < y_max)
    # B = np.zeros(y.size)
    # B[~cond] = pslot * b[~cond]
    # B[cond] = b[cond]
    if cond:
        B = b
    else:
        B = pslot * b
    return B


@njit
def Rect_Open_A_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute cross-sectional area of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    A = y * b
    return A

@njit
def Rect_Open_Pe_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute perimeter of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    Pe = b + 2 * y
    return Pe

@njit
def Rect_Open_R_ik(A_ik, Pe_ik):
    """
    Compute hydraulic radius for link i, superlink k.
    """
    cond = Pe_ik > 0
    # R = np.zeros(A_ik.size)
    # R[cond] = A_ik[cond] / Pe_ik[cond]
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Rect_Open_B_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute top width of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    return b


@njit
def Triangular_A_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute cross-sectional area of flow for link i, superlink k.
    """
    y_max = g1
    m = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    A = m * y**2
    return A

@njit
def Triangular_Pe_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute perimeter of flow for link i, superlink k.
    """
    y_max = g1
    m = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    Pe = 2 * y * np.sqrt(1 + m**2)
    return Pe

@njit
def Triangular_R_ik(A_ik, Pe_ik):
    """
    Compute hydraulic radius for link i, superlink k.
    """
    cond = Pe_ik > 0
    # R = np.zeros(A_ik.size)
    # R[cond] = A_ik[cond] / Pe_ik[cond]
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Triangular_B_ik(h_Ik, h_Ip1k, g1, g2):
    """
    Compute top width of flow for link i, superlink k.
    """
    y_max = g1
    m = g2
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    if y < 0:
        y = 0
    cond = (y < y_max)
    # B = np.zeros(y.size)
    # # B[~cond] = 0.001 * 2 * m[~cond] * y[~cond]
    # B[~cond] = 2 * m[~cond] * y_max[~cond]
    # B[cond] = 2 * m[cond] * y[cond]
    if cond:
        B = 2 * m * y
    else:
        B = 2 * m * y_max
    return B


@njit
def Trapezoidal_A_ik(h_Ik, h_Ip1k, g1, g2, g3):
    """
    Compute cross-sectional area of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    m = g3
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    A = y * (b + m * y)
    return A

@njit
def Trapezoidal_Pe_ik(h_Ik, h_Ip1k, g1, g2, g3):
    """
    Compute perimeter of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    m = g3
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    # y[y > y_max] = y_max[y > y_max]
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    Pe = b + 2 * y * np.sqrt(1 + m**2)
    return Pe

@njit
def Trapezoidal_R_ik(A_ik, Pe_ik):
    """
    Compute hydraulic radius for link i, superlink k.
    """
    cond = Pe_ik > 0
    # R = np.zeros(A_ik.size)
    # R[cond] = A_ik[cond] / Pe_ik[cond]
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Trapezoidal_B_ik(h_Ik, h_Ip1k, g1, g2, g3):
    """
    Compute top width of flow for link i, superlink k.
    """
    y_max = g1
    b = g2
    m = g3
    y = (h_Ik + h_Ip1k) / 2
    # y[y < 0] = 0
    if y < 0:
        y = 0
    cond = (y < y_max)
    # B = np.zeros(y.size)
    # # B[~cond] = 0.001 * b[~cond]
    # B[~cond] = b[~cond] + 2 * m[~cond] * y_max[~cond]
    # B[cond] = b[cond] + 2 * m[cond] * y[cond]
    if cond:
        B = b + 2 * m * y
    else:
        B = b + 2 * m * y_max
    return B

