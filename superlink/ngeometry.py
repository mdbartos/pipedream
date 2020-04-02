import numpy as np
import scipy.interpolate
from numba import njit

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

eps = np.finfo(float).eps

@njit
def Circular_A_ik(h_Ik, h_Ip1k, g1):
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
    if y < 0:
        y = 0
    if y > d:
        y = d
    r = d / 2
    phi = y / r
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
    if y < 0:
        y = 0
    if y > d:
        y = d
    r = d / 2
    phi = y / r
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

    Inputs:
    -------
    A_ik: np.ndarray
        Area of cross section (square meters)
    Pe_ik: np.ndarray
        Wetted perimeter of cross section (meters)
    """
    cond = Pe_ik > 0
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Circular_B_ik(h_Ik, h_Ip1k, g1, pslot=0.001):
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
    # y[y < 0] = 0
    if y < 0:
        y = 0
    r = d / 2
    phi = y / r
    if phi < 0:
        phi = 0
    if phi > 2:
        phi = 2
    theta = np.arccos(1 - phi)
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

    Inputs:
    -------
    A_ik: np.ndarray
        Area of cross section (square meters)
    Pe_ik: np.ndarray
        Wetted perimeter of cross section (meters)
    """
    cond = Pe_ik > 0
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Rect_Closed_B_ik(h_Ik, h_Ip1k, g1, g2, pslot=0.001):
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
    if y < 0:
        y = 0
    cond = (y < y_max)
    if cond:
        B = b
    else:
        B = pslot * b
    return B


@njit
def Rect_Open_A_ik(h_Ik, h_Ip1k, g1, g2):
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

    Inputs:
    -------
    A_ik: np.ndarray
        Area of cross section (square meters)
    Pe_ik: np.ndarray
        Wetted perimeter of cross section (meters)
    """
    cond = Pe_ik > 0
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Rect_Open_B_ik(h_Ik, h_Ip1k, g1, g2):
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


@njit
def Triangular_A_ik(h_Ik, h_Ip1k, g1, g2):
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

    Inputs:
    -------
    A_ik: np.ndarray
        Area of cross section (square meters)
    Pe_ik: np.ndarray
        Wetted perimeter of cross section (meters)
    """
    cond = Pe_ik > 0
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Triangular_B_ik(h_Ik, h_Ip1k, g1, g2):
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
    if y < 0:
        y = 0
    cond = (y < y_max)
    if cond:
        B = 2 * m * y
    else:
        B = 2 * m * y_max
    return B


@njit
def Trapezoidal_A_ik(h_Ik, h_Ip1k, g1, g2, g3):
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

    Inputs:
    -------
    A_ik: np.ndarray
        Area of cross section (square meters)
    Pe_ik: np.ndarray
        Wetted perimeter of cross section (meters)
    """
    cond = Pe_ik > 0
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Trapezoidal_B_ik(h_Ik, h_Ip1k, g1, g2, g3):
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
    if y < 0:
        y = 0
    cond = (y < y_max)
    if cond:
        B = b + 2 * m * y
    else:
        B = b + 2 * m * y_max
    return B

@njit
def Parabolic_A_ik(h_Ik, h_Ip1k, g1, g2):
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
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    A = 2 * b * y / 3
    return A

@njit
def Parabolic_Pe_ik(h_Ik, h_Ip1k, g1, g2):
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
    if y <= 0:
        y = eps
    if y > y_max:
        y = y_max
    x = 4 * y / b
    Pe = (b / 2) * (np.sqrt(1 + x**2) + (1 / x) * np.log(x + np.sqrt(1 + x**2)))
    return Pe

@njit
def Parabolic_R_ik(A_ik, Pe_ik):
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
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Parabolic_B_ik(h_Ik, h_Ip1k, g1, g2):
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
    if y < 0:
        y = 0
    B = b * np.sqrt(y / y_max)
    return B

@njit
def Elliptical_A_ik(h_Ik, h_Ip1k, g1, g2):
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
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    theta = np.arcsin((y - b) / b)
    A_a = a * b * (np.pi / 2 + theta)
    A_b = a * b * np.cos(theta) * np.sin(theta)
    A = A_a + A_b
    return A

@njit
def Elliptical_R_ik(A_ik, Pe_ik):
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
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Elliptical_B_ik(h_Ik, h_Ip1k, g1, g2):
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
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    theta = np.arcsin((y - b) / b)
    B = 2 * np.cos(theta) * np.sqrt(a**2 * np.cos(theta)**2
                                    + b**2 * np.sin(theta)**2)
    return B


@njit
def Wide_A_ik(h_Ik, h_Ip1k, g1, g2):
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
    if y < 0:
        y = 0
    if y > y_max:
        y = y_max
    A = y * b
    return A

@njit
def Wide_Pe_ik(h_Ik, h_Ip1k, g1, g2):
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
    b = g2
    Pe = b
    return Pe

@njit
def Wide_R_ik(A_ik, Pe_ik):
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
    if cond:
        R = A_ik / Pe_ik
    else:
        R = 0
    return R

@njit
def Wide_B_ik(h_Ik, h_Ip1k, g1, g2):
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

