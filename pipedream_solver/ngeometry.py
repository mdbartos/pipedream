import numpy as np
import scipy.interpolate
from numba import njit
from numba.types import float64, int64, uint32, uint16, uint8, boolean, UniTuple, Tuple, List, DictType, void

geom_code = {
    'circular' : 1,
    'rect_closed' : 2,
    'rect_open' : 3,
    'triangular' : 4,
    'trapezoidal' : 5,
    'parabolic' : 6,
    'elliptical' : 7,
    'wide' : 8,
    'force_main' : 9,
    'floodplain' : 10
}

eps = np.finfo(float).eps

@njit(float64(float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
def Circular_B_ik(h_Ik, h_Ip1k, g1, g2):
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
    g2: float
        Width of Preissman slot (as a ratio of the diameter)
    """
    d = g1
    pslot = g2
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


@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64, float64),
      cache=True)
def Rect_Closed_B_ik(h_Ik, h_Ip1k, g1, g2, g3):
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
    g3: float
        Width of Preissman slot (as a ratio of the width)
    """
    y_max = g1
    b = g2
    pslot = g3
    y = (h_Ik + h_Ip1k) / 2
    if y < 0:
        y = 0
    cond = (y < y_max)
    if cond:
        B = b
    else:
        B = pslot * b
    return B


@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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


@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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


@njit(float64(float64, float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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


@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
def Force_Main_A_ik(h_Ik, h_Ip1k, g1, g2):
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
    g2: np.ndarray
        Width of Preissman slot (as a ratio of the diameter)
    """
    d = g1
    r = d / 2
    A = np.pi * r**2
    return A

@njit(float64(float64, float64, float64, float64),
      cache=True)
def Force_Main_Pe_ik(h_Ik, h_Ip1k, g1, g2):
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
    g2: np.ndarray
        Width of Preissman slot (as a ratio of the diameter)
    """
    d = g1
    Pe = np.pi * d
    return Pe

@njit(float64(float64, float64),
      cache=True)
def Force_Main_R_ik(A_ik, Pe_ik):
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

@njit(float64(float64, float64, float64, float64),
      cache=True)
def Force_Main_B_ik(h_Ik, h_Ip1k, g1, g2):
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
    g2: np.ndarray
        Width of Preissman slot (as a ratio of the diameter)
    """
    d = g1
    pslot = g2
    B = pslot * d
    return B

@njit(float64(float64, float64, float64, float64, float64, float64, float64, float64, float64),
      cache=True)
def Floodplain_A_ik(h_Ik, h_Ip1k, g1, g2, g3, g4, g5, g6, g7):
    """
    Compute cross-sectional area of flow for link i, superlink k.

    Inputs:
    -------
    h_Ik: np.ndarray
        Depth at upstream junction (meters)
    h_Ip1k: np.ndarray
        Depth at downstream junction (meters)
    h_Ik: np.ndarray
        Depth at upstream junction (meters)
    h_Ip1k: np.ndarray
        Depth at downstream junction (meters)
    g1: np.ndarray
        Height of channel (meters)
    g2: np.ndarray
        Height of lower floodplain section above channel bottom (meters)
    g3: np.ndarray
        Height of upper floodplain section above lower floodplain section (meters)
    g4: np.ndarray
        Inverse slope of upper channel sides (run/rise)
    g5: np.ndarray
        Inverse slope of lower channel sides (run/rise)
    g6: np.ndarray
        Inverse slope of middle channel sides (run/rise)
    g7: np.ndarray
        bottom width of channel (meters)    
    """
    h_max = g1
    h_low = g2
    h_mid = g2 + g3
    m_top = g4
    m_base = g5
    m_middle = g6
    y = (h_Ik + h_Ip1k) / 2
    if y < 0.:
        y = 0.
    if y > h_max:
        y = h_max
    if y < h_low:
        y_base = y
        y_middle = 0.
        y_top = 0.
        b_middle = 0.
        b_top = 0.
    elif (y >= h_low) and (y < h_mid):
        y_base = h_low
        y_middle = y - h_low
        y_top = 0.
        b_middle = 2 * m_base * h_low
        b_top = 0.
    elif (y >= h_mid):
        y_base = h_low
        y_middle = (h_mid - h_low)
        y_top = y - h_mid
        b_middle = 2 * m_base * h_low
        b_top = b_middle + 2 * m_middle * (h_mid - h_low)
    A_base = y_base * (m_base * y_base)
    A_middle = y_middle * (b_middle + m_middle * y_middle)
    A_top = y_top * (b_top + m_top * y_top)
    A = A_base + A_middle + A_top + g7*y*(y>0)
    return A


@njit(float64(float64, float64, float64, float64, float64, float64, float64, float64, float64),
      cache=True)
def Floodplain_Pe_ik(h_Ik, h_Ip1k, g1, g2, g3, g4, g5, g6, g7):
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
        Height of lower floodplain section above channel bottom (meters)
    g3: np.ndarray
        Height of upper floodplain section above lower floodplain section (meters)
    g4: np.ndarray
        Inverse slope of upper channel sides (run/rise)
    g5: np.ndarray
        Inverse slope of lower channel sides (run/rise)
    g6: np.ndarray
        Inverse slope of middle channel sides (run/rise)
    g7: np.ndarray
        bottom width of channel (meters)    
    """
    h_max = g1
    h_low = g2
    h_mid = g2 + g3
    m_top = g4
    m_base = g5
    m_middle = g6
    y = (h_Ik + h_Ip1k) / 2
    if y < 0.:
        y = 0.
    if y > h_max:
        y = h_max
    if y < h_low:
        y_base = y
        y_middle = 0.
        y_top = 0.
        b_middle = 0.
        b_top = 0.
    elif (y >= h_low) and (y < h_mid):
        y_base = h_low
        y_middle = y - h_low
        y_top = 0.
        b_middle = 2 * m_base * h_low
        b_top = 0.
    elif (y >= h_mid):
        y_base = h_low
        y_middle = (h_mid - h_low)
        y_top = y - h_mid
        b_middle = 2 * m_base * h_low
        b_top = 2 * m_middle * h_mid
    Pe_base = 2 * y_base * np.sqrt(1 + m_base**2)
    Pe_middle = 2 * y_middle * np.sqrt(1 + m_middle**2)
    Pe_top = 2 * y_top * np.sqrt(1 + m_top**2)
    Pe = Pe_base + Pe_middle + Pe_top + g7*(y>0)
    return Pe

@njit(float64(float64, float64),
      cache=True)
def Floodplain_R_ik(A_ik, Pe_ik):
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

@njit(float64(float64, float64, float64, float64, float64, float64, float64, float64, float64),
      cache=True)
def Floodplain_B_ik(h_Ik, h_Ip1k, g1, g2, g3, g4, g5, g6, g7):
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
        Height of lower floodplain section above channel bottom (meters)
    g3: np.ndarray
        Height of upper floodplain section above lower floodplain section (meters)
    g4: np.ndarray
        Inverse slope of upper channel sides (run/rise)
    g5: np.ndarray
        Inverse slope of lower channel sides (run/rise)
    g6: np.ndarray
        Inverse slope of middle channel sides (run/rise)
    g7: np.ndarray
        bottom width of channel (meters)            
    """
    h_max = g1
    h_low = g2
    h_mid = g2 + g3
    m_top = g4
    m_base = g5
    m_middle = g6
    y = (h_Ik + h_Ip1k) / 2
    if y < 0:
        y = 0.
    if y > h_max:
        y = h_max
    if y < h_low:
        B = 2 * m_base * y + g7*(y>0)
    elif (y >= h_low) and (y < h_mid):
        b_middle = 2 * m_base * h_low
        B = b_middle + (2 * m_middle * (y - h_low)) + g7*(y>0)
    elif (y >= h_mid):
        b_middle = 2 * m_base * h_low
        b_top = 2 * m_middle * (h_mid - h_low)
        B = b_middle + b_top + (2 * m_top * (y - h_mid)) + g7*(y>0)
    return B
