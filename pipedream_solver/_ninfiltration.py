import numpy as np
from numba import njit
from pipedream_solver.nutils import newton_raphson, bounded_newton_raphson, numba_any

@njit
def run_green_ampt_newton(F_2, x0, F_1, dt, Ks, theta_d, psi_f, ia, max_iter=50,
                          atol=1.48e-8, rtol=0.0, bounded=True):
    """
    Use Newton-Raphson iteration to find cumulative infiltration at next time step (F_2).

    Inputs:
    -------
    F_2 : np.ndarray (float)
        Cumulative infiltration at next time step (meters).
    x0 : np.ndarray (float)
        Initial guess for cumulative infiltration (meters)
    F_1 : np.ndarray (float)
        Cumulative infiltration at previous time step (meters)
    dt : np.ndarray (float)
        Time step (seconds)
    Ks : np.ndarray (float)
        Saturated hydraulic conductivity (m/s)
    theta_d : np.ndarray (float)
        Soil moisture deficit (-)
    psi_f : np.ndarray (float)
        Matric potential of the wetting front (m)
    ia : np.ndarray (float)
        Available rainfall depth (meters)
    max_iter : int
        Maximum number of Newton-Raphson iterations
    atol : float
        Allowable (absolute) error of the zero value
    rtol : float
        Allowable (relative) error of the zero value
    bounded : bool
        If True, use bounded Newton-Raphson iteration
    """
    n = F_2.size
    for i in range(n):
        x_0_i = x0[i]
        F_1_i = F_1[i]
        dt_i = dt[i]
        nargs = np.zeros(5)
        nargs[0] = F_1_i
        nargs[1] = dt_i
        nargs[2] = Ks[i]
        nargs[3] = theta_d[i]
        nargs[4] = psi_f[i]
        if bounded:
            min_F = 0
            max_F = F_1_i + ia[i] * dt_i
            F_est = bounded_newton_raphson(numba_integrated_green_ampt,
                                        numba_derivative_green_ampt,
                                        x_0_i, min_F, max_F,
                                        nargs, max_iter=max_iter,
                                        atol=atol, rtol=rtol)
        else:
            F_est = newton_raphson(numba_integrated_green_ampt,
                                   numba_derivative_green_ampt,
                                   x_0_i, nargs, max_iter=max_iter,
                                   atol=atol, rtol=rtol)
        F_2[i] = F_est
    return F_2

@njit
def numba_integrated_green_ampt(F_2, args):
    """
    Solve integrated form of Green Ampt equation for cumulative infiltration.

    Inputs:
    -------
    F_2: np.ndarray (float)
        Cumulative infiltration at current timestep (m)
    F_1: np.ndarray (float)
        Cumulative infiltration at next timestep (m)
    dt: float
        Time step (seconds)
    Ks: np.ndarray (float)
        Saturated hydraulic conductivity (m/s)
    theta_d: np.ndarray (float)
        Soil moisture deficit
    psi_s: np.ndarray (float)
        Soil suction head (m)
    """
    F_1 = args[0]
    dt = args[1]
    Ks = args[2]
    theta_d = args[3]
    psi_s = args[4]
    C = Ks * dt + F_1 - psi_s * theta_d * np.log(F_1 + np.abs(psi_s) * theta_d)
    zero = C + psi_s * theta_d * np.log(F_2 + np.abs(psi_s) * theta_d) - F_2
    return zero

@njit
def numba_derivative_green_ampt(F_2, args):
    """
    Derivative of Green Ampt equation for cumulative infiltration.

    Inputs:
    -------
    F_2: np.ndarray (float)
        Cumulative infiltration at current timestep (m)
    F_1: np.ndarray (float)
        Cumulative infiltration at next timestep (m)
    dt: float
        Time step (seconds)
    Ks: np.ndarray (float)
        Saturated hydraulic conductivity (m/s)
    theta_d: np.ndarray (float)
        Soil moisture deficit
    psi_s: np.ndarray (float)
        Soil suction head (m)
    """
    F_1 = args[0]
    dt = args[1]
    Ks = args[2]
    theta_d = args[3]
    psi_s = args[4]
    zero = (psi_s * theta_d / (psi_s * theta_d + F_2)) - 1
    return zero

