import numpy as np
from pipedream_solver.nutils import newton_raphson, bounded_newton_raphson, numba_any
from pipedream_solver.infiltration import GreenAmpt
from numba import njit
import scipy.optimize

class nGreenAmpt(GreenAmpt):
    """
    Green Ampt infiltration model, as described in:

    Green, W.H. & Ampt, G. (1911). Studies of soil physics, Part I -
    the flow of air and water through soils. J. Ag. Sci. 4:1-24.

    Inputs:
    -------
    soil_params: pd.DataFrame
        Table containing soil parameters for all catchments.
        The following fields are required.

        |---------+-------+------+------------------------------------------------------|
        | Field   | Type  | Unit | Description                                          |
        |---------+-------+------+------------------------------------------------------|
        | psi_f   | float | m    | Matric potential of the wetting front (suction head) |
        | Ks      | float | m/s  | Saturated hydraulic conductivity                     |
        | theta_s | float | -    | Saturated soil moisture content                      |
        | theta_i | float | -    | Initial soil moisture content                        |
        | A_s     | float | m^2  | Surface area of soil element                         |
        |---------+-------+------+------------------------------------------------------|

    Methods:
    --------
    step: Advances model forward in time, computing infiltration

    Attributes:
    -----------
    f: infiltration rate (m/s)
    F: cumulative infiltration (m)
    d: ponded depth (m)
    T: recovery time (s)
    is_saturated: indicates whether soil element is currently saturated (t/f)
    """
    def __init__(self, soil_params):
        super().__init__(soil_params)

    def unsaturated_case_3(self, dt, ia, case_3):
        """
        Solve Green-Ampt model for second unsaturated case:

        1) Soil is unsaturated
        2) Available rainfall is greater than saturated hydraulic conductivity

        Inputs:
        -------
        dt: float
            Time step (seconds)
        ia: np.ndarray (float)
            Available rainfall (m/s)
        case_3: np.ndarray (bool)
            Indicates whether case 3 applies to given element
        """
        # 5. If availble rainfall rate exceeds saturated hydr. conductivity
        orig_ia = np.copy(ia)
        ia = ia[case_3]
        is_saturated = self.is_saturated[case_3]
        Ks = self.Ks[case_3]
        Lu = self.Lu[case_3]
        theta_dmax = self.theta_dmax[case_3]
        psi_f = self.psi_f[case_3]
        theta_d = self.theta_d[case_3]
        # Variables to write
        F = self.F[case_3]
        f = self.f[case_3]
        theta_du = self.theta_du[case_3]
        # Reset recovery time
        self.T[case_3] = self.Tr[case_3]
        # Compute volume needed to saturate surface layer
        Fs = Ks * psi_f * theta_d / (ia - Ks)
        cond_0 = (F >= Fs)
        cond_1 = (F + ia * dt < Fs) & (~cond_0)
        cond_2 = (~cond_0 & ~cond_1)
        # For first case
        if cond_0.any():
            # Set to saturated
            is_saturated[cond_0] = True
            self.is_saturated[case_3] = is_saturated
            # Run saturated case
            # TODO: This is kind of confusing and probably not efficient
            sat_case = np.zeros(self.N, dtype=bool)
            sat_case[np.flatnonzero(case_3)[np.flatnonzero(cond_0)]] = True
            self.saturated_case(dt, orig_ia, sat_case)
            F[cond_0] = self.F[case_3][cond_0]
            f[cond_0] = self.f[case_3][cond_0]
            theta_du[cond_0] = self.theta_du[case_3][cond_0]
        # Run second case
        if cond_1.any():
            f[cond_1] = ia[cond_1]
            dF = ia[cond_1] * dt
            F[cond_1] += dF
            theta_du[cond_1] -= dF / Lu[cond_1]
            theta_du[cond_1] = np.minimum(np.maximum(theta_du[cond_1], 0),
                                                theta_dmax[cond_1])
        # Run third case
        # Solve integrated equation
        if cond_2.any():
            sub_dt = dt - (Fs[cond_2] - F[cond_2]) / ia[cond_2]
            n = sub_dt.size
            F_2 = np.zeros(n, dtype=float)
            # Run green ampt
            F_2 = run_green_ampt_newton(F_2, F[cond_2], Fs[cond_2], sub_dt, Ks[cond_2],
                                        theta_d[cond_2], psi_f[cond_2], ia[cond_2])
            dF = F_2 - F[cond_2]
            F[cond_2] = F_2
            theta_du[cond_2] -= dF / Lu[cond_2]
            theta_du[cond_2] = np.minimum(np.maximum(theta_du[cond_2], 0),
                                                theta_dmax[cond_2])
            f[cond_2] = dF / dt
        # Export instance variables
        self.F[case_3] = F
        self.f[case_3] = f
        self.theta_du[case_3] = theta_du

    def saturated_case(self, dt, ia, sat_case):
        """
        Solve Green-Ampt model for saturated case:

        1) Soil is saturated

        Inputs:
        -------
        dt: float
            Time step (seconds)
        ia: np.ndarray (float)
            Available rainfall (m/s)
        sat_case: np.ndarray (bool)
            Indicates whether saturated case applies to given element
        """
        ia = ia[sat_case]
        Lu = self.Lu[sat_case]
        Ks = self.Ks[sat_case]
        theta_dmax = self.theta_dmax[sat_case]
        theta_d = self.theta_d[sat_case]
        psi_f = self.psi_f[sat_case]
        # Variables to write
        F = self.F[sat_case]
        f = self.f[sat_case]
        theta_du = self.theta_du[sat_case]
        is_saturated = self.is_saturated[sat_case]
        # Reset recovery time
        self.T[sat_case] = self.Tr[sat_case]
        # Solve integrated equation
        n = F.size
        F_2 = np.zeros(n, dtype=float)
        # Run green ampt equation
        dts = np.full(n, dt)
        F_2 = run_green_ampt_newton(F_2, F, F, dts, Ks, theta_d, psi_f, ia)
        dF = F_2 - F
        cond = (dF > ia * dt)
        if cond.any():
            dF[cond] = ia[cond] * dt
            # Set current surface layer to unsaturated
            is_saturated[cond] = False
        F += dF
        theta_du -= dF / Lu
        theta_du[:] = np.minimum(np.maximum(theta_du, 0), theta_dmax)
        f[:] = dF / dt
        # Export instance variables
        self.F[sat_case] = F
        self.f[sat_case] = f
        self.theta_du[sat_case] = theta_du
        self.is_saturated[sat_case] = is_saturated

    def step(self, dt, i):
        """
        Advance model forward in time, computing infiltration rate and cumulative
        infiltration.

        Inputs:
        -------
        dt: float
            Time step (seconds)
        i: np.ndarray (float)
            Precipitation rate (m/s)
        """
        is_saturated = self.is_saturated
        Ks = self.Ks
        ia = self.available_rainfall(dt, i)
        self.decrement_recovery_time(dt)
        sat_case = is_saturated
        unsat_case_1 = (~is_saturated & (ia == 0.))
        unsat_case_2 = (~is_saturated & (ia <= Ks))
        unsat_case_3 = (~is_saturated & (ia > Ks))
        if numba_any(sat_case):
            self.saturated_case(dt, ia, sat_case)
        if numba_any(unsat_case_1):
            self.unsaturated_case_1(dt, unsat_case_1)
        if numba_any(unsat_case_2):
            self.unsaturated_case_2(dt, ia, unsat_case_2)
        if numba_any(unsat_case_3):
            self.unsaturated_case_3(dt, ia, unsat_case_3)
        self.compute_runoff(i)
        self.compute_ponding_depth(i)
        self.iter_count += 1
        self.t += dt

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
