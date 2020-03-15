import numpy as np
from superlink.nutils import bounded_newton_raphson
from superlink.infiltration import GreenAmpt
from numba import njit

class nGreenAmpt(GreenAmpt):
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
            run_green_ampt_newton(F_2, F[cond_2], sub_dt, Ks[cond_2],
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
        run_green_ampt_newton(F_2, F, dts, Ks, theta_d, psi_f, ia)
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

@njit
def run_green_ampt_newton(F_2, F_1, dt, Ks, theta_d, psi_f, ia):
    n = F_2.size
    for i in range(n):
        F_1_i = F_1[i]
        dt_i = dt[i]
        nargs = np.zeros(5)
        nargs[0] = F_1_i
        nargs[1] = dt_i
        nargs[2] = Ks[i]
        nargs[3] = theta_d[i]
        nargs[4] = psi_f[i]
        min_F = 0
        max_F = F_1_i + ia[i] * dt_i
        F_est = bounded_newton_raphson(numba_integrated_green_ampt,
                                       numba_derivative_green_ampt,
                                       F_1_i, min_F, max_F,
                                       nargs)
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
