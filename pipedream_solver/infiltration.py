import numpy as np
import scipy.optimize

class GreenAmpt():
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
        self.N = len(soil_params)
        for field in ('psi_f', 'Ks', 'theta_s', 'theta_i', 'A_s'):
            if field in soil_params.columns:
                setattr(self, field, soil_params[field].values.astype(float))
            else:
                raise KeyError('Field `{}` required in soil_params table'.format(field))
        # Initial soil moisture deficit
        self.theta_d = (self.theta_s - self.theta_i)
        # Soil moisture deficit in upper layer
        self.theta_du = np.copy(self.theta_d)
        # Maximum soil moisture deficit
        self.theta_dmax = np.copy(self.theta_d)
        # Initialize constant parameters
        m_to_in = 39.97
        in_to_m = 0.0254
        s_to_hr = 3600
        hr_to_s = 1 / 3600
        # Soil capillary suction
        self.Lu = (4 * np.sqrt(self.Ks * m_to_in * s_to_hr)) * in_to_m
        self.kr = (np.sqrt(self.Ks * m_to_in * s_to_hr) / 75) * hr_to_s
        self.Tr = (4.5 / np.sqrt(self.Ks * m_to_in * s_to_hr)) * s_to_hr
        # Initialize time-dependent variables
        self.d = np.zeros(self.N, dtype=float)
        self.F = np.zeros(self.N, dtype=float)
        self.f = np.zeros(self.N, dtype=float)
        self.T = np.zeros(self.N, dtype=float)
        self.is_saturated = np.zeros(self.N, dtype=bool)
        self.iter_count = 0
        self.t = 0

    @classmethod
    def _s2(cls, theta_i, theta_s, theta_r, Ks, psi_b, lambda_o):
        num = 2*(theta_s - theta_i) * Ks * psi_b * ((theta_s - theta_r)**(1 / lambda_o + 3)
                                            - (theta_i - theta_r)**(1 / lambda_o + 3))
        den = (lambda_o * ((theta_s - theta_r)**(1 / lambda_o + 3)))*(1 / lambda_o + 3)
        return num / den

    @classmethod
    def suction_head(cls, theta_i, theta_s, theta_r, Ks, psi_b, lambda_o):
        """
        Compute suction head of soil (matric potential of wetting front).
        """
        num = cls._s2(theta_i, theta_s, theta_r, Ks, psi_b, lambda_o)
        den = (2 * Ks * (theta_s - theta_i))
        return num / den

    def available_rainfall(self, dt, i):
        """
        Compute available rainfall rate

        Inputs:
        -------
        dt: float
            Time step (seconds)
        i: float
            Rainfall rate (m/s)
        """
        d = self.d
        # 1. Compute available rainfall rate
        ia = i + d / dt
        return ia

    def decrement_recovery_time(self, dt):
        """
        Decrement recovery time

        Inputs:
        -------
        dt: float
            Time step (seconds)
        """
        # 2. Decrease recovery time
        self.T -= dt

    def unsaturated_case_1(self, dt, case_1):
        """
        Solve Green-Ampt model for first unsaturated case:

        1) Soil is unsaturated
        2) Available rainfall is zero

        Inputs:
        -------
        dt: float
            Time step (seconds)
        case_1: np.ndarray (bool)
            Indicates whether case 1 applies to given element
        """
        # 3. If available rainfall is zero:
        kr = self.kr[case_1]
        Lu = self.Lu[case_1]
        theta_dmax = self.theta_dmax[case_1]
        T = self.T[case_1]
        # Variables to be written
        F = self.F[case_1]
        f = self.f[case_1]
        theta_du = self.theta_du[case_1]
        theta_d = self.theta_d[case_1]
        # Set infiltration rate to zero
        f[:] = 0.
        # Recover upper zone moisture deficit and cumulative infiltration
        d_theta = kr * theta_dmax * dt
        theta_du += d_theta
        theta_du[:] = np.minimum(np.maximum(theta_du, 0), theta_dmax)
        F -= d_theta * Lu
        # If min recovery time has expired, mark beginning of new rainfall event
        cond = (T <= 0)
        if cond.any():
            theta_d[cond] = theta_du[cond]
            F[cond] = 0.
        # Export instance variables
        self.F[case_1] = F
        self.f[case_1] = f
        self.theta_du[case_1] = theta_du
        self.theta_d[case_1] = theta_d

    def unsaturated_case_2(self, dt, ia, case_2):
        """
        Solve Green-Ampt model for second unsaturated case:

        1) Soil is unsaturated
        2) Available rainfall greater than zero
        3) Available rainfall is less than saturated hydraulic conductivity

        Inputs:
        -------
        dt: float
            Time step (seconds)
        ia: np.ndarray (float)
            Available rainfall (m/s)
        case_2: np.ndarray (bool)
            Indicates whether case 2 applies to given element
        """
        # 4. If available rainfall does not exceed saturated hydr. conductivity
        ia = ia[case_2]
        Lu = self.Lu[case_2]
        theta_dmax = self.theta_dmax[case_2]
        # Variables to be written
        F = self.F[case_2]
        f = self.f[case_2]
        theta_du = self.theta_du[case_2]
        f[:] = ia
        dF = ia * dt
        F += dF
        theta_du -= dF / Lu
        theta_du[:] = np.minimum(np.maximum(theta_du, 0), theta_dmax)
        # Export instance variables
        self.F[case_2] = F
        self.f[case_2] = f
        self.theta_du[case_2] = theta_du

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
            for i in range(n):
                F_2[i] = scipy.optimize.newton(self.integrated_green_ampt, F[cond_2][i],
                                               self.derivative_green_ampt,
                                               args=(Fs[cond_2][i], sub_dt[i], Ks[cond_2][i],
                                                     theta_d[cond_2][i], psi_f[cond_2][i]))
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
        for i in range(n):
            F_2[i] = scipy.optimize.newton(self.integrated_green_ampt, F[i],
                                        self.derivative_green_ampt,
                                        args=(F[i], dt, Ks[i], theta_d[i], psi_f[i]))
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

    def integrated_green_ampt(self, F_2, F_1, dt, Ks, theta_d, psi_s):
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
        C = Ks * dt + F_1 - psi_s * theta_d * np.log(F_1 + np.abs(psi_s) * theta_d)
        zero = C + psi_s * theta_d * np.log(F_2 + np.abs(psi_s) * theta_d) - F_2
        return zero

    def derivative_green_ampt(self, F_2, F_1, dt, Ks, theta_d, psi_s):
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
        zero = (psi_s * theta_d / (psi_s * theta_d + F_2)) - 1
        return zero

    def compute_runoff(self, i):
        f = self.f
        A_s = self.A_s
        self.Q = np.maximum((i - f), 0.) * A_s

    def compute_ponding_depth(self, i):
        f = self.f
        self.d += (i - f)
        self.d = np.maximum(self.d, 0.)

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
        if sat_case.any():
            self.saturated_case(dt, ia, sat_case)
        if unsat_case_1.any():
            self.unsaturated_case_1(dt, unsat_case_1)
        if unsat_case_2.any():
            self.unsaturated_case_2(dt, ia, unsat_case_2)
        if unsat_case_3.any():
            self.unsaturated_case_3(dt, ia, unsat_case_3)
        self.compute_runoff(i)
        self.compute_ponding_depth(i)
        self.iter_count += 1
        self.t += dt


