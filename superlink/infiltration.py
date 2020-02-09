import numpy as np
import scipy.optimize

class GreenAmpt():
    def __init__(self, Ks, theta_s, theta_i, theta_r, psi_b, lambda_o):
        self.Ks = Ks
        self.theta_s = theta_s
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.psi_b = psi_b
        self.theta_d = (theta_s - theta_i)
        self.theta_du = self.theta_d
        self.theta_dmax = self.theta_d
        # Initialize constant parameters
        m_to_in = 39.97
        in_to_m = 0.0254
        self.psi_f = self.suction_head(theta_i, theta_s, theta_r, Ks, psi_b, lambda_o)
        self.Lu = (4 * np.sqrt(Ks * m_to_in)) * in_to_m
        self.kr = np.sqrt(Ks * m_to_in) / 75
        self.Tr = 4.5 / np.sqrt(Ks * m_to_in)
        # Initialize time-dependent variables
        self.d = 0.                  # Ponded depth
        self.F = 0.                  # Cumulative infiltration (m)
        self.f = 0.                  # Instantaneous infiltration rate (m/hr)
        self.T = 0.                  # Recovery time before next event (hr)
        self.is_saturated = False

    def s2(self, theta_i, theta_s, theta_r, Ks, psi_b, lambda_o):
        num = 2*(theta_s - theta_i) * Ks * psi_b * ((theta_s - theta_r)**(1 / lambda_o + 3)
                                            - (theta_i - theta_r)**(1 / lambda_o + 3))
        den = (lambda_o * ((theta_s - theta_r)**(1 / lambda_o + 3)))*(1 / lambda_o + 3)
        return num / den

    def suction_head(self, theta_i, theta_s, theta_r, Ks, psi_b, lambda_o):
        num = self.s2(theta_i, theta_s, theta_r, Ks, psi_b, lambda_o)
        den = (2 * Ks * (theta_s - theta_i))
        return num / den

    def available_rainfall(self, dt, i):
        d = self.d
        # 1. Compute available rainfall rate
        ia = i + d / dt
        return ia

    def decrement_recovery_time(self, dt):
        # 2. Decrease recovery time
        self.T -= dt

    def unsaturated_case_1(self, dt):
        # 3. If available rainfall is zero:
        kr = self.kr
        Lu = self.Lu
        theta_dmax = self.theta_dmax
        T = self.T
        self.f = 0.
        # Recover upper zone moisture deficit and cumulative infiltration
        d_theta = kr * theta_dmax * dt
        self.theta_du += d_theta
        self.theta_du = np.minimum(np.maximum(self.theta_du, 0), theta_dmax)
        self.F -= d_theta * Lu
        # If min recovery time has expired, mark beginning of new rainfall event
        if (T <= 0):
            self.theta_d = theta_du
            self.F = 0.

    def unsaturated_case_2(self, dt, ia):
        # 4. If available rainfall does not exceed saturated hydr. conductivity
        Lu = self.Lu
        theta_dmax = self.theta_dmax
        self.f = ia
        dF = ia * dt
        self.F += dF
        self.theta_du -= dF / Lu
        self.theta_du = np.minimum(np.maximum(self.theta_du, 0), theta_dmax)

    def unsaturated_case_3(self, dt, ia):
        # 5. If availble rainfall rate exceeds saturated hydr. conductivity
        Ks = self.Ks
        Lu = self.Lu
        theta_dmax = self.theta_dmax
        psi_f = self.psi_f
        theta_d = self.theta_d
        F = self.F
        # Reset recovery time
        self.T = self.Tr
        # Compute volume needed to saturate surface layer
        Fs = Ks * psi_f * theta_d / (ia - Ks)
        if (F >= Fs):
            self.is_saturated = True
            # Run saturated conditions
            self.saturated_case(dt, ia)
        elif (F + ia * dt < Fs):
            # TODO: DRY
            self.f = ia
            dF = ia * dt
            self.F += dF
            self.theta_du -= dF / Lu
            self.theta_du = np.minimum(np.maximum(self.theta_du, 0), theta_dmax)
        else:
            # Solve integrated equation
            sub_dt = dt - (Fs - F) / ia
            F_2 = scipy.optimize.newton(self.integrated_green_ampt, F,
                                        self.derivative_green_ampt,
                                        args=(Fs, sub_dt, Ks, theta_d, psi_f))
            dF = F_2 - F
            self.F = F_2
            self.theta_du -= dF / Lu
            self.theta_du = np.minimum(np.maximum(self.theta_du, 0), theta_dmax)
            self.f = dF / dt

    def saturated_case(self, dt, ia):
        Lu = self.Lu
        Ks = self.Ks
        theta_dmax = self.theta_dmax
        theta_d = self.theta_d
        psi_f = self.psi_f
        F = self.F
        # Reset recovery time
        self.T = self.Tr
        # Solve integrated equation
        F_2 = scipy.optimize.newton(self.integrated_green_ampt, F,
                                    self.derivative_green_ampt,
                                    args=(F, dt, Ks, theta_d, psi_f))
        dF = F_2 - F
        if (dF > ia * dt):
            dF = ia * dt
            # Set current surface layer to unsaturated
            self.is_saturated = False
        self.F += dF
        self.theta_du -= dF / Lu
        self.theta_du = np.minimum(np.maximum(self.theta_du, 0), theta_dmax)
        self.f = dF / dt

    def integrated_green_ampt(self, F_2, F_1, dt, Ks, theta_d, psi_s):
        # NOTE: I'm putting abs to prevent log error
        C = Ks * dt + F_1 - psi_s * theta_d * np.log(F_1 + np.abs(psi_s) * theta_d)
        zero = C + psi_s * theta_d * np.log(F_2 + np.abs(psi_s) * theta_d) - F_2
        return zero

    def derivative_green_ampt(self, F_2, F_1, dt, Ks, theta_d, psi_s):
        zero = (psi_s * theta_d / (psi_s * theta_d + F_2)) - 1
        return zero

    def step(self, dt, i):
        is_saturated = self.is_saturated
        Ks = self.Ks
        ia = self.available_rainfall(dt, i)
        self.decrement_recovery_time(dt)
        if is_saturated:
            self.saturated_case(dt, ia)
        else:
            if (ia == 0.):
                self.unsaturated_case_1(dt)
            elif (ia <= Ks):
                self.unsaturated_case_2(dt, ia)
            elif (ia > Ks):
                self.unsaturated_case_3(dt, ia)


