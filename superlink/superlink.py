import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg

class SuperLink():
    def __init__(self, superlinks, superjunctions, links, junctions, dt=60):
        self.superlinks = superlinks
        self.superjunctions = superjunctions
        self.links = links
        self.junctions = junctions
        self._dt = dt
        self._I = junctions.index.values
        self._ik = links.index.values
        self._Ik = links['j_0'].values.astype(int)
        self._Ip1k = links['j_1'].values.astype(int)
        self.start_nodes = superlinks['j_0'].values.astype(int)
        self.end_nodes = superlinks['j_1'].values.astype(int)
        self._is_start = np.zeros(self._I.size, dtype=bool)
        self._is_end = np.zeros(self._I.size, dtype=bool)
        self._is_start[self.start_nodes] = True
        self._is_end[self.end_nodes] = True
        self.middle_nodes = self._I[(~self._is_start) & (~self._is_end)]
        self.forward_I_i = pd.Series(self._ik, index=self._Ik)
        self.backward_I_i = pd.Series(self._ik, index=self._Ip1k)
        self.forward_I_I = pd.Series(self._Ip1k, index=self._Ik)
        self.backward_I_I = pd.Series(self._Ik, index=self._Ip1k)
        self._w_ik = links['w'].values.astype(float)
        self._Q_ik = links['Q_0'].values.astype(float)
        self._dx_ik = links['dx'].values.astype(float)
        self._n_ik = links['n'].values.astype(float)
        self._ctrl = links['ctrl'].values.astype(bool)
        self._A_c_ik = links['A_c'].values.astype(float)
        self._C_ik = links['C'].values.astype(float)
        self._h_Ik = junctions.loc[self._I, 'h_0'].values.astype(float)
        self._A_SIk = junctions.loc[self._I, 'A_s'].values.astype(float)
        self._z_inv_Ik = junctions.loc[self._I, 'z_inv'].values.astype(float)
        self._S_o_ik = ((self._z_inv_Ik[self._Ik] - self._z_inv_Ik[self._Ip1k])
                        / self._dx_ik)
        self._Q_0Ik = np.zeros(self._I.size, dtype=float)
        # Node velocities
        self._u_Ik = np.zeros(self._Ik.size, dtype=float)
        self._u_Ip1k = np.zeros(self._Ip1k.size, dtype=float)
        # Node coefficients
        self._E_Ik = np.zeros(self._I.size)
        self._D_Ik = np.zeros(self._I.size)
        # Forward recurrence relations
        self._I_end = np.zeros(self._I.size, dtype=bool)
        self._I_end[self.end_nodes] = True
        self._I_1k = self.start_nodes
        self._I_2k = self.forward_I_I[self._I_1k].values
        self._i_1k = self.forward_I_i[self._I_1k].values
        self._T_ik = np.zeros(self._ik.size)
        self._U_Ik = np.zeros(self._I.size)
        self._V_Ik = np.zeros(self._I.size)
        self._W_Ik = np.zeros(self._I.size)
        # Backward recurrence relations
        self._I_start = np.zeros(self._I.size, dtype=bool)
        self._I_start[self.start_nodes] = True
        self._I_Np1k = self.end_nodes
        self._I_Nk = self.backward_I_I[self._I_Np1k].values
        self._i_nk = self.backward_I_i[self._I_Np1k].values
        self._O_ik = np.zeros(self._ik.size)
        self._X_Ik = np.zeros(self._I.size)
        self._Y_Ik = np.zeros(self._I.size)
        self._Z_Ik = np.zeros(self._I.size)
        # Head at superjunctions
        self._z_inv_j = self.superjunctions['z_inv'].values
        self.H_j = self.superjunctions['h_0'].values + self._z_inv_j
        # Coefficients for head at upstream ends of superlink k
        self._J_uk = self.superlinks['sj_0'].values.astype(int)
        self._z_inv_uk = self._z_inv_Ik[self._I_1k]
        # Coefficients for head at downstream ends of superlink k
        self._J_dk = self.superlinks['sj_1'].values.astype(int)
        self._z_inv_dk = self._z_inv_Ik[self._I_Np1k]
        # Sparse matrix coefficients
        self.M = len(superjunctions)
        self.A = scipy.sparse.lil_matrix((self.M, self.M))
        self.b = np.zeros(self.M)
        self.bc = self.superjunctions['bc'].values.astype(bool)
        self._alpha_ukm = np.zeros(self.M)
        self._beta_dkl = np.zeros(self.M)
        self._chi_ukl = np.zeros(self.M)
        self._chi_dkm = np.zeros(self.M)
        self._k = superlinks.index.values
        self._A_sj = self.superjunctions['A_sj'].values.astype(float)
        self._Q_0j = 0
        # Set upstream and downstream superlink variables
        self._Q_uk = self._Q_ik[self._i_1k]
        self._Q_dk = self._Q_ik[self._i_nk]
        self._h_uk = self._h_Ik[self._I_1k]
        self._h_dk = self._h_Ik[self._I_Np1k]
        # Other parameters
        self.min_depth = 1e-5
        # Initialize to stable state
        self.step(_dt=1e-6, first_time=True)

    def safe_divide(function):
        def inner(*args, **kwargs):
            num, den = function(*args, **kwargs)
            cond = (den != 0)
            result = np.zeros(num.size)
            result[cond] = num[cond] / den[cond]
            return result
        return inner

    # Node velocities
    def A_ik(self, h_Ik, h_Ip1k, w):
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > w] = w[y > w]
        r = w / 2
        theta = np.arccos(1 - y / r)
        A = r**2 * (theta - np.cos(theta) * np.sin(theta))
        return A

    def Pe_ik(self, h_Ik, h_Ip1k, w):
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        y[y > w] = w[y > w]
        r = w / 2
        theta = np.arccos(1 - y / r)
        Pe = 2 * r * theta
        return Pe

    def R_ik(self, A_ik, Pe_ik):
        cond = Pe_ik > 0
        R = np.zeros(A_ik.size)
        R[cond] = A_ik[cond] / Pe_ik[cond]
        return R

    def B_ik(self, h_Ik, h_Ip1k, w):
        y = (h_Ik + h_Ip1k) / 2
        y[y < 0] = 0
        r = w / 2
        theta = np.arccos(1 - y / r)
        cond = (y < w)
        B = np.zeros(y.size)
        B[~cond] = 0.001 * w[~cond]
        B[cond] = 2 * r[cond] * np.sin(theta[cond])
        return B

    @safe_divide
    def u_ik(self, Q_ik, A_ik):
        num = Q_ik
        den = np.where(A_ik > 0, A_ik, 0)
        return num, den

    @safe_divide
    def u_Ip1k(self, dx_ik, u_ip1k, dx_ip1k, u_ik):
        num = dx_ik * u_ip1k + dx_ip1k * u_ik
        den = dx_ik + dx_ip1k
        return num, den

    @safe_divide
    def u_Ik(self, dx_ik, u_im1k, dx_im1k, u_ik):
        num = dx_ik * u_im1k + dx_im1k * u_ik
        den = dx_ik + dx_im1k
        return num, den

    # Link coefficients for superlink k
    def a_ik(self, u_Ik):
        return -np.maximum(u_Ik, 0)

    def c_ik(self, u_Ip1k):
        return -np.maximum(-u_Ip1k, 0)

    def b_ik(self, dx_ik, dt, n_ik, Q_ik_t, A_ik, R_ik,
             A_c_ik, C_ik, a_ik, c_ik, ctrl, g=9.81):
        # TODO: Clean up
        cond = A_ik > 0
        t_0 = dx_ik / dt
        t_1 = np.zeros(Q_ik_t.size)
        t_1[cond] = (g * n_ik[cond]**2 * np.abs(Q_ik_t[cond]) * dx_ik[cond]
                    / A_ik[cond] / R_ik[cond]**(4/3))
        t_2 = np.zeros(ctrl.size)
        cond = ctrl
        t_2[cond] = A_ik[cond] * np.abs(Q_ik_t[cond]) / A_c_ik[cond]**2 / C_ik[cond]**2
        t_3 = a_ik
        t_4 = c_ik
        return t_0 + t_1 + t_2 - t_3 - t_4

    def P_ik(self, Q_ik_t, dx_ik, dt, A_ik, S_o_ik, g=9.81):
        t_0 = Q_ik_t * dx_ik / dt
        t_1 = g * A_ik * S_o_ik * dx_ik
        return t_0 + t_1

    # Node coefficients for superlink k
    def E_Ik(self, B_ik, dx_ik, B_im1k, dx_im1k, A_SIk, dt):
        t_0 = B_ik * dx_ik / 2
        t_1 = B_im1k * dx_im1k / 2
        t_2 = A_SIk
        t_3 = dt
        return (t_0 + t_1 + t_2) / t_3

    def D_Ik(self, Q_0IK, B_ik, dx_ik, B_im1k, dx_im1k, A_SIk, h_Ik_t, dt):
        t_0 = Q_0IK
        t_1 = B_ik * dx_ik / 2
        t_2 = B_im1k * dx_im1k / 2
        t_3 = A_SIk
        t_4 = h_Ik_t / dt
        return t_0 + ((t_1 + t_2 + t_3) * t_4)

    # Forward recurrence relation coefficients
    @safe_divide
    def U_1k(self, E_2k, c_1k, A_1k, T_1k, g=9.81):
        num = E_2k * c_1k - g * A_1k
        den = T_1k
        return num, den

    @safe_divide
    def V_1k(self, P_1k, D_2k, c_1k, T_1k):
        num = P_1k - D_2k * c_1k
        den = T_1k
        return num, den

    @safe_divide
    def W_1k(self, A_1k, T_1k, g=9.81):
        num = g * A_1k
        den = T_1k
        return num, den

    def T_1k(self, a_1k, b_1k, c_1k):
        return a_1k + b_1k + c_1k

    @safe_divide
    def U_Ik(self, E_Ip1k, c_ik, A_ik, T_ik, g=9.81):
        num = E_Ip1k * c_ik - g * A_ik
        den = T_ik
        return num, den

    @safe_divide
    def V_Ik(self, P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ik, V_Im1k, U_Im1k, T_ik, g=9.81):
        t_0 = P_ik
        t_1 = a_ik * D_Ik
        t_2 = D_Ip1k * c_ik
        t_3 = (g * A_ik - E_Ik * a_ik)
        t_4 = V_Im1k + D_Ik
        t_5 = U_Im1k - E_Ik
        t_6 = T_ik
        # TODO: There is still a divide by zero here
        num = (t_0 + t_1 - t_2 - (t_3 * t_4 / t_5))
        den = t_6
        return  num, den

    @safe_divide
    def W_Ik(self, A_ik, E_Ik, a_ik, W_Im1k, U_Im1k, T_ik, g=9.81):
        num = -(g * A_ik - E_Ik * a_ik) * W_Im1k
        den = (U_Im1k - E_Ik) * T_ik
        return num, den

    def T_ik(self, a_ik, b_ik, c_ik, A_ik, E_Ik, U_Im1k, g=9.81):
        t_0 = a_ik + b_ik + c_ik
        t_1 = g * A_ik - E_Ik * a_ik
        t_2 = U_Im1k - E_Ik
        # TODO: Can't use decorator here
        cond = t_2 != 0
        result = np.zeros(t_0.size)
        # TODO: Not sure if ~cond should be zero
        result[cond] = t_0[cond] - (t_1[cond] / t_2[cond])
        return result

    # Reverse recurrence relation coefficients
    @safe_divide
    def X_Nk(self, A_nk, E_Nk, a_nk, O_nk, g=9.81):
        num = g * A_nk - E_Nk * a_nk
        den = O_nk
        return num, den

    @safe_divide
    def Y_Nk(self, P_nk, D_Nk, a_nk, O_nk):
        num = P_nk + D_Nk * a_nk
        den = O_nk
        return num, den

    @safe_divide
    def Z_Nk(self, A_nk, O_nk, g=9.81):
        num = - g * A_nk
        den = O_nk
        return num, den

    def O_nk(self, a_nk, b_nk, c_nk):
        return a_nk + b_nk + c_nk

    @safe_divide
    def X_Ik(self, A_ik, E_Ik, a_ik, O_ik, g=9.81):
        num = g * A_ik - E_Ik * a_ik
        den = O_ik
        return num, den

    @safe_divide
    def Y_Ik(self, P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ip1k, Y_Ip1k, X_Ip1k, O_ik, g=9.81):
        t_0 = P_ik
        t_1 = a_ik * D_Ik
        t_2 = D_Ip1k * c_ik
        t_3 = (g * A_ik - E_Ip1k * c_ik)
        t_4 = D_Ip1k - Y_Ip1k
        t_5 = X_Ip1k + E_Ip1k
        t_6 = O_ik
        num = (t_0 + t_1 - t_2 - (t_3 * t_4 / t_5))
        den = t_6
        return num, den

    @safe_divide
    def Z_Ik(self, A_ik, E_Ip1k, c_ik, Z_Ip1k, X_Ip1k, O_ik, g=9.81):
        num = (g * A_ik - E_Ip1k * c_ik) * Z_Ip1k
        den = (X_Ip1k + E_Ip1k) * O_ik
        return num, den

    def O_ik(self, a_ik, b_ik, c_ik, A_ik, E_Ip1k, X_Ip1k, g=9.81):
        t_0 = a_ik + b_ik + c_ik
        t_1 = g * A_ik - E_Ip1k * c_ik
        t_2 = X_Ip1k + E_Ip1k
        cond = t_2 != 0
        result = np.zeros(t_0.size)
        # TODO: Not sure if ~cond should be zero
        result[cond] = t_0[cond] + (t_1[cond] / t_2[cond])
        return result

    # Coefficients for head at upstream and downstream ends of superlink k
    def dH_uk(self, H_juk, zinv_uk, h_uk):
        return H_juk - zinv_uk - h_uk

    def dH_dk(self, H_jdk, zinv_dk, h_dk):
        return zinv_dk + h_dk - H_jdk

    @safe_divide
    def gamma_uk(self, Q_uk_t, C_uk, A_uk, g=9.81):
        num = np.abs(Q_uk_t)
        den = (C_uk**2) * (A_uk**2) * g
        return num, den

    @safe_divide
    def gamma_dk(self, Q_dk_t, C_dk, A_dk, g=9.81):
        num = -np.abs(Q_dk_t)
        den = (C_dk**2) * (A_dk**2) * g
        return num, den

    def D_k_star(self, X_1k, gamma_uk, U_Nk, gamma_dk, Z_1k, W_Nk):
        t_0 = (X_1k * gamma_uk - 1) * (U_Nk * gamma_dk - 1)
        t_1 = (Z_1k * gamma_dk) * (W_Nk * gamma_uk)
        return t_0 - t_1

    @safe_divide
    def alpha_uk(self, U_Nk, gamma_dk, X_1k, Z_1k, W_Nk, D_k_star):
        num = (1 - U_Nk * gamma_dk) * X_1k + (Z_1k * gamma_dk * W_Nk)
        den = D_k_star
        return num, den

    @safe_divide
    def beta_uk(self, U_Nk, gamma_dk, Z_1k, W_Nk, D_k_star):
        num = (1 - U_Nk * gamma_dk) * Z_1k + (Z_1k * gamma_dk * U_Nk)
        den = D_k_star
        return num, den

    @safe_divide
    def chi_uk(self, U_Nk, gamma_dk, Y_1k, X_1k, z_inv_uk, Z_1k,
               z_inv_dk, V_Nk, W_Nk, D_k_star):
        t_0 = (1 - U_Nk * gamma_dk) * (Y_1k - X_1k * z_inv_uk - Z_1k * z_inv_dk)
        t_1 = (Z_1k * gamma_dk) * (V_Nk - W_Nk * z_inv_uk - U_Nk * z_inv_dk)
        num = t_0 + t_1
        den = D_k_star
        return num, den

    @safe_divide
    def alpha_dk(self, X_1k, gamma_uk, W_Nk, D_k_star):
        num = (1 - X_1k * gamma_uk) * W_Nk + (W_Nk * gamma_uk * X_1k)
        den = D_k_star
        return num, den

    @safe_divide
    def beta_dk(self, X_1k, gamma_uk, U_Nk, W_Nk, Z_1k, D_k_star):
        num = (1 - X_1k * gamma_uk) * U_Nk + (W_Nk * gamma_uk * Z_1k)
        den = D_k_star
        return num, den

    @safe_divide
    def chi_dk(self, X_1k, gamma_uk, V_Nk, W_Nk, z_inv_uk, U_Nk,
               z_inv_dk, Y_1k, Z_1k, D_k_star):
        t_0 = (1 - X_1k * gamma_uk) * (V_Nk - W_Nk * z_inv_uk - U_Nk * z_inv_dk)
        t_1 = (W_Nk * gamma_uk) * (Y_1k - X_1k * z_inv_uk - Z_1k * z_inv_dk)
        num = t_0 + t_1
        den = D_k_star
        return num, den

    # Sparse matrix coefficients
    def F_jj(self, A_sj, dt, beta_dkl, alpha_ukm):
        t_0 = A_sj / dt
        t_1 = beta_dkl
        t_2 = alpha_ukm
        return t_0 - t_1 + t_2

    def G_j(self, A_sj, dt, H_j, Q_0j, chi_ukl, chi_dkm):
        t_0 = A_sj * H_j / dt
        t_1 = Q_0j
        t_2 = chi_ukl
        t_3 = chi_dkm
        # chi signs are switched in original paper
        return t_0 + t_1 - t_2 + t_3

    def node_velocities(self):
        # Import instance variables for better readability
        # TODO: Should probably use forward_I_i instead of _ik directly
        _ik = self._ik
        _Ik = self._Ik
        _Ip1k = self._Ip1k
        _h_Ik = self._h_Ik
        _w_ik = self._w_ik
        _Q_ik = self._Q_ik
        _u_Ik = self._u_Ik
        _u_Ip1k = self._u_Ip1k
        backward_I_i = self.backward_I_i
        forward_I_i = self.forward_I_i
        _dx_ik = self._dx_ik
        # TODO: Watch this
        _is_start_Ik = self._is_start[_Ik]
        _is_end_Ip1k = self._is_end[_Ip1k]
        # Compute hydraulic geometry and link velocities
        _A_ik = self.A_ik(_h_Ik[_Ik], _h_Ik[_Ip1k], _w_ik)
        _Pe_ik = self.Pe_ik(_h_Ik[_Ik], _h_Ik[_Ip1k], _w_ik)
        _B_ik = self.B_ik(_h_Ik[_Ik], _h_Ik[_Ip1k], _w_ik)
        _R_ik = self.R_ik(_A_ik, _Pe_ik)
        _u_ik = self.u_ik(_Q_ik, _A_ik)
        # Compute velocities for start nodes (1 -> Nk)
        _u_Ik[_is_start_Ik] = _u_ik[_is_start_Ik]
        backward = backward_I_i[_Ik[~_is_start_Ik]].values
        center = _ik[~_is_start_Ik]
        _u_Ik[~_is_start_Ik] = self.u_Ik(_dx_ik[center], _u_ik[backward],
                                         _dx_ik[backward], _u_ik[center])
        # Compute velocities for end nodes (2 -> Nk+1)
        _u_Ip1k[_is_end_Ip1k] = _u_ik[_is_end_Ip1k]
        forward = forward_I_i[_Ip1k[~_is_end_Ip1k]].values
        center = _ik[~_is_end_Ip1k]
        _u_Ip1k[~_is_end_Ip1k] = self.u_Ip1k(_dx_ik[center], _u_ik[forward],
                                             _dx_ik[forward], _u_ik[center])
        # Export to instance variables
        self._A_ik = _A_ik
        self._Pe_ik = _Pe_ik
        self._B_ik = _B_ik
        self._R_ik = _R_ik
        self._u_ik = _u_ik
        self._u_Ik = _u_Ik
        self._u_Ip1k = _u_Ip1k

    def link_coeffs(self, _dt=None):
        # Import instance variables
        _u_Ik = self._u_Ik
        _u_Ip1k = self._u_Ip1k
        _dx_ik = self._dx_ik
        _n_ik = self._n_ik
        _Q_ik = self._Q_ik
        _A_ik = self._A_ik
        _R_ik = self._R_ik
        _S_o_ik = self._S_o_ik
        _A_c_ik = self._A_c_ik
        _C_ik = self._C_ik
        _ctrl = self._ctrl
        if _dt is None:
            _dt = self._dt
        # Compute link coefficients
        _a_ik = self.a_ik(_u_Ik)
        _c_ik = self.c_ik(_u_Ip1k)
        _b_ik = self.b_ik(_dx_ik, _dt, _n_ik, _Q_ik, _A_ik, _R_ik,
                          _A_c_ik, _C_ik, _a_ik, _c_ik, _ctrl)
        _P_ik = self.P_ik(_Q_ik, _dx_ik, _dt, _A_ik, _S_o_ik)
        # Export to instance variables
        self._a_ik = _a_ik
        self._b_ik = _b_ik
        self._c_ik = _c_ik
        self._P_ik = _P_ik

    def node_coeffs(self, _Q_0Ik=None, _dt=None):
        # Import instance variables
        _I = self._I
        start_nodes = self.start_nodes
        end_nodes = self.end_nodes
        middle_nodes = self.middle_nodes
        forward_I_i = self.forward_I_i
        backward_I_i = self.backward_I_i
        _B_ik = self._B_ik
        _dx_ik = self._dx_ik
        _A_SIk = self._A_SIk
        _h_Ik = self._h_Ik
        _E_Ik = self._E_Ik
        _D_Ik = self._D_Ik
        if _dt is None:
            _dt = self._dt
        if _Q_0Ik is None:
            _Q_0Ik = np.zeros(_I.size)
        # Compute E_Ik and D_Ik
        start_links = self.forward_I_i[start_nodes].values
        end_links = self.backward_I_i[end_nodes].values
        backward = self.backward_I_i[middle_nodes].values
        forward = self.forward_I_i[middle_nodes].values
        _E_Ik[start_nodes] = self.E_Ik(_B_ik[start_links], _dx_ik[start_links],
                                        _B_ik[start_links], _dx_ik[start_links],
                                        _A_SIk[start_nodes], _dt)
        _E_Ik[end_nodes] = self.E_Ik(_B_ik[end_links], _dx_ik[end_links],
                                        _B_ik[end_links], _dx_ik[end_links],
                                        _A_SIk[end_nodes], _dt)
        _E_Ik[middle_nodes] = self.E_Ik(_B_ik[forward], _dx_ik[forward],
                                        _B_ik[backward], _dx_ik[backward],
                                        _A_SIk[middle_nodes], _dt)
        _D_Ik[start_nodes] = self.D_Ik(_Q_0Ik[start_nodes], _B_ik[start_links],
                                        _dx_ik[start_links], _B_ik[start_links],
                                        _dx_ik[start_links], _A_SIk[start_nodes],
                                        _h_Ik[start_nodes], _dt)
        _D_Ik[end_nodes] = self.D_Ik(_Q_0Ik[end_nodes], _B_ik[end_links],
                                        _dx_ik[end_links], _B_ik[end_links],
                                        _dx_ik[end_links], _A_SIk[end_nodes],
                                        _h_Ik[end_nodes], _dt)
        _D_Ik[middle_nodes] = self.D_Ik(_Q_0Ik[middle_nodes], _B_ik[forward],
                                        _dx_ik[forward], _B_ik[backward],
                                        _dx_ik[backward], _A_SIk[middle_nodes],
                                        _h_Ik[middle_nodes], _dt)
        # Export instance variables
        self._E_Ik = _E_Ik
        self._D_Ik = _D_Ik

    def forward_recurrence(self):
        # Import instance variables
        backward_I_I = self.backward_I_I
        forward_I_I = self.forward_I_I
        forward_I_i = self.forward_I_i
        _I_end = self._I_end
        _I_1k = self._I_1k
        _I_2k = self._I_2k
        _i_1k = self._i_1k
        _A_ik = self._A_ik
        _E_Ik = self._E_Ik
        _D_Ik = self._D_Ik
        _a_ik = self._a_ik
        _b_ik = self._b_ik
        _c_ik = self._c_ik
        _P_ik = self._P_ik
        _T_ik = self._T_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        # Compute coefficients for starting nodes
        _E_2k = _E_Ik[_I_2k]
        _D_2k = _D_Ik[_I_2k]
        _T_1k = self.T_1k(_a_ik[_i_1k], _b_ik[_i_1k], _c_ik[_i_1k])
        _U_1k = self.U_1k(_E_2k, _c_ik[_i_1k], _A_ik[_i_1k], _T_1k)
        _V_1k = self.V_1k(_P_ik[_i_1k], _D_2k, _c_ik[_i_1k], _T_1k)
        _W_1k = self.W_1k(_A_ik[_i_1k], _T_1k)
        # I = 1, i = 1
        _T_ik[_i_1k] = _T_1k
        _U_Ik[_I_1k] = _U_1k
        _V_Ik[_I_1k] = _V_1k
        _W_Ik[_I_1k] = _W_1k
        # I = 2, i = 2
        _I_next = _I_2k[~_I_end[_I_2k]]
        # Loop from 2 -> Nk
        while _I_next.size:
            _Im1_next = backward_I_I[_I_next].values
            _Ip1_next = forward_I_I[_I_next].values
            _i_next = forward_I_i[_I_next].values
            _T_ik[_i_next] = self.T_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                       _A_ik[_i_next], _E_Ik[_I_next], _U_Ik[_Im1_next])
            _U_Ik[_I_next] = self.U_Ik(_E_Ik[_Ip1_next], _c_ik[_i_next],
                                       _A_ik[_i_next], _T_ik[_i_next])
            _V_Ik[_I_next] = self.V_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                       _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                       _E_Ik[_I_next], _V_Ik[_Im1_next], _U_Ik[_Im1_next],
                                       _T_ik[_i_next])
            _W_Ik[_I_next] = self.W_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                       _W_Ik[_Im1_next], _U_Ik[_Im1_next], _T_ik[_i_next])
            _I_next = _Ip1_next[~_I_end[_Ip1_next]]
        # Try resetting
        _T_ik[_i_1k] = _T_1k
        _U_Ik[_I_1k] = _U_1k
        _V_Ik[_I_1k] = _V_1k
        _W_Ik[_I_1k] = _W_1k
        # Export instance variables
        self._T_ik = _T_ik
        self._U_Ik = _U_Ik
        self._V_Ik = _V_Ik
        self._W_Ik = _W_Ik

    def backward_recurrence(self):
        # Import instance variables
        backward_I_I = self.backward_I_I
        forward_I_I = self.forward_I_I
        forward_I_i = self.forward_I_i
        _I_start = self._I_start
        _I_Nk = self._I_Nk
        _i_nk = self._i_nk
        _A_ik = self._A_ik
        _E_Ik = self._E_Ik
        _D_Ik = self._D_Ik
        _a_ik = self._a_ik
        _b_ik = self._b_ik
        _c_ik = self._c_ik
        _P_ik = self._P_ik
        _O_ik = self._O_ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        # Compute coefficients for starting nodes
        _E_Nk = _E_Ik[_I_Nk]
        _D_Nk = _D_Ik[_I_Nk]
        _O_nk = self.O_nk(_a_ik[_i_nk], _b_ik[_i_nk], _c_ik[_i_nk])
        _X_Nk = self.X_Nk(_A_ik[_i_nk], _E_Nk, _a_ik[_i_nk], _O_nk)
        _Y_Nk = self.Y_Nk(_P_ik[_i_nk], _D_Nk, _a_ik[_i_nk], _O_nk)
        _Z_Nk = self.Z_Nk(_A_ik[_i_nk], _O_nk)
        # I = Nk, i = nk
        _O_ik[_i_nk] = _O_nk
        _X_Ik[_I_Nk] = _X_Nk
        _Y_Ik[_I_Nk] = _Y_Nk
        _Z_Ik[_I_Nk] = _Z_Nk
        # I = Nk-1, i = nk-1
        _I_next = backward_I_I[_I_Nk[~_I_start[_I_Nk]]].values
        # Loop from Nk - 1 -> 1
        while _I_next.size:
            _Ip1_next = forward_I_I[_I_next].values
            _i_next = forward_I_i[_I_next].values
            _O_ik[_i_next] = self.O_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                _A_ik[_i_next], _E_Ik[_Ip1_next], _X_Ik[_Ip1_next])
            _X_Ik[_I_next] = self.X_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                       _O_ik[_i_next])
            _Y_Ik[_I_next] = self.Y_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                       _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                       _E_Ik[_Ip1_next], _Y_Ik[_Ip1_next], _X_Ik[_Ip1_next],
                                       _O_ik[_i_next])
            _Z_Ik[_I_next] = self.Z_Ik(_A_ik[_i_next], _E_Ik[_Ip1_next], _c_ik[_i_next],
                                       _Z_Ik[_Ip1_next], _X_Ik[_Ip1_next], _O_ik[_i_next])
            _I_next = backward_I_I[_I_next[~_I_start[_I_next]]].values
        # Try resetting
        _O_ik[_i_nk] = _O_nk
        _X_Ik[_I_Nk] = _X_Nk
        _Y_Ik[_I_Nk] = _Y_Nk
        _Z_Ik[_I_Nk] = _Z_Nk
        # Export instance variables
        self._O_ik = _O_ik
        self._X_Ik = _X_Ik
        self._Y_Ik = _Y_Ik
        self._Z_Ik = _Z_Ik

    def superlink_upstream_head_coefficients(self):
        # Import instance variables
        _I_1k = self._I_1k
        _i_1k = self._i_1k
        _h_Ik = self._h_Ik
        _J_uk = self._J_uk
        _z_inv_uk = self._z_inv_uk
        _A_ik = self._A_ik
        # _B_ik = self._B_ik
        _Q_ik = self._Q_ik
        _w_ik = self._w_ik
        H_j = self.H_j
        # Placeholder discharge coefficient
        _C_uk = 0.67
        # Current upstream flows
        _Q_uk_t = _Q_ik[_i_1k]
        # Compute superjunction head
        _H_juk = H_j[_J_uk]
        _dH_uk = self.dH_uk(_H_juk, _z_inv_uk, _h_Ik[_I_1k])
        # Compute flow area
        _h_juk = _H_juk - _z_inv_uk
        _A_juk = self.A_ik(_h_juk, _h_juk, _w_ik[_i_1k])
        # _A_uk = (_A_juk + _A_ik[_i_1k]) / 2
        _A_uk = _A_ik[_i_1k]
        # Compute superlink upstream coefficients
        _gamma_uk = self.gamma_uk(_Q_uk_t, _C_uk, _A_uk)
        self._gamma_uk = _gamma_uk

    def superlink_downstream_head_coefficients(self):
        # Import instance variables
        _I_Np1k = self._I_Np1k
        _i_nk = self._i_nk
        _h_Ik = self._h_Ik
        _J_dk = self._J_dk
        _z_inv_dk = self._z_inv_dk
        _A_ik = self._A_ik
        # _B_ik = self._B_ik
        _Q_ik = self._Q_ik
        _w_ik = self._w_ik
        H_j = self.H_j
        # Placeholder discharge coefficient
        _C_dk = 0.67
        # Current upstream flows
        _Q_dk_t = _Q_ik[_i_nk]
        # Compute superjunction head
        _H_jdk = H_j[_J_dk]
        _dH_dk = self.dH_dk(_H_jdk, _z_inv_dk, _h_Ik[_I_Np1k])
        # Compute flow area
        _h_jdk = _H_jdk - _z_inv_dk
        _A_jdk = self.A_ik(_h_jdk, _h_jdk, _w_ik[_i_nk])
        # _A_dk = (_A_jdk + _A_ik[_i_nk]) / 2
        _A_dk = _A_ik[_i_nk]
        # Compute superlink downstream coefficients
        _gamma_dk = self.gamma_dk(_Q_dk_t, _C_dk, _A_dk)
        self._gamma_dk = _gamma_dk

    def superlink_flow_coefficients(self):
        # Import instance variables
        _I_1k = self._I_1k
        _I_Nk = self._I_Nk
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _gamma_uk = self._gamma_uk
        _gamma_dk = self._gamma_dk
        _z_inv_uk = self._z_inv_uk
        _z_inv_dk = self._z_inv_dk
        # Compute D_k_star
        _D_k_star = self.D_k_star(_X_Ik[_I_1k], _gamma_uk, _U_Ik[_I_Nk],
                                  _gamma_dk, _Z_Ik[_I_1k], _W_Ik[_I_Nk])
        # Compute upstream superlink flow coefficients
        _alpha_uk = self.alpha_uk(_U_Ik[_I_Nk], _gamma_dk, _X_Ik[_I_1k],
                                  _Z_Ik[_I_1k], _W_Ik[_I_Nk], _D_k_star)
        _beta_uk = self.beta_uk(_U_Ik[_I_Nk], _gamma_dk, _Z_Ik[_I_1k],
                                _W_Ik[_I_Nk], _D_k_star)
        _chi_uk = self.chi_uk(_U_Ik[_I_Nk], _gamma_dk, _Y_Ik[_I_1k],
                              _X_Ik[_I_1k], _z_inv_uk, _Z_Ik[_I_1k],
                              _z_inv_dk, _V_Ik[_I_Nk], _W_Ik[_I_Nk],
                              _D_k_star)
        # Compute downstream superlink flow coefficients
        _alpha_dk = self.alpha_dk(_X_Ik[_I_1k], _gamma_uk, _W_Ik[_I_Nk],
                                  _D_k_star)
        _beta_dk = self.beta_dk(_X_Ik[_I_1k], _gamma_uk, _U_Ik[_I_Nk],
                                _W_Ik[_I_Nk], _Z_Ik[_I_1k], _D_k_star)
        _chi_dk = self.chi_dk(_X_Ik[_I_1k], _gamma_uk, _V_Ik[_I_Nk],
                              _W_Ik[_I_Nk], _z_inv_uk, _U_Ik[_I_Nk],
                              _z_inv_dk, _Y_Ik[_I_1k], _Z_Ik[_I_1k],
                              _D_k_star)
        # Export instance variables
        self._D_k_star = _D_k_star
        self._alpha_uk = _alpha_uk
        self._beta_uk = _beta_uk
        self._chi_uk = _chi_uk
        self._alpha_dk = _alpha_dk
        self._beta_dk = _beta_dk
        self._chi_dk = _chi_dk

    def sparse_matrix_equations(self, H_bc=None, _Q_0j=0, _dt=None, first_time=False):
        # TODO: May want to consider reconstructing A each time while debugging
        # Import instance variables
        _k = self._k
        _J_uk = self._J_uk
        _J_dk = self._J_dk
        _alpha_uk = self._alpha_uk
        _alpha_dk = self._alpha_dk
        _beta_uk = self._beta_uk
        _beta_dk = self._beta_dk
        _chi_uk = self._chi_uk
        _chi_dk = self._chi_dk
        _alpha_ukm = self._alpha_ukm
        _beta_dkl = self._beta_dkl
        _chi_ukl = self._chi_ukl
        _chi_dkm = self._chi_dkm
        _A_sj = self._A_sj
        M = self.M
        H_j = self.H_j
        bc = self.bc
        if _dt is None:
            _dt = self._dt
        if H_bc is None:
            H_bc = self.H_j
        # Compute F_jj
        _alpha_ukm_J = pd.Series(_alpha_uk, index=_J_uk).groupby(level=0).sum()
        _beta_dkl_J = pd.Series(_beta_dk, index=_J_dk).groupby(level=0).sum()
        _alpha_ukm[_alpha_ukm_J.index.values] = _alpha_ukm_J.values
        _beta_dkl[_beta_dkl_J.index.values] = _beta_dkl_J.values
        _F_jj = self.F_jj(_A_sj, _dt, _beta_dkl, _alpha_ukm)
        # Set diagonals
        i = np.arange(M)
        self.A[i[~bc], i[~bc]] = _F_jj[i[~bc]]
        self.A[i[bc], i[bc]] = 1
        # Compute off-diagonals
        bc_uk = bc[_J_uk]
        bc_dk = bc[_J_dk]
        self.A[_J_uk[~bc_uk], _J_dk[~bc_uk]] = _beta_uk[~bc_uk]
        self.A[_J_dk[~bc_dk], _J_uk[~bc_dk]] = -_alpha_dk[~bc_dk]
        # Compute G_j
        _chi_ukl_J = pd.Series(_chi_uk, index=_J_uk).groupby(level=0).sum()
        _chi_dkm_J = pd.Series(_chi_dk, index=_J_dk).groupby(level=0).sum()
        _chi_ukl[_chi_ukl_J.index.values] = _chi_ukl_J.values
        _chi_dkm[_chi_dkm_J.index.values] = _chi_dkm_J.values
        b = self.G_j(_A_sj, _dt, H_j, _Q_0j, _chi_ukl, _chi_dkm)
        b[bc] = H_bc[bc]
        # Export instance variables
        self.b = b
        self._beta_dkl = _beta_dkl
        self._alpha_ukm = _alpha_ukm
        self._chi_ukl = _chi_ukl
        self._chi_dkm = _chi_dkm
        if first_time:
            self.A = self.A.tocsr()

    def solve_sparse_matrix(self):
        A = self.A
        b = self.b
        _z_inv_j = self._z_inv_j
        min_depth = self.min_depth
        H_j_next = scipy.sparse.linalg.spsolve(A, b)
        # H_j_next = np.maximum(H_j_next, _z_inv_j + min_depth)
        H_j_next = np.maximum(H_j_next, _z_inv_j)
        self.H_j = H_j_next

    def solve_superlink_flows(self):
        # Import instance variables
        _J_uk = self._J_uk
        _J_dk = self._J_dk
        _alpha_uk = self._alpha_uk
        _alpha_dk = self._alpha_dk
        _beta_uk = self._beta_uk
        _beta_dk = self._beta_dk
        _chi_uk = self._chi_uk
        _chi_dk = self._chi_dk
        H_j = self.H_j
        # Compute flow at next time step
        _Q_uk_next = _alpha_uk * H_j[_J_uk] + _beta_uk * H_j[_J_dk] + _chi_uk
        _Q_dk_next = _alpha_dk * H_j[_J_uk] + _beta_dk * H_j[_J_dk] + _chi_dk
        # Export instance variables
        self._Q_uk = _Q_uk_next
        self._Q_dk = _Q_dk_next

    def solve_superlink_depths(self):
        # Import instance variables
        _J_uk = self._J_uk
        _J_dk = self._J_dk
        _gamma_uk = self._gamma_uk
        _gamma_dk = self._gamma_dk
        _z_inv_uk = self._z_inv_uk
        _z_inv_dk = self._z_inv_dk
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        H_j = self.H_j
        # Compute flow at next time step
        _h_uk_next = _gamma_uk * _Q_uk + H_j[_J_uk] - _z_inv_uk
        _h_dk_next = _gamma_dk * _Q_dk + H_j[_J_dk] - _z_inv_dk
        # Export instance variables
        self._h_uk = _h_uk_next
        self._h_dk = _h_dk_next

    def solve_superlink_depths_alt(self):
        # Import instance variables
        _I_1k = self._I_1k
        _I_Nk = self._I_Nk
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        H_j = self.H_j
        # Compute flow at next time step
        det = (_X_Ik[_I_1k] * _U_Ik[_I_Nk]) - (_Z_Ik[_I_1k] * _W_Ik[_I_Nk])
        _h_uk_next = (_U_Ik[_I_Nk] * (_Q_uk - _Y_Ik[_I_1k])
                      - _Z_Ik[_I_1k] * (_Q_dk - _V_Ik[_I_Nk])) / det
        _h_dk_next = (-_W_Ik[_I_Nk] * (_Q_uk - _Y_Ik[_I_1k])
                      + _X_Ik[_I_1k] * (_Q_dk - _V_Ik[_I_Nk])) / det
        # Export instance variables
        self._h_uk = _h_uk_next
        self._h_dk = _h_dk_next

    def solve_internals(self):
        # Import instance variables
        _I_1k = self._I_1k
        _I_2k = self._I_2k
        _I_Nk = self._I_Nk
        _I_Np1k = self._I_Np1k
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _I_end = self._I_end
        forward_I_I = self.forward_I_I
        forward_I_i = self.forward_I_i
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        # Set first elements
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Get rid of superlinks with one link
        keep = (_I_2k != _I_Np1k)
        _Im1_next = _I_1k[keep]
        _I_next = _I_2k[keep]
        _I_1k_next = _I_1k[keep]
        _I_Nk_next = _I_Nk[keep]
        _I_Np1k_next = _I_Np1k[keep]
        # Get rid of superlinks with two_links
        keep = (_I_next != _I_Nk_next)
        _Im1_next = _I_1k_next[keep]
        _I_next = _I_next[keep]
        _I_1k_next = _I_1k_next[keep]
        _I_Nk_next = _I_Nk_next[keep]
        _I_Np1k_next = _I_Np1k_next[keep]
        _im1_next = forward_I_i[_Im1_next].values
        # Loop from 1 -> Nk
        while _I_next.size:
            _i_next = forward_I_i[_I_next].values
            _Ip1_next = forward_I_I[_I_next].values
            _h_Ik[_I_next] = (_Q_ik[_im1_next] - _V_Ik[_Im1_next]
                              - _W_Ik[_Im1_next] * _h_Ik[_I_1k_next]) / _U_Ik[_Im1_next]
            _Q_ik[_i_next] = (_X_Ik[_I_next] * _h_Ik[_I_next] + _Y_Ik[_I_next]
                                + _Z_Ik[_I_next] * _h_Ik[_I_Np1k_next])
            keep = (_Ip1_next != _I_Nk_next)
            _Im1_next = _I_next[keep]
            _im1_next = _i_next[keep]
            _I_next = _Ip1_next[keep]
            _I_1k_next = _I_1k_next[keep]
            _I_Nk_next = _I_Nk_next[keep]
            _I_Np1k_next = _I_Np1k_next[keep]
        # TODO: This will overwrite superlinks with one link
        _h_Ik[_I_Nk] = (_Q_ik[_i_nk] - _Y_Ik[_I_Nk] -
                        _Z_Ik[_I_Nk] * _h_Ik[_I_Np1k]) / _X_Ik[_I_Nk]
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    def solve_internals_backwards(self):
        # Import instance variables
        _I_1k = self._I_1k
        _I_2k = self._I_2k
        _I_Nk = self._I_Nk
        _I_Np1k = self._I_Np1k
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _I_end = self._I_end
        backward_I_I = self.backward_I_I
        forward_I_i = self.forward_I_i
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        min_depth = self.min_depth
        # Set first elements
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Get rid of superlinks with one link
        keep = (_I_1k != _I_Nk)
        _Im1_next = _I_1k[keep]
        _I_next = _I_Nk[keep]
        _I_1k_next = _I_1k[keep]
        _I_2k_next = _I_2k[keep]
        _I_Np1k_next = _I_Np1k[keep]
        # Loop from Nk -> 1
        while _I_next.size:
            _i_next = forward_I_i[_I_next].values
            _Im1_next = backward_I_I[_I_next].values
            _im1_next = forward_I_i[_Im1_next].values
            _h_Ik[_I_next] = self._h_Ik_next(_Q_ik[_i_next], _Y_Ik[_I_next],
                                             _Z_Ik[_I_next], _h_Ik[_I_Np1k_next],
                                             _X_Ik[_I_next])
            # Ensure non-negative depths?
            # _h_Ik[_I_next[_h_Ik[_I_next] < min_depth]] = min_depth
            _Q_ik[_im1_next] = self._Q_im1k_next(_U_Ik[_Im1_next], _h_Ik[_I_next],
                                                 _V_Ik[_Im1_next], _W_Ik[_Im1_next],
                                                 _h_Ik[_I_1k_next])
            keep = (_Im1_next != _I_1k_next)
            _I_next = _Im1_next[keep]
            _I_1k_next = _I_1k_next[keep]
            _I_Np1k_next = _I_Np1k_next[keep]
        # Set upstream flow
        # TODO: May want to delete where this is set earlier
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Ensure non-negative depths?
        # _h_Ik[_h_Ik < min_depth] = min_depth
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    @safe_divide
    def _h_Ik_next(self, Q_ik, Y_Ik, Z_Ik, h_Np1k, X_Ik):
        num = Q_ik - Y_Ik - Z_Ik * h_Np1k
        den = X_Ik
        return num, den

    def _Q_im1k_next(self, U_Im1k, h_Ik, V_Im1k, W_Im1k, h_1k):
        t_0 = U_Im1k * h_Ik
        t_1 = V_Im1k
        t_2 = W_Im1k * h_1k
        return t_0 + t_1 + t_2

    def step(self, H_bc=None, _dt=None, first_time=False):
        self.node_velocities()
        self.link_coeffs(_dt=_dt)
        self.node_coeffs(_dt=_dt)
        self.forward_recurrence()
        self.backward_recurrence()
        self.superlink_upstream_head_coefficients()
        self.superlink_downstream_head_coefficients()
        self.superlink_flow_coefficients()
        self.sparse_matrix_equations(H_bc=H_bc, first_time=first_time, _dt=_dt)
        self.solve_sparse_matrix()
        self.solve_superlink_flows()
        self.solve_superlink_depths()
        self.solve_internals_backwards()
