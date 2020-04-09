import numpy as np
import pandas as pd
from numba import njit

class QualityBuilder():
    def __init__(hydraulics, quality_params):
        self.hydraulics = hydraulics
        self._D_ik = quality_params['D_ik'].values.astype(float)
        self._K_ik = quality_params['K_ik'].values.astype(float)
        self._c_ik = quality_params['c0_ik'].values.astype(float)
        self.import_hydraulics()

    def import_hydraulics():
        self.H_j_next = self.hydraulics.H_j
        self.h_Ik_next = self.hydraulics.h_Ik
        self.Q_ik_next = self.hydraulics.Q_ik
        self.Q_uk_next = self.hydraulics.Q_uk
        self.Q_dk_next = self.hydraulics.Q_dk
        self.H_j_prev = self.hydraulics['states']['H_j']
        self.h_Ik_prev = self.hydraulics['states']['h_Ik']
        self.Q_ik_prev = self.hydraulics['states']['Q_ik']
        self.Q_uk_prev = self.hydraulics['states']['Q_uk']
        self.Q_dk_prev = self.hydraulics['states']['Q_dk']
        # TODO: Don't forget control structures
        self.u_ik = self.hydraulics.u_ik
        self.u_Ik = self.hydraulics.u_Ik
        # TODO
        self.dx_ik = self.hydraulics._dx_ik

    def link_coeffs(self, _dt=None, first_iter=True):
        """
        Compute link momentum coefficients: a_ik, b_ik, c_ik and P_ik.
        """
        # Import instance variables
        _u_Ik = self._u_Ik         # Flow velocity at junction Ik
        _u_Ip1k = self._u_Ip1k     # Flow velocity at junction I + 1k
        _dx_ik = self._dx_ik       # Length of link ik
        _D_ik = self._D_ik
        _K_ik = self._K_ik
        _c_ik = self._c_ik
        _alpha_ik = self._alpha_ik
        _beta_ik = self._beta_ik
        _chi_ik = self._chi_ik
        _gamma_ik = self._gamma_ik
        # If time step not specified, use instance time
        if _dt is None:
            _dt = self._dt
        # Compute link coefficients
        _alpha_ik = alpha_ik(_u_Ik, _dx_ik, _D_ik)
        _beta_ik = beta_ik(_dt, _D_ik, _dx_ik, _K_ik)
        _chi_ik = chi_ik(_u_Ip1k, _dx_ik, _D_ik)
        _gamma_ik = gamma_ik(_dt, _c_ik)
        # Export to instance variables
        self._alpha_ik = _alpha_ik
        self._beta_ik = _beta_ik
        self._chi_ik = _chi_ik
        self._gamma_ik = _gamma_ik

    def node_coeffs(self, _Q_0Ik=None, _c_0_Ik=None, _dt=None, first_iter=True):
        """
        Compute nodal continuity coefficients: D_Ik and E_Ik.
        """
        # Import instance variables
        forward_I_i = self.forward_I_i       # Index of link after junction Ik
        backward_I_i = self.backward_I_i     # Index of link before junction Ik
        _is_start = self._is_start
        _is_end = self._is_end
        _kI = self._kI
        _A_SIk = self._A_SIk                 # Surface area of junction Ik
        _h_Ik_next = self._h_Ik_next         # Depth at junction Ik
        _Q_ik_next = self._Q_ik_next
        _h_Ik_prev = self._h_Ik_prev
        _Q_uk_next = self._Q_uk_next
        _Q_dk_next = self._Q_dk_next
        _c_Ik_prev = self._c_Ik
        _K_Ik = self._K_Ik
        _kappa_Ik = self._kappa_Ik
        _lambda_Ik = self._lambda_Ik
        _mu_Ik = self._mu_Ik
        _eta_Ik = self._eta_Ik
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no nodal input specified, use zero input
        if _Q_0Ik is None:
            _Q_0Ik = np.zeros(_h_Ik.size)
        if _c_0Ik is None:
            _c_0Ik = np.zeros(_h_Ik.size)
        # Compute continuity coefficients
        numba_node_coeffs(_kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                          _Q_ik_next, _h_Ik_next, _h_Ik_prev, _c_Ik_prev,
                          _Q_uk_next, _Q_dk_next, _c_0Ik, _Q_0Ik, _A_SIk, _K_Ik,
                          _forward_I_i, _backward_I_i, _kI, _dt)
        # Export instance variables
        self._kappa_Ik = _kappa_Ik
        self._lambda_Ik = _lambda_Ik
        self._mu_Ik = _mu_Ik
        self._eta_Ik = _eta_Ik

    def forward_recurrence(self):
        """
        Compute forward recurrence coefficients: T_ik, U_Ik, V_Ik, and W_Ik.
        """
        # Import instance variables
        _I_1k = self._I_1k                # Index of first junction in each superlink
        _i_1k = self._i_1k                # Index of first link in each superlink
        _alpha_ik = self._alpha_ik
        _beta_ik = self._beta_ik
        _chi_ik = self._chi_ik
        _gamma_ik = self._gamma_ik
        _kappa_Ik = self._kappa_Ik
        _lambda_Ik = self._lambda_Ik
        _mu_Ik = self._mu_Ik
        _eta_Ik = self._eta_Ik
        _T_ik = self._T_ik                # Recurrence coefficient T_ik
        _U_Ik = self._U_Ik                # Recurrence coefficient U_Ik
        _V_Ik = self._V_Ik                # Recurrence coefficient V_Ik
        _W_Ik = self._W_Ik                # Recurrence coefficient W_Ik
        NK = self.NK
        nk = self.nk
        numba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _alpha_ik, _beta_ik, _chi_ik,
                                 _gamma_ik, _kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                                 NK, nk, _I_1k, _i_1k)
        # Export instance variables
        self._T_ik = _T_ik
        self._U_Ik = _U_Ik
        self._V_Ik = _V_Ik
        self._W_Ik = _W_Ik

    def backward_recurrence(self):
        """
        Compute backward recurrence coefficients: O_ik, X_Ik, Y_Ik, and Z_Ik.
        """
        _I_Nk = self._I_Nk                # Index of penultimate junction in each superlink
        _i_nk = self._i_nk                # Index of last link in each superlink
        _A_ik = self._A_ik                # Flow area in link ik
        _alpha_ik = self._alpha_ik
        _beta_ik = self._beta_ik
        _chi_ik = self._chi_ik
        _gamma_ik = self._gamma_ik
        _kappa_Ik = self._kappa_Ik
        _lambda_Ik = self._lambda_Ik
        _mu_Ik = self._mu_Ik
        _eta_Ik = self._eta_Ik
        _O_ik = self._O_ik                # Recurrence coefficient O_ik
        _X_Ik = self._X_Ik                # Recurrence coefficient X_Ik
        _Y_Ik = self._Y_Ik                # Recurrence coefficient Y_Ik
        _Z_Ik = self._Z_Ik                # Recurrence coefficient Z_Ik
        NK = self.NK
        nk = self.nk
        numba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _alpha_ik, _beta_ik, _chi_ik,
                                  _gamma_ik, _kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                                  NK, nk, _I_Nk, _i_nk)
        # Export instance variables
        self._O_ik = _O_ik
        self._X_Ik = _X_Ik
        self._Y_Ik = _Y_Ik
        self._Z_Ik = _Z_Ik

    def boundary_coefficients(self):
        _T_ik = self._T_ik                # Recurrence coefficient T_ik
        _U_Ik = self._U_Ik                # Recurrence coefficient U_Ik
        _V_Ik = self._V_Ik                # Recurrence coefficient V_Ik
        _W_Ik = self._W_Ik                # Recurrence coefficient W_Ik
        _O_ik = self._O_ik                # Recurrence coefficient O_ik
        _X_Ik = self._X_Ik                # Recurrence coefficient X_Ik
        _Y_Ik = self._Y_Ik                # Recurrence coefficient Y_Ik
        _Z_Ik = self._Z_Ik                # Recurrence coefficient Z_Ik
        _I_1k = self._I_1k                # Index of first junction in each superlink
        _i_1k = self._i_1k                # Index of first link in each superlink
        _I_Nk = self._I_Nk                # Index of penultimate junction in each superlink
        _i_nk = self._i_nk                # Index of last link in each superlink
        _X_uk = self._X_uk
        _Y_uk = self._Y_uk
        _Z_uk = self._Z_uk
        _U_dk = self._U_dk
        _V_dk = self._V_dk
        _W_dk = self._W_dk
        # Compute boundary coefficients
        numba_boundary_coefficients(_X_uk, _Y_uk, _Z_uk, _U_dk, _V_dk, _W_dk,
                                    _kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                                    NK, _I_1k, _I_Nk)
        # Export instance variables
        self._X_uk = _X_uk
        self._Y_uk = _Y_uk
        self._Z_uk = _Z_uk
        self._U_dk = _U_dk
        self._V_dk = _V_dk
        self._W_dk = _W_dk

@njit
def safe_divide(num, den):
    if (den == 0):
        return 0
    else:
        return num / den

@njit
def safe_divide_vec(num, den):
    result = np.zeros_like(num)
    cond = (den != 0)
    result[cond] = num[cond] / den[cond]
    return result

@njit
def alpha_ik(u_Ik, dx_ik, D_ik):
    t_0 = - u_Ik / dx_ik
    t_1 = - 2 * D_ik / (dx_ik**2)
    return t_0 + t_1

@njit
def beta_ik(dt, D_ik, dx_ik, K_ik):
    t_0 = 1 / dt
    t_1 = 4 * D_ik / (dx_ik**2)
    t_2 = - K_ik
    return t_0 + t_1 + t_2

@njit
def chi_ik(u_Ip1k, dx_ik, D_ik):
    t_0 = u_Ip1k / dx_ik
    t_1 = - 2 * D_ik / (dx_ik**2)
    return t_0 + t_1

@njit
def gamma_ik(dt, c_ik_prev):
    t_0 = c_ik_prev / dt
    return t_0

@njit
def kappa_Ik(Q_im1k_next):
    t_0 = - Q_im1k_next
    return t_0

@njit
def lambda_Ik(A_SIk, h_Ik_next, dt, K_Ik):
    t_0 = A_SIk * h_Ik_next / dt
    t_1 = K_Ik
    return t_0 + t_1

@njit
def mu_Ik(Q_ik_next):
    t_0 = Q_ik_next
    return t_0

@njit
def eta_Ik(c_0_Ik, Q_0_Ik, A_SIk, h_Ik_prev, c_Ik_prev, dt):
    t_0 = c_0_Ik * Q_0_Ik
    t_1 = A_SIk * h_Ik_prev * c_Ik_prev / dt
    return t_0 + t_1

@njit
def U_1k(chi_1k, beta_1k):
    return safe_divide(-chi_1k, beta_1k)

@njit
def V_1k(gamma_1k, beta_1k):
    return safe_divide(gamma_1k, beta_1k)

@njit
def W_1k(alpha_1k, beta_1k):
    return safe_divide(-alpha_1k, beta_1k)

@njit
def U_Ik(alpha_ik, kappa_Ik, W_Im1k, T_ik, lambda_Ik, kappa_Ik, U_Im1k):
    t_0 = alpha_ik * kappa_Ik * W_Im1k
    t_1 = T_ik * (lambda_Ik + kappa_Ik * U_Im1k)
    return safe_divide(t_0, t_1)

@njit
def V_Ik(gamma_ik, T_ik, alpha_ik, eta_Ik, kappa_Ik, V_Im1k, lambda_Ik, U_Im1k):
    t_0 = safe_divide(gamma_ik, T_ik)
    t_1 = - alpha_ik * (eta_Ik + kappa_Ik * V_Im1k)
    # TODO: Note that this denominator is being computed 3 times
    t_2 = T_ik * (lambda_Ik + kappa_Ik * U_Im1k)
    return t_0 + safe_divide(t_1, t_2)

@njit
def W_Ik(chi_ik, T_ik):
    return safe_divide(-chi_ik, T_ik)

@njit
def T_ik(beta_ik, alpha_ik, mu_Ik, lambda_Ik, kappa_Ik, U_Im1k):
    t_0 = beta_ik
    t_1 = - alpha_ik * mu_Ik
    t_2 = lambda_Ik + kappa_Ik * U_Im1k
    return t_0 + safe_divide(t_1, t_2)

@njit
def X_Nk(alpha_nk, beta_nk):
    return safe_divide(-alpha_nk, beta_nk)

@njit
def Y_Nk(gamma_nk, beta_nk):
    return safe_divide(gamma_nk, beta_nk)

@njit
def Z_Nk(chi_nk, beta_nk):
    return safe_divide(-chi_nk, beta_nk)

@njit
def X_Ik(alpha_ik, O_ik):
    return safe_divide(-alpha_ik, O_ik)

@njit
def Y_Ik(gamma_ik, O_ik, chi_ik, eta_Ip1k, mu_Ip1k, Y_Ip1k, lambda_Ip1k, X_Ip1k):
    t_0 = safe_divide(gamma_ik, O_ik)
    t_1 = - chi_ik * (eta_Ip1k - mu_Ip1k * Y_Ip1k)
    t_2 = O_ik * (lambda_Ip1k + mu_Ip1k * X_Ip1k)
    return t_0 + safe_divide(t_1, t_2)

@njit
def Z_Ik(chi_ik, mu_Ip1k, Z_Ip1k, O_ik, lambda_Ip1k, X_Ip1k):
    t_0 = chi_ik * mu_Ip1k * Z_Ip1k
    t_1 = O_ik * (lambda_Ip1k + mu_Ip1k * X_Ip1k)
    return safe_divide(t_0, t_1)

@njit
def O_ik(beta_ik, chi_ik, kappa_Ip1k, lambda_Ip1k, mu_Ip1k, X_Ip1k):
    t_0 = beta_ik
    t_1 = - chi_ik * kappa_Ip1k
    # TODO: Note that this denominator is being computed 3 times
    t_2 = lambda_Ip1k + mu_Ip1k * X_Ip1k
    return t_0 + safe_divide(t_1, t_2)

@njit
def X_uk(mu_1k, X_1k, lambda_1k, kappa_1k):
    t_0 = - mu_1k * X_1k - lambda_1k
    t_1 = kappa_1k
    return safe_divide(t_0, t_1)

@njit
def Y_uk(eta_1k, mu_1k, Y_1k, kappa_1k):
    t_0 = eta_1k - mu_1k * Y_1k
    t_1 = kappa_1k
    return safe_divide(t_0, t_1)

@njit
def Z_uk(mu_1k, Z_1k, kappa_1k):
    t_0 = -mu_1k * Z_1k
    t_1 = kappa_1k
    return safe_divide(t_0, t_1)

@njit
def U_dk(kappa_Np1k, U_Nk, lambda_Np1k, mu_Np1k):
    t_0 = - kappa_Np1k * U_Nk - lambda_Np1k
    t_1 = mu_Np1k
    return safe_divide(t_0, t_1)

@njit
def V_dk(eta_Np1k, kappa_Np1k, V_Nk, mu_Np1k):
    t_0 = eta_Np1k - kappa_Np1k * V_Nk
    t_1 = mu_Np1k
    return safe_divide(t_0, t_1)

@njit
def W_dk(kappa_Np1k, W_Nk, mu_Np1k):
    t_0 = - kappa_Np1k * W_Nk
    t_1 = mu_Np1k
    return safe_divide(t_0, t_1)

@njit
def numba_node_coeffs(Q_ik_next, h_Ik_next, h_Ik_prev, c_Ik_prev,
                      Q_uk_next, Q_dk_next, c_0_Ik, Q_0_Ik, A_SIk, K_Ik,
                      _forward_I_i, _backward_I_i, _kI, dt):
    N = _kI.size
    for I in range(N):
        if _is_start[I]:
            i = _forward_I_i[I]
            k = _kI[I]
            _kappa_Ik[I] = kappa_Ik(Q_uk_next[k])
            _lambda_Ik[I] = lambda_Ik(A_SIk[I], h_Ik_next[I], dt, K_Ik[I])
            _mu_Ik[I] = mu_Ik(Q_ik_next[i])
            _eta_Ik[I] = eta_Ik(c_0_Ik[I], Q_0_Ik[I], A_SIk[I],
                                h_Ik_prev[I], c_Ik_prev[I], dt)
        elif _is_end[I]:
            im1 = _backward_I_i[I]
            k = _kI[I]
            _kappa_Ik[I] = kappa_Ik(Q_ik_next[im1])
            _lambda_Ik[I] = lambda_Ik(A_SIk[I], h_Ik_next[I], dt, K_Ik[I])
            _mu_Ik[I] = mu_Ik(Q_dk_next[k])
            _eta_Ik[I] = eta_Ik(c_0_Ik[I], Q_0_Ik[I], A_SIk[I],
                                h_Ik_prev[I], c_Ik_prev[I], dt)
        else:
            i = _forward_I_i[I]
            im1 = i - 1
            _kappa_Ik[I] = kappa_Ik(Q_ik_next[im1])
            _lambda_Ik[I] = lambda_Ik(A_SIk[I], h_Ik_next[I], dt, K_Ik[I])
            _mu_Ik[I] = mu_Ik(Q_ik_next[i])
            _eta_Ik[I] = eta_Ik(c_0_Ik[I], Q_0_Ik[I], A_SIk[I],
                                h_Ik_prev[I], c_Ik_prev[I], dt)
    return 1

@njit
def numba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _alpha_ik, _beta_ik, _chi_ik,
                             _gamma_ik, _kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                             NK, nk, _I_1k, _i_1k):
    for k in range(NK):
        # Start at junction 1
        _I_1 = _I_1k[k]
        _i_1 = _i_1k[k]
        _I_2 = _I_1 + 1
        _i_2 = _i_1 + 1
        nlinks = nk[k]
        _T_ik[_i_1] = 1.
        _U_Ik[_I_1] = U_1k(_chi_ik[_i_1], _beta_ik[_i_1])
        _V_Ik[_I_1] = V_1k(_gamma_ik[_i_1], _beta_ik[_i_1])
        _W_Ik[_I_1] = W_1k(_alpha_ik[_i_1], _beta_ik[_i_1])
        # Loop from junction 2 -> Nk
        for i in range(nlinks - 1):
            _i_next = _i_2 + i
            _I_next = _I_2 + i
            _Im1_next = _I_next - 1
            _Ip1_next = _I_next + 1
            _T_ik[_i_next] = T_ik(_beta_ik[_i_next], _alpha_ik[_i_next],
                                  _mu_Ik[_I_next], _lambda_Ik[_I_next], _kappa_Ik[_I_next],
                                  _U_Ik[_Im1_next])
            _U_Ik[_I_next] = U_Ik(_alpha_ik[_i_next], _kappa_Ik[_I_next], _W_Ik[_Im1_next],
                                  _T_ik[_i_next], _lambda_Ik[_I_next], _kappa_Ik[_I_next],
                                  _U_Ik[_Im1_next])
            _V_Ik[_I_next] = V_Ik(_gamma_ik[_i_next], _T_ik[_i_next], _alpha_ik[_i_next],
                                  _eta_Ik[_I_next], _kappa_Ik[_I_next], _V_Ik[_Im1_next],
                                  _lambda_Ik[_I_next], _U_Ik[_Im1_next])
            _W_Ik[_I_next] = W_Ik(_chi_ik[_i_next], _T_ik[_i_next])
    return 1

@njit
def numba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _alpha_ik, _beta_ik, _chi_ik,
                              _gamma_ik, _kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                              NK, nk, _I_Nk, _i_nk):
    for k in range(NK):
        _I_N = _I_Nk[k]
        _i_n = _i_nk[k]
        _I_Nm1 = _I_N - 1
        _i_nm1 = _i_n - 1
        _I_Np1 = _I_N + 1
        nlinks = nk[k]
        _O_ik[_i_n] = 1.
        _X_Ik[_I_N] = X_Nk(_alpha_ik[_i_n], _beta_ik[_i_n])
        _Y_Ik[_I_N] = Y_Nk(_gamma_ik[_i_n], _beta_ik[_i_n])
        _Z_Ik[_I_N] = Z_Nk(_chi_ik[_i_n], _beta_ik[_i_n])
        for i in range(nlinks - 1):
            _i_next = _i_nm1 - i
            _I_next = _I_Nm1 - i
            _Ip1_next = _I_next + 1
            _O_ik[_i_next] = O_ik(_beta_ik[_i_next], _chi_ik[_i_next], _kappa_Ik[_Ip1_next],
                                  _lambda_Ik[_Ip1_next], _mu_Ik[_Ip1_next], _X_Ik[_Ip1_next])
            _X_Ik[_I_next] = X_Ik(_alpha_ik[_i_next], _O_ik[_i_next])
            _Y_Ik[_I_next] = Y_Ik(_gamma_ik[_i_next], _O_ik[_i_next], _chi_ik[_i_next],
                                  _eta_Ik[_Ip1_next], _mu_Ik[_Ip1_next], _Y_Ik[_Ip1_next],
                                  _lambda_Ik[_Ip1_next], _X_Ik[_Ip1_next])
            _Z_Ik[_I_next] = Z_Ik(_chi_ik[_i_next], _mu_Ik[_Ip1_next], _Z_Ik[_Ip1_next],
                                  _O_ik[_i_next], _lambda_Ik[_Ip1_next], _X_Ik[_Ip1_next])
    return 1

@njit
def numba_boundary_coefficients(_X_uk, _Y_uk, _Z_uk, _U_dk, _V_dk, _W_dk,
                                _kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                                NK, _I_1k, _I_Nk):
    for k in range(NK):
        _I_1 = _I_1k[k]
        _I_N = _I_Nk[k]
        _I_Np1 = _I_N + 1
        _X_uk[k] = X_uk(_mu_Ik[_I_1], _X_Ik[_I_1], _lambda_Ik[_I_1], _kappa_Ik[_I_1])
        _Y_uk[k] = Y_uk(_eta_Ik[_I_1], _mu_Ik[_I_1], _Y_Ik[_I_1], _kappa_Ik[_I_1])
        _Z_uk[k] = Z_uk(_mu_Ik[_I_1], _Z_Ik[_I_1], _kappa_Ik[_I_1])
        _U_dk[k] = U_dk(_kappa_Ik[_I_Np1], _U_Ik[_I_N], _lambda_Ik[_I_Np1], _mu_Ik[_I_Np1])
        _V_dk[k] = V_dk(_eta_Ik[_I_Np1], _kappa_Ik[_I_Np1], _V_Ik[_I_N], _mu_Ik[_I_Np1])
        _W_dk[k] = W_dk(_kappa_Ik[_I_Np1], _W_Ik[_I_N], _mu_Ik[_I_Np1])
    return 1

