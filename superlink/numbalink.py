import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
from numba import njit, prange
import superlink.geometry
import superlink.storage
from superlink.superlink import SuperLink

class NumbaLink(SuperLink):
    def __init__(self, superlinks, superjunctions,
                 links=None, junctions=None,
                 transects={}, storages={},
                 orifices=None, weirs=None, pumps=None,
                 dt=60, sparse=False, min_depth=1e-5, method='b',
                 inertial_damping=False, bc_method='z',
                 exit_hydraulics=False, end_length=None,
                 end_method='o'):
        super().__init__(superlinks, superjunctions,
                         links, junctions, transects, storages,
                         orifices, weirs, pumps, dt, sparse,
                         min_depth, method, inertial_damping,
                         bc_method, exit_hydraulics, end_length,
                         end_method)

    def forward_recurrence(self):
        """
        Compute forward recurrence coefficients: T_ik, U_Ik, V_Ik, and W_Ik.
        """
        # Import instance variables
        _I_1k = self._I_1k                # Index of first junction in each superlink
        _i_1k = self._i_1k                # Index of first link in each superlink
        _A_ik = self._A_ik                # Flow area in link ik
        _E_Ik = self._E_Ik                # Continuity coefficient E_Ik
        _D_Ik = self._D_Ik                # Continuity coefficient D_Ik
        _a_ik = self._a_ik                # Momentum coefficient a_ik
        _b_ik = self._b_ik                # Momentum coefficient b_ik
        _c_ik = self._c_ik                # Momentum coefficient c_ik
        _P_ik = self._P_ik                # Momentum coefficient P_ik
        _T_ik = self._T_ik                # Recurrence coefficient T_ik
        _U_Ik = self._U_Ik                # Recurrence coefficient U_Ik
        _V_Ik = self._V_Ik                # Recurrence coefficient V_Ik
        _W_Ik = self._W_Ik                # Recurrence coefficient W_Ik
        NK = self.NK
        nk = self.nk
        numba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _a_ik, _b_ik, _c_ik,
                                 _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_1k, _i_1k)
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
        _E_Ik = self._E_Ik                # Continuity coefficient E_Ik
        _D_Ik = self._D_Ik                # Continuity coefficient D_Ik
        _a_ik = self._a_ik                # Momentum coefficient a_ik
        _b_ik = self._b_ik                # Momentum coefficient b_ik
        _c_ik = self._c_ik                # Momentum coefficient c_ik
        _P_ik = self._P_ik                # Momentum coefficient P_ik
        _O_ik = self._O_ik                # Recurrence coefficient O_ik
        _X_Ik = self._X_Ik                # Recurrence coefficient X_Ik
        _Y_Ik = self._Y_Ik                # Recurrence coefficient Y_Ik
        _Z_Ik = self._Z_Ik                # Recurrence coefficient Z_Ik
        NK = self.NK
        nk = self.nk
        numba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _a_ik, _b_ik, _c_ik,
                                    _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_Nk, _i_nk)
        # Export instance variables
        self._O_ik = _O_ik
        self._X_Ik = _X_Ik
        self._Y_Ik = _Y_Ik
        self._Z_Ik = _Z_Ik

    def solve_internals_ls_alt(self):
        NK = self.NK
        nk = self.nk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _kI = self._kI
        _ki = self._ki
        _i_1k = self._i_1k
        _I_1k = self._I_1k
        _k_1k = self._k_1k
        _I_Nk = self._I_Nk
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _is_start = self._is_start
        _is_end = self._is_end
        _is_penult = self._is_penult
        _link_start = self._link_start
        _link_end = self._link_end
        _kk = _ki[~_link_end]
        min_depth = self.min_depth
        # Solve non-negative least squares
        _X = _X_Ik[~_is_start & ~_is_end]
        _U = _U_Ik[~_is_penult & ~_is_end]
        t0 = _W_Ik[~_is_end] * _h_uk[_ki]
        t1 = _Z_Ik[~_is_end] * _h_dk[_ki]
        t2 = _Y_Ik[~_is_end]
        t3 = _V_Ik[~_is_end]
        _b = -t0 + t1 + t2 - t3
        _b[_link_start] += _X_Ik[_is_start] * _h_uk
        _b[_link_end] -= _U_Ik[_is_penult] * _h_dk
        # Call numba function
        _h_Ik = numba_solve_internals_ls(_h_Ik, NK, nk, _k_1k, _i_1k, _I_1k,
                                         _U, _X, _b)
        # Set depths at upstream and downstream ends
        _h_Ik[_is_start] = _h_uk
        _h_Ik[_is_end] = _h_dk
        # Set min depth
        _h_Ik[_h_Ik < min_depth] = min_depth
        # Solve for flows using new depths
        Q_ik_b, Q_ik_f = self.superlink_flow_error()
        _Q_ik = (Q_ik_b + Q_ik_f) / 2
        # Export instance variables
        self._Q_ik = _Q_ik
        self._h_Ik = _h_Ik

@njit
def numba_solve_internals_ls(_h_Ik, NK, nk, _k_1k, _i_1k, _I_1k, _U, _X, _b):
    for k in range(NK):
        nlinks = nk[k]
        lstart = _k_1k[k]
        rstart = _i_1k[k]
        jstart = _I_1k[k]
        _Ak = np.zeros((nlinks, nlinks - 1))
        for i in range(nlinks - 1):
            _Ak[i, i] = _U[lstart + i]
            _Ak[i + 1, i] = -_X[lstart + i]
        _bk = _b[rstart:rstart+nlinks]
        _AA = _Ak.T @ _Ak
        _Ab = _Ak.T @ _bk
        # If want to prevent singular matrix, set ( diag == 0 ) = 1
        for i in range(nlinks - 1):
            if (_AA[i, i] == 0.0):
                _AA[i, i] = 1.0
        _h_inner = np.linalg.solve(_AA, _Ab)
        _h_Ik[jstart+1:jstart+nlinks] = _h_inner
    return _h_Ik

@njit
def safe_divide(num, den):
    if (den == 0):
        return 0
    else:
        return num / den

@njit
def U_1k(E_2k, c_1k, A_1k, T_1k, g=9.81):
    """
    Compute forward recurrence coefficient 'U' for node 1, superlink k.
    """
    num = E_2k * c_1k - g * A_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit
def V_1k(P_1k, D_2k, c_1k, T_1k, a_1k=0.0, D_1k=0.0):
    """
    Compute forward recurrence coefficient 'V' for node 1, superlink k.
    """
    num = P_1k - D_2k * c_1k + D_1k * a_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit
def W_1k(A_1k, T_1k, a_1k=0.0, E_1k=0.0, g=9.81):
    """
    Compute forward recurrence coefficient 'W' for node 1, superlink k.
    """
    num = g * A_1k - E_1k * a_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit
def T_1k(a_1k, b_1k, c_1k):
    """
    Compute forward recurrence coefficient 'T' for link 1, superlink k.
    """
    return a_1k + b_1k + c_1k

@njit
def U_Ik(E_Ip1k, c_ik, A_ik, T_ik, g=9.81):
    """
    Compute forward recurrence coefficient 'U' for node I, superlink k.
    """
    num = E_Ip1k * c_ik - g * A_ik
    den = T_ik
    result = safe_divide(num, den)
    return result

@njit
def V_Ik(P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ik, V_Im1k, U_Im1k, T_ik, g=9.81):
    """
    Compute forward recurrence coefficient 'V' for node I, superlink k.
    """
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
    result = safe_divide(num, den)
    return  result

@njit
def W_Ik(A_ik, E_Ik, a_ik, W_Im1k, U_Im1k, T_ik, g=9.81):
    """
    Compute forward recurrence coefficient 'W' for node I, superlink k.
    """
    num = -(g * A_ik - E_Ik * a_ik) * W_Im1k
    den = (U_Im1k - E_Ik) * T_ik
    result = safe_divide(num, den)
    return result

@njit
def T_ik(a_ik, b_ik, c_ik, A_ik, E_Ik, U_Im1k, g=9.81):
    """
    Compute forward recurrence coefficient 'T' for link i, superlink k.
    """
    t_0 = a_ik + b_ik + c_ik
    t_1 = g * A_ik - E_Ik * a_ik
    t_2 = U_Im1k - E_Ik
    result = t_0 - safe_divide(t_1, t_2)
    return result

@njit
def X_Nk(A_nk, E_Nk, a_nk, O_nk, g=9.81):
    """
    Compute backward recurrence coefficient 'X' for node N, superlink k.
    """
    num = g * A_nk - E_Nk * a_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit
def Y_Nk(P_nk, D_Nk, a_nk, O_nk, c_nk=0.0, D_Np1k=0.0):
    """
    Compute backward recurrence coefficient 'Y' for node N, superlink k.
    """
    num = P_nk + D_Nk * a_nk - D_Np1k * c_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit
def Z_Nk(A_nk, O_nk, c_nk=0.0, E_Np1k=0.0, g=9.81):
    """
    Compute backward recurrence coefficient 'Z' for node N, superlink k.
    """
    num = E_Np1k * c_nk - g * A_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit
def O_nk(a_nk, b_nk, c_nk):
    """
    Compute backward recurrence coefficient 'O' for link n, superlink k.
    """
    return a_nk + b_nk + c_nk

@njit
def X_Ik(A_ik, E_Ik, a_ik, O_ik, g=9.81):
    """
    Compute backward recurrence coefficient 'X' for node I, superlink k.
    """
    num = g * A_ik - E_Ik * a_ik
    den = O_ik
    result = safe_divide(num, den)
    return result

@njit
def Y_Ik(P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ip1k, Y_Ip1k, X_Ip1k, O_ik, g=9.81):
    """
    Compute backward recurrence coefficient 'Y' for node I, superlink k.
    """
    t_0 = P_ik
    t_1 = a_ik * D_Ik
    t_2 = D_Ip1k * c_ik
    t_3 = (g * A_ik - E_Ip1k * c_ik)
    t_4 = D_Ip1k - Y_Ip1k
    t_5 = X_Ip1k + E_Ip1k
    t_6 = O_ik
    # TODO: There is still a divide by zero here
    num = (t_0 + t_1 - t_2 - (t_3 * t_4 / t_5))
    den = t_6
    result = safe_divide(num, den)
    return result

@njit
def Z_Ik(A_ik, E_Ip1k, c_ik, Z_Ip1k, X_Ip1k, O_ik, g=9.81):
    """
    Compute backward recurrence coefficient 'Z' for node I, superlink k.
    """
    num = (g * A_ik - E_Ip1k * c_ik) * Z_Ip1k
    den = (X_Ip1k + E_Ip1k) * O_ik
    result = safe_divide(num, den)
    return result

@njit
def O_ik(a_ik, b_ik, c_ik, A_ik, E_Ip1k, X_Ip1k, g=9.81):
    """
    Compute backward recurrence coefficient 'O' for link i, superlink k.
    """
    t_0 = a_ik + b_ik + c_ik
    t_1 = g * A_ik - E_Ip1k * c_ik
    t_2 = X_Ip1k + E_Ip1k
    result = t_0 + safe_divide(t_1, t_2)
    return result

@njit
def numba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _a_ik, _b_ik, _c_ik,
                             _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_1k, _i_1k):
    for k in range(NK):
        # Start at junction 1
        _I_1 = _I_1k[k]
        _i_1 = _i_1k[k]
        _I_2 = _I_1 + 1
        _i_2 = _i_1 + 1
        nlinks = nk[k]
        _T_ik[_i_1] = T_1k(_a_ik[_i_1], _b_ik[_i_1], _c_ik[_i_1])
        _U_Ik[_I_1] = U_1k(_E_Ik[_I_2], _c_ik[_i_1], _A_ik[_i_1], _T_ik[_i_1])
        _V_Ik[_I_1] = V_1k(_P_ik[_i_1], _D_Ik[_I_2], _c_ik[_i_1], _T_ik[_i_1],
                            _a_ik[_i_1], _D_Ik[_I_1])
        _W_Ik[_I_1] = W_1k(_A_ik[_i_1], _T_ik[_i_1], _a_ik[_i_1], _E_Ik[_I_1])
        # Loop from junction 2 -> Nk
        for i in range(nlinks - 1):
            _i_next = _i_2 + i
            _I_next = _I_2 + i
            _Im1_next = _I_next - 1
            _Ip1_next = _I_next + 1
            _T_ik[_i_next] = T_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                  _A_ik[_i_next], _E_Ik[_I_next], _U_Ik[_Im1_next])
            _U_Ik[_I_next] = U_Ik(_E_Ik[_Ip1_next], _c_ik[_i_next],
                                  _A_ik[_i_next], _T_ik[_i_next])
            _V_Ik[_I_next] = V_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                  _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                  _E_Ik[_I_next], _V_Ik[_Im1_next], _U_Ik[_Im1_next],
                                  _T_ik[_i_next])
            _W_Ik[_I_next] = W_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                  _W_Ik[_Im1_next], _U_Ik[_Im1_next], _T_ik[_i_next])
    return 1

@njit
def numba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _a_ik, _b_ik, _c_ik,
                              _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_Nk, _i_nk):
    for k in range(NK):
        _I_N = _I_Nk[k]
        _i_n = _i_nk[k]
        _I_Nm1 = _I_N - 1
        _i_nm1 = _i_n - 1
        _I_Np1 = _I_N + 1
        nlinks = nk[k]
        _O_ik[_i_n] = O_nk(_a_ik[_i_n], _b_ik[_i_n], _c_ik[_i_n])
        _X_Ik[_I_N] = X_Nk(_A_ik[_i_n], _E_Ik[_I_N], _a_ik[_i_n], _O_ik[_i_n])
        _Y_Ik[_I_N] = Y_Nk(_P_ik[_i_n], _D_Ik[_I_N], _a_ik[_i_n], _O_ik[_i_n],
                            _c_ik[_i_n], _D_Ik[_I_Np1])
        _Z_Ik[_I_N] = Z_Nk(_A_ik[_i_n], _O_ik[_i_n], _c_ik[_i_n], _E_Ik[_I_Np1])
        for i in range(nlinks - 1):
            _i_next = _i_nm1 - i
            _I_next = _I_Nm1 - i
            _Ip1_next = _I_next + 1
            _O_ik[_i_next] = O_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                  _A_ik[_i_next], _E_Ik[_Ip1_next], _X_Ik[_Ip1_next])
            _X_Ik[_I_next] = X_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                  _O_ik[_i_next])
            _Y_Ik[_I_next] = Y_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                  _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                  _E_Ik[_Ip1_next], _Y_Ik[_Ip1_next], _X_Ik[_Ip1_next],
                                  _O_ik[_i_next])
            _Z_Ik[_I_next] = Z_Ik(_A_ik[_i_next], _E_Ik[_Ip1_next], _c_ik[_i_next],
                                  _Z_Ik[_Ip1_next], _X_Ik[_Ip1_next], _O_ik[_i_next])
    return 1
