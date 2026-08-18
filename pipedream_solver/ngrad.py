import numpy as np
from numba import njit
from numba.types import float64, int64, uint32, uint16, uint8, boolean, UniTuple, Tuple, List, DictType, void
from pipedream_solver._nsuperlink import numba_add_at

@njit
def numba_continuity_error_j(_H_j_next, _H_j_prev, _A_sj, 
                             _Q_in, _Q_uk, _Q_dk, _Q_o, _Q_w, _Q_p,
                             _J_uk, _J_dk, _J_uo, _J_do, _J_uw, _J_dw, _J_up, _J_dp, 
                             _B_uk, _B_dk, _dx_uk, _dx_dk, _theta_uk, _theta_dk,
                             _bc, _dt):
    """
    err = [A_sj (dH_j / dt) + Q_dk - Q_uk] - [Q_in] 
    """
    error = np.zeros(_H_j_next.size, dtype=np.float64)
    Q_in = _Q_in
    error += _A_sj / _dt
    numba_add_at(error, _J_uk, _B_uk * _dx_uk * _theta_uk / 2 / _dt)
    numba_add_at(error, _J_dk, _B_dk * _dx_dk * _theta_dk / 2 / _dt)
    error *= (_H_j_next - _H_j_prev)
    numba_add_at(error, _J_uk, _Q_uk)
    numba_add_at(error, _J_dk, -_Q_dk)
    numba_add_at(error, _J_uo, _Q_o)
    numba_add_at(error, _J_do, -_Q_o)
    numba_add_at(error, _J_uw, _Q_w)
    numba_add_at(error, _J_dw, -_Q_w)
    numba_add_at(error, _J_up, _Q_p)
    numba_add_at(error, _J_dp, -_Q_p)
    error -= Q_in
    error[_bc] = 0.
    return error

@njit
def numba_momentum_error_uk(_Q_uk, _Q_ik, _H_j, _h_Ik, _b_uk, _c_uk, _P_uk, 
                            _A_uuk, _A_duk, _theta_uk, _i_1k, _I_1k, _J_uk):
    """
    err = [b_uk Q_uk + c_uk Q_1k] - [P_uk + g A_uuk theta_uk H_juk - A_duk h_uk] 
    """
    g = 9.81
    error = np.zeros(_Q_uk.size, dtype=np.float64)
    Q_1k_next = _Q_ik[_i_1k]
    Q_uk_next = _Q_uk
    h_uk_next = _h_Ik[_I_1k]
    H_juk_next = _H_j[_J_uk]
    b_uk = _b_uk
    c_uk = _c_uk
    P_uk = _P_uk
    A_uuk = _A_uuk
    A_duk = _A_duk
    theta_uk = _theta_uk
    LHS = b_uk * Q_uk_next + c_uk * Q_1k_next
    RHS = P_uk + g * (A_uuk * theta_uk * H_juk_next - A_duk * h_uk_next)
    error = LHS - RHS
    return error

@njit
def numba_momentum_error_dk(_Q_dk, _Q_ik, _H_j, _h_Ik, _b_dk, _a_dk, _P_dk, 
                            _A_udk, _A_ddk, _theta_dk, _i_nk, _I_Np1k, _J_dk):
    """
    err = [b_dk Q_dk + a_dk Q_nk] - [P_dk + g A_udk h_dk - A_ddk theta_dk H_jdk] 
    """
    g = 9.81
    error = np.zeros(_Q_dk.size, dtype=np.float64)
    Q_dk_next = _Q_dk
    Q_nk_next = _Q_ik[_i_nk]
    h_dk_next = _h_Ik[_I_Np1k]
    H_jdk_next = _H_j[_J_dk]
    b_dk = _b_dk
    a_dk = _a_dk
    A_udk = _A_udk
    A_ddk = _A_ddk
    P_dk = _P_dk
    theta_dk = _theta_dk
    # Friction and local losses
    LHS = b_dk * Q_dk_next + a_dk * Q_nk_next
    RHS = P_dk + g * (A_udk * h_dk_next - A_ddk * theta_dk * H_jdk_next)
    error = LHS - RHS
    return error

@njit
def numba_continuity_error_Ik(_h_Ik_next, _Q_ik, _Q_uk, _Q_dk, _E_Ik, _D_Ik, 
                              _is_start, _is_end, _forward_I_i, _backward_I_i, _kI):
    """
    err = [E_Ik h_Ik - Q_im1k + Q_ik] - [D_Ik] 
    """
    N = _h_Ik_next.size
    error = np.zeros(N, dtype=np.float64)
    for I in range(N):
        k = _kI[I]
        error[I] += _E_Ik[I] * _h_Ik_next[I]
        error[I] -= _D_Ik[I]
        if _is_start[I]:
            i = _forward_I_i[I]
            _Q_in = _Q_uk[k]
            _Q_out = _Q_ik[i]
        elif _is_end[I]:
            im1 = _backward_I_i[I]
            _Q_in = _Q_ik[im1]
            _Q_out = _Q_dk[k]
        else:
            i = _forward_I_i[I]
            im1 = i - 1
            _Q_in = _Q_ik[im1]
            _Q_out = _Q_ik[i]
        error[I] -= _Q_in
        error[I] += _Q_out
    return error

@njit
def numba_momentum_error_ik(_Q_ik, _h_Im1k_next, _h_Ip1k_next, _Q_uk, _Q_dk, _A_uik, _A_dik, 
                            _a_ik, _b_ik, _c_ik, _P_ik, _ki, _link_start, _link_end):
    """
    err = [a_ik Q_im1k + b_ik Q_ik + c_ik Q_ip1k] - [P_ik + g A_uik h_Ik - g A_dik h_Ip1k]
    """
    g = 9.81
    n = _Q_ik.size
    error = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ip1 = i + 1
        im1 = i - 1
        link_is_start = _link_start[i]
        link_is_end = _link_end[i]
        single_link = link_is_start and link_is_end
        k = _ki[i]
        error[i] += _b_ik[i] * _Q_ik[i]
        error[i] -= _P_ik[i]
        error[i] -= g * (_A_uik[i] * _h_Im1k_next[i] - _A_dik[i] * _h_Ip1k_next[i])
        if single_link:
            error[i] += _a_ik[i] * _Q_uk[k]
            error[i] += _c_ik[i] * _Q_dk[k]
        else:
            if link_is_start:
                error[i] += _a_ik[i] * _Q_uk[k]
                error[i] += _c_ik[i] * _Q_ik[ip1]
            elif link_is_end:
                error[i] += _a_ik[i] * _Q_ik[im1]
                error[i] += _c_ik[i] * _Q_dk[k]
            else:
                error[i] += _a_ik[i] * _Q_ik[im1]
                error[i] += _c_ik[i] * _Q_ik[ip1]
    return error

@njit
def numba_momentum_error_o(_Q_o, _H_j, _alpha_o, _beta_o, _chi_o, _J_uo, _J_do):
    H_juo = _H_j[_J_uo]
    H_jdo = _H_j[_J_do]
    error = _Q_o - _alpha_o * H_juo - _beta_o * H_jdo - _chi_o
    return error

@njit
def numba_momentum_error_w(_Q_w, _H_j, _alpha_w, _beta_w, _chi_w, _J_uw, _J_dw):
    H_juw = _H_j[_J_uw]
    H_jdw = _H_j[_J_dw]
    error = _Q_w - _alpha_w * H_juw - _beta_w * H_jdw - _chi_w
    return error

@njit
def numba_momentum_error_p(_Q_p, _H_j, _alpha_p, _beta_p, _chi_p, _J_up, _J_dp):
    H_jup = _H_j[_J_up]
    H_jdp = _H_j[_J_dp]
    error = _Q_p - _alpha_p * H_jup - _beta_p * H_jdp - _chi_p
    return error