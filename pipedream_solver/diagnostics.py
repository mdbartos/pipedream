import numpy as np
from pipedream_solver._nsuperlink import numba_compute_functional_storage_volumes, numba_compute_tabular_storage_volumes
from pipedream_solver.callbacks import BaseCallback

class ErrorTracker(BaseCallback):
    def __init__(self, model):
        self.model = model
        self.error = None

    def __on_step_end__(self):
        model = self.model
        dt = model._dt
        error = compute_error(model, dt)
        self.error = error


def continuity_error_j(model, dt):
    error = np.zeros(model.M)
    H_j_next = model.H_j
    H_j_prev = model.states['H_j']
    bc = model.bc
    Q_in = model._Q_in
    error += model.A_sj / dt
    np.add.at(error, model._J_uk, model._B_uk * model._dx_uk * model._theta_uk / 2 / dt)
    np.add.at(error, model._J_dk, model._B_dk * model._dx_dk * model._theta_dk / 2 / dt)
    error *= (H_j_next - H_j_prev)
    np.add.at(error, model._J_uk, model._Q_uk)
    np.subtract.at(error, model._J_dk, model._Q_dk)
    np.add.at(error, model._J_uo, model._Qo)
    np.subtract.at(error, model._J_do, model._Qo)
    np.add.at(error, model._J_uw, model._Qw)
    np.subtract.at(error, model._J_dw, model._Qw)
    np.add.at(error, model._J_up, model._Qp)
    np.subtract.at(error, model._J_dp, model._Qp)
    error -= Q_in
    error[bc] = 0.
    return error

def momentum_error_uk(model, dt):
    error = np.zeros(model.NK)
    raise NotImplementedError

def momentum_error_dk(model, dt):
    error = np.zeros(model.NK)
    raise NotImplementedError

def momentum_error_o(model, dt):
    error = np.zeros(model.n_o)
    raise NotImplementedError

def momentum_error_w(model, dt):
    error = np.zeros(model.n_w)
    raise NotImplementedError

def momentum_error_p(model, dt):
    error = np.zeros(model.n_p)
    raise NotImplementedError

def continuity_error_Ik(model, dt):
    error = np.zeros(model._I.size)
    h_Ik_next = model.h_Ik
    h_Ik_prev = model.states['h_Ik']
    _I_internal = (~model._I_start) & (~model._I_end)
    error += model._E_Ik * h_Ik_next
    error -= model._D_Ik
    error[model._I_start] -= model._Q_uk
    error[model._I_start] += model._Q_ik[model._i_1k]
    error[model._I_end] -= model._Q_ik[model._i_nk]
    error[model._I_end] += model._Q_dk
    error[_I_internal] -= model._Q_ik[model.backward_I_i[_I_internal]]
    error[_I_internal] += model._Q_ik[model.forward_I_i[_I_internal]]
    return error

def momentum_error_ik(model, dt):
    error = np.zeros(model._i.size)
    _i_is_start = np.zeros(model._i.size, dtype=np.bool_)
    _i_is_start[model._i_1k] = True
    _i_is_end = np.zeros(model._i.size, dtype=np.bool_)
    _i_is_end[model._i_nk] = True
    _i_is_internal = (~_i_is_start) & (~_i_is_end)
    _im1 = (model._i - 1)[_i_is_internal]
    _ip1 = (model._i + 1)[_i_is_internal]
    g = 9.81
    Q_ik_next = model.Q_ik
    Q_ik_prev = model.states['Q_ik']
    h_Ik_next = model.h_Ik
    error -= model._P_ik
    error -= g * model._A_ik * (h_Ik_next[model._Ik] - h_Ik_next[model._Ip1k])
    error += model._b_ik * Q_ik_next
    error[_i_is_start] += model._a_ik[_i_is_start] * model.Q_uk
    error[_i_is_end] += model._c_ik[_i_is_end] * model.Q_dk
    error[_i_is_internal] += model._a_ik[_i_is_internal] * model.Q_ik[_im1]
    error[_i_is_internal] += model._c_ik[_i_is_internal] * model.Q_ik[_ip1]
    return error

def compute_error(model, dt):
    error_j = continuity_error_j(model, dt)
    error_Ik = continuity_error_Ik(model, dt)
    error_ik = momentum_error_ik(model, dt)
    error = np.concatenate([error_j, error_Ik, error_ik])
    return error

def volume_j(model):
    # Import instance variables
    _functional = model._functional              # Superlinks with functional area curves
    _tabular = model._tabular                    # Superlinks with tabular area curves
    _storage_a = model._storage_a                # Coefficient of functional storage curve
    _storage_b = model._storage_b                # Exponent of functional storage curve
    _storage_c = model._storage_c                # Constant of functional storage curve
    H_j = model.H_j                              # Head at superjunction j
    _z_inv_j = model._z_inv_j                    # Invert elevation at superjunction j
    min_depth = model.min_depth                  # Minimum depth allowed at superjunctions/nodes
    _storage_hs = model._storage_hs
    _storage_As = model._storage_As
    _storage_Vs = model._storage_Vs
    _storage_inds = model._storage_inds
    _storage_lens = model._storage_lens
    _storage_js = model._storage_js
    _storage_codes = model._storage_codes
    # Compute storage areas
    _V_sj = np.zeros(model.M, dtype=np.float64)
    _h_j = np.maximum(H_j - _z_inv_j, min_depth)
    numba_compute_functional_storage_volumes(_h_j, _V_sj, _storage_a, _storage_b,
                                                _storage_c, _functional)
    if _tabular.any():
        numba_compute_tabular_storage_volumes(_h_j, _V_sj, _storage_hs, _storage_As,
                                                _storage_Vs, _storage_js, _storage_codes,
                                                _storage_inds, _storage_lens)
    return _V_sj

def volume_uk(model):
    V_uk = model._A_uk * model._dx_uk
    return V_uk

def volume_dk(model):
    V_dk = model._A_dk * model._dx_dk
    return V_dk

def volume_Ik(model):
    V_Ik = model._A_SIk * model.h_Ik
    return V_Ik

def volume_ik(model):
    V_ik = model._A_ik * model._dx_ik
    return V_ik

def total_volume(model):
    V_sj = volume_j(model)
    V_uk = volume_uk(model)
    V_dk = volume_dk(model)
    V_o = np.zeros(model.n_o, dtype=np.float64)
    V_w = np.zeros(model.n_w, dtype=np.float64)
    V_p = np.zeros(model.n_p, dtype=np.float64)
    V_Ik = volume_Ik(model)
    V_ik = volume_ik(model)
    V_total = np.concatenate([V_sj, V_uk, V_dk, V_o, V_w, V_p, V_Ik, V_ik])
    return V_total
