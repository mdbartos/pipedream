import numpy as np
from pipedream_solver._nsuperlink import numba_compute_functional_storage_volumes, numba_compute_tabular_storage_volumes
from pipedream_solver.callbacks import BaseCallback

class ErrorTracker(BaseCallback):
    def __init__(self, model):
        self.model = model
        self.continuity_error_j = np.zeros(model.M)
        self.continuity_error_Ik = np.zeros(model._I.size)
        self.momentum_error_ik = np.zeros(model._i.size)
        self.momentum_error_uk = np.zeros(model.NK)
        self.momentum_error_dk = np.zeros(model.NK)
        self.momentum_error_o = np.zeros(model.n_o)
        self.momentum_error_w = np.zeros(model.n_w)
        self.momentum_error_p = np.zeros(model.n_p)

    def __on_step_end__(self, *args, **kwargs):
        model = self.model
        # TODO: This could cause problems, need to make sure this stays updated at each step
        dt = model._dt
        self.continuity_error_j = continuity_error_j(model, dt)
        self.continuity_error_Ik = continuity_error_Ik(model, dt)
        self.momentum_error_ik = momentum_error_ik(model, dt)
        self.momentum_error_uk = momentum_error_uk(model, dt)
        self.momentum_error_dk = momentum_error_dk(model, dt)
        #self.momentum_error_o = 
        #self.momentum_error_w = 
        #self.momentum_error_p = 
        #self.error = error

class VolumeTracker(BaseCallback):
    def __init__(self, model):
        self.model = model
        self.volume_j = np.zeros(model.M)
        self.volume_Ik = np.zeros(model._I.size)
        self.volume_ik = np.zeros(model._i.size)
        self.volume_uk = np.zeros(model.NK)
        self.volume_dk = np.zeros(model.NK)
        self.volume_flux_j = np.zeros(model.M)
        self.volume_flux_Ik = np.zeros(model._I.size)
        self.cumulative_vol_flux_j = np.zeros(model.M)
        self.cumulative_vol_flux_Ik = np.zeros(model._I.size)

    def __on_step_end__(self, *args, **kwargs):
        model = self.model
        dt = model._dt
        Q_in = model._Q_in
        Q_0Ik = model._Q_0Ik
        Q_bc = model._Q_bc
        self.volume_j = volume_j(model)
        self.volume_Ik = volume_Ik(model)
        self.volume_ik = volume_ik(model)
        self.volume_uk = volume_uk(model)
        self.volume_dk = volume_dk(model)
        self.volume_flux_j = volume_flux_j(Q_in, Q_bc, dt)
        self.volume_flux_Ik = volume_flux_Ik(Q_0Ik, dt)
        self.cumulative_vol_flux_j += self.volume_flux_j
        self.cumulative_vol_flux_Ik += self.volume_flux_Ik

class ConvergenceTracker(BaseCallback):
    def __init__(self, model, rtol=1e-5, atol=1e-8):
        self.model = model
        self.prior_guess = 0.
        self.next_guess = 0.
        self.rtol = rtol
        self.atol = atol

    def _convergence_met(self, prior_guess, next_guess, rtol=None, atol=None):
        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol
        e = np.abs(next_guess - prior_guess)
        ewt = np.maximum(rtol * np.maximum(np.abs(prior_guess), np.abs(next_guess)),  atol)
        valid = (e > 0.) & (ewt > 0.)
        condition = (np.log(e[valid]) - np.log(ewt[valid])).max() <= 0.
        return condition

    def _compute_prior_guess(self):
        #return np.copy(self.model.H_j)
        return self.model.state_vector
    
    def _compute_next_guess(self):
        #return np.copy(self.model.H_j)
        return self.model.state_vector

    def __on_step_start__(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
                          first_time=False, implicit=True, banded=None, first_iter=True,
                          num_iter=0, rtol=None, atol=None, head_tol=0.0015):
       self.prior_guess = self._compute_prior_guess() 
    
    def __on_step_end__(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
                        first_time=False, implicit=True, banded=None, first_iter=True,
                        num_iter=1, rtol=None, atol=None, head_tol=0.0015):
        # Perform fixed-point iteration until convergence
        iter_elapsed = 1
        if (num_iter > 0):
            self.next_guess = self._compute_next_guess()
            convergence_met = self._convergence_met(self.prior_guess, self.next_guess, rtol=rtol, atol=atol)
            if not convergence_met:
                for _ in range(num_iter):
                    # TODO: Rename this to step count
                    self.model.iter_count -= 1
                    self.model.t -= dt
                    self.prior_guess = self._compute_prior_guess()
                    self.model._setup_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                        first_time=first_time, implicit=implicit, banded=banded,
                                        first_iter=False)
                    self.model._solve_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                        first_time=first_time, implicit=implicit, banded=banded,
                                        first_iter=False)
                    self.next_guess = self._compute_next_guess()
                    iter_elapsed += 1
                    convergence_met = self._convergence_met(self.prior_guess, self.next_guess, rtol=rtol, atol=atol)
                    if convergence_met:
                        break
        self.model.iter_elapsed = iter_elapsed 

class LegacyConvergenceTracker(ConvergenceTracker):
    def __init__(self, model):
        self.model = model
        self.prior_guess = 0.
        self.next_guess = 0.

    def _convergence_met(self, prior_guess, next_guess, head_tol=0.0015):
        e = np.abs(next_guess - prior_guess)
        condition = e.max() <= head_tol
        return condition

    def _compute_prior_guess(self):
        #return np.copy(self.model.H_j)
        return self.model.H_j
    
    def _compute_next_guess(self):
        #return np.copy(self.model.H_j)
        return self.model.H_j

    def __on_step_end__(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
                        first_time=False, implicit=True, banded=None, first_iter=True,
                        num_iter=1, rtol=None, atol=None, head_tol=0.0015):
        # Perform fixed-point iteration until convergence
        iter_elapsed = 1
        if (num_iter > 0):
            self.next_guess = self._compute_next_guess()
            convergence_met = self._convergence_met(self.prior_guess, self.next_guess, head_tol=head_tol)
            if not convergence_met:
                for _ in range(num_iter):
                    # TODO: Rename this to step count
                    self.model.iter_count -= 1
                    self.model.t -= dt
                    self.prior_guess = self._compute_prior_guess()
                    self.model._setup_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                        first_time=first_time, implicit=implicit, banded=banded,
                                        first_iter=False)
                    self.model._solve_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                        first_time=first_time, implicit=implicit, banded=banded,
                                        first_iter=False)
                    self.next_guess = self._compute_next_guess()
                    iter_elapsed += 1
                    convergence_met = self._convergence_met(self.prior_guess, self.next_guess, head_tol=head_tol)
                    if convergence_met:
                        break
        self.model.iter_elapsed = iter_elapsed 

def continuity_error_j(model, dt):
    error = np.zeros(model.M)
    H_j_next = model.H_j
    H_j_prev = model.states['H_j']
    bc = model.bc
    Q_in = model._Q_in
    Q_bc = model._Q_bc
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
    error -= Q_bc
    #error[bc] = 0.
    return error

def momentum_error_uk(model, dt):
    error = np.zeros(model.NK)
    g = 9.81
    Q_uk_next = model.Q_uk
    h_uk_next = model._h_uk
    H_juk_next = model.H_j[model._J_uk]
    error += g * model._A_uk * model._kappa_uk * Q_uk_next
    error += g * model._A_uk * model._lambda_uk * H_juk_next
    error += g * model._A_uk * model._mu_uk
    error -= g * model._A_uk * h_uk_next
    # Friction and local losses
    #error += 0.
    return error

def momentum_error_uk_2(model, dt):
    error = np.zeros(model.NK)
    g = 9.81
    Q_uk_next = model.Q_uk
    Q_uk_prev = model.states['Q_uk']
    h_uk_next = model._h_uk
    H_juk_next = model.H_j[model._J_uk]
    z_inv_uk = model._z_inv_uk
    theta_uk = model._theta_uk
    error += (Q_uk_next - Q_uk_prev) * model._dx_uk / dt
    error += g * model._A_uk * (h_uk_next - theta_uk * (H_juk_next - z_inv_uk))
    error -= g * model._A_uk * model._dx_uk * model._S_o_uk
    # Friction and local losses
    #error += 0.
    return error

def momentum_error_dk(model, dt):
    error = np.zeros(model.NK)
    g = 9.81
    Q_dk_next = model.Q_dk
    h_dk_next = model._h_dk
    H_jdk_next = model.H_j[model._J_dk]
    error += g * model._A_dk * model._kappa_dk * Q_dk_next
    error += g * model._A_dk * model._lambda_dk * H_jdk_next
    error += g * model._A_dk * model._mu_dk
    error -= g * model._A_dk * h_dk_next
    # Friction and local losses
    #error += 0.
    return error

def momentum_error_dk_2(model, dt):
    error = np.zeros(model.NK)
    g = 9.81
    Q_dk_next = model.Q_dk
    Q_dk_prev = model.states['Q_dk']
    h_dk_next = model._h_dk
    H_jdk_next = model.H_j[model._J_dk]
    z_inv_dk = model._z_inv_dk
    theta_dk = model._theta_dk
    error += (Q_dk_next - Q_dk_prev) * model._dx_dk / dt
    error += g * model._A_dk * (theta_dk * (H_jdk_next - z_inv_dk) - h_dk_next)
    error -= g * model._A_dk * model._dx_dk * model._S_o_dk
    # Friction and local losses
    #error += 0.
    return error

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
    _i_1k = model._i_1k
    _i_nk = model._i_nk
    _i_is_start = np.zeros(model._i.size, dtype=np.bool_)
    _i_is_start[_i_1k] = True
    _i_is_end = np.zeros(model._i.size, dtype=np.bool_)
    _i_is_end[_i_nk] = True
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
    # Addition 2025-06-05
    error[_i_is_start] += model._c_ik[_i_is_start] * model.Q_ik[_i_1k + 1]
    error[_i_is_end] += model._a_ik[_i_is_end] * model.Q_ik[_i_nk - 1]
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

def volume_flux_j(Q_in, Q_bc, dt):
    V_in_exog_j = Q_in * dt
    V_bc_j = Q_bc * dt
    V_in_j = V_in_exog_j + V_bc_j
    return V_in_j

def volume_flux_Ik(Q_0Ik, dt):
    V_in_Ik = Q_0Ik * dt
    return V_in_Ik

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
