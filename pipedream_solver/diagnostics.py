import numpy as np
from pipedream_solver._nsuperlink import numba_compute_functional_storage_volumes, numba_compute_tabular_storage_volumes
from pipedream_solver._nsuperlink import junction_numerator, junction_denominator, superjunction_numerator, superjunction_denominator
from pipedream_solver.callbacks import BaseCallback

from numba import njit, prange
from numba.types import float64, int64, uint32, uint16, uint8, boolean, UniTuple, Tuple, List, DictType, void

class ErrorTracker(BaseCallback):
    """
    Tracks the error between the left-hand and right-hand side of each equation
    """
    def __init__(self, model, rtol=1e-1, atol=1e-10):
        self.model = model
        self.errors = {}
        self.continuity_error_j = np.zeros(model.M)
        self.continuity_error_Ik = np.zeros(model._I.size)
        self.momentum_error_ik = np.zeros(model._i.size)
        self.momentum_error_uk = np.zeros(model.NK)
        self.momentum_error_dk = np.zeros(model.NK)
        self.momentum_error_o = np.zeros(model.n_o)
        self.momentum_error_w = np.zeros(model.n_w)
        self.momentum_error_p = np.zeros(model.n_p)
        self.continuity_magnitude_j = np.zeros(model.M)
        self.continuity_magnitude_Ik = np.zeros(model._I.size)
        self.momentum_magnitude_ik = np.zeros(model._i.size)
        self.momentum_magnitude_uk = np.zeros(model.NK)
        self.momentum_magnitude_dk = np.zeros(model.NK)
        self.momentum_magnitude_o = np.zeros(model.n_o)
        self.momentum_magnitude_w = np.zeros(model.n_w)
        self.momentum_magnitude_p = np.zeros(model.n_p)
        self.rtol = rtol
        self.atol = atol

    @property
    def continuity_error(self):
        return np.concatenate([self.continuity_error_j, self.continuity_error_Ik])

    @property
    def momentum_error(self):
        return np.concatenate([self.momentum_error_uk, self.momentum_error_dk,
                               self.momentum_error_ik])

    @property
    def error(self):
        return np.concatenate([self.continuity_error_j, self.momentum_error_uk,
                               self.momentum_error_dk, self.continuity_error_Ik,
                               self.momentum_error_ik])
    @property
    def magnitude(self):
        return np.concatenate([self.continuity_magnitude_j, self.momentum_magnitude_uk,
                               self.momentum_magnitude_dk, self.continuity_magnitude_Ik,
                               self.momentum_magnitude_ik])

    @property
    def error_metric(self):
        rtol = self.rtol
        atol = self.atol
        dt = self.model._dt
        e = np.abs(self.error) * dt
        # TODO: Is maximum correct here?
        ewt = np.maximum(rtol * np.abs(self.magnitude), atol)
        metric = (e / ewt).max()
        return metric

    @property
    def success(self):
        error_metric = self.error_metric
        condition = error_metric <= 1.
        return condition
    
    def _record_error(self, *args, **kwargs):
        t = self.model.t
        self.errors[t] = self.error
        
    def _compute_error(self, *args, **kwargs):
        model = self.model
        # TODO: This could cause problems, need to make sure this stays updated at each step
        dt = model._dt
        self.continuity_error_j = continuity_error_j(model, dt)
        self.continuity_error_Ik = continuity_error_Ik(model, dt)
        self.momentum_error_ik = momentum_error_ik(model, dt)
        self.momentum_error_uk = momentum_error_uk_2(model, dt)
        self.momentum_error_dk = momentum_error_dk_2(model, dt)
        # Use these for tsuperlink
        #self.momentum_error_ik = momentum_error_ik_2(model, dt)
        #self.momentum_error_uk = momentum_error_uk_3(model, dt)
        #self.momentum_error_dk = momentum_error_dk_3(model, dt)
        #self.momentum_error_o = 
        #self.momentum_error_w = 
        #self.momentum_error_p = 

    def _compute_magnitudes(self, *args, **kwargs):
        model = self.model
        # TODO: This dt could cause problems, need to make sure this stays updated at each step
        dt = model._dt
        self.continuity_magnitude_j = continuity_magnitude_j(model, dt)
        self.continuity_magnitude_Ik = continuity_magnitude_Ik(model, dt)
        self.momentum_magnitude_uk = momentum_magnitude_uk(model, dt)
        self.momentum_magnitude_dk = momentum_magnitude_dk(model, dt)
        self.momentum_magnitude_ik = momentum_magnitude_ik(model, dt)

    def __on_step_end__(self, *args, **kwargs):
        self._compute_error(*args, **kwargs)
        self._compute_magnitudes(*args, **kwargs)


class ConditionTracker(BaseCallback):
    def __init__(self, model):
        self.model = model

    def _superlink_condition_number(self):
        model = self.model
        singular_values = (model._X_Ik[model._Ik])**2
        max_singular_value = max(1., singular_values.max())
        min_singular_value = min(1., singular_values.min())
        condition_number = max_singular_value / min_singular_value

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
        self.set_init_volume()

    @property
    def init_volume(self):
        result = (self.init_volume_j.sum() + self.init_volume_Ik.sum() + self.init_volume_ik.sum() 
                  + self.init_volume_uk.sum() + self.init_volume_dk.sum())
        return result

    @property
    def total_volume(self):
        result = (self.volume_j.sum() + self.volume_Ik.sum() + self.volume_ik.sum() 
                  + self.volume_uk.sum() + self.volume_dk.sum())
        return result

    @property
    def total_volume_flux(self):
        result = (self.volume_flux_j.sum() + self.volume_flux_Ik.sum())
        return result

    @property
    def cumulative_volume_flux(self):
        result = (self.cumulative_vol_flux_j.sum() + self.cumulative_vol_flux_Ik.sum())
        return result

    def set_init_volume(self):
        model = self.model
        self.init_volume_j = volume_j(model)
        self.init_volume_Ik = volume_Ik(model)
        self.init_volume_ik = volume_ik(model)
        self.init_volume_uk = volume_uk(model)
        self.init_volume_dk = volume_dk(model)

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
    def __init__(self, model, xtol=1e-6, max_learning_rate=0.5, min_learning_rate=0.01, beta=2.):
        self.model = model
        self.x_old = self._compute_prior_guess()
        self.x_new = self._compute_next_guess()
        self.dx = self._compute_guess_difference(self.x_old, self.x_new)
        self.xtol = xtol
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate = max_learning_rate
        self.beta = beta
        self.success = None
        self.convergence_queue = []
        self.learning_queue = []
        self.convergence_metric = np.inf

    def _compute_step_ratio(self, dx):
        # TODO: Need to add weirs, orifices, pumps
        # TODO: Need to account for elements with multiple outflows
        model = self.model
        # Compute bounds on allowable discharges
        Q_ik_ub = model._E_Ik[model._Ik] * model._h_Ik[model._Ik]
        Q_ik_lb = -model._E_Ik[model._Ip1k] * model._h_Ik[model._Ip1k]
        Q_uk_ub = (model._A_sj[model._J_uk] * model.H_j[model._J_uk] 
                + (model._B_uk * model._dx_uk / 2) 
                * ((model._theta_uk * (model.H_j[model._J_uk] - model._z_inv_j[model._J_uk]) 
                    + model._h_Ik[model._I_1k]) / 2)) / model._dt
        Q_uk_lb = -model._E_Ik[model._I_1k] * model._h_Ik[model._I_1k]
        Q_dk_ub = model._E_Ik[model._I_Np1k] * model._h_Ik[model._I_Np1k]
        Q_dk_lb = -(model._A_sj[model._J_dk] * model.H_j[model._J_dk] 
                + (model._B_dk * model._dx_dk / 2) 
                * ((model._theta_dk * (model.H_j[model._J_dk] - model._z_inv_j[model._J_dk]) 
                    + model._h_Ik[model._I_Np1k]) / 2)) / model._dt
        Q_w_ub = (model._A_sj[model._J_uw] * model.H_j[model._J_uw]) / model._dt
        Q_w_lb = -(model._A_sj[model._J_dw] * model.H_j[model._J_dw]) / model._dt
        Q_o_ub = (model._A_sj[model._J_uo] * model.H_j[model._J_uo]) / model._dt
        Q_o_lb = -(model._A_sj[model._J_do] * model.H_j[model._J_do]) / model._dt
        Q_p_ub = (model._A_sj[model._J_up] * model.H_j[model._J_up]) / model._dt
        Q_p_lb = -(model._A_sj[model._J_dp] * model.H_j[model._J_dp]) / model._dt


        # Compute binding step ratio
        ratio_ub = max(max(max_if(dx.get('Q_ik', 0.) / Q_ik_ub), 0.), 
                       max(max_if(dx.get('Q_uk', 0.) / Q_uk_ub), 0.),
                       max(max_if(dx.get('Q_dk', 0.) / Q_dk_ub), 0.),
                       max(max_if(dx.get('Q_w', 0.) / Q_w_ub), 0.),
                       max(max_if(dx.get('Q_o', 0.) / Q_o_ub), 0.),
                       max(max_if(dx.get('Q_p', 0.) / Q_p_ub), 0.)
                       )
        # TODO: Check if this should be max or min for maxif
        ####################################################
        # Negative signs removed before max_if because lb should be negative
        ratio_lb = max(max(max_if(dx.get('Q_ik', 0.) / Q_ik_lb), 0.), 
                       max(max_if(dx.get('Q_uk', 0.) / Q_uk_lb), 0.), 
                       max(max_if(dx.get('Q_dk', 0.) / Q_dk_lb), 0.),
                       max(max_if(dx.get('Q_w', 0.) / Q_w_lb), 0.),
                       max(max_if(dx.get('Q_o', 0.) / Q_o_lb), 0.),
                       max(max_if(dx.get('Q_p', 0.) / Q_p_lb), 0.)
                       )
        step_ratio = max(ratio_ub, ratio_lb)
        return step_ratio

    def _compute_learning_rate(self, step_ratio):
        max_learning_rate = self.max_learning_rate
        min_learning_rate = self.min_learning_rate
        beta = self.beta
        if step_ratio > 0:
            learning_rate = min(1 / step_ratio / beta, 1.)
        else:
            learning_rate = 1.
        learning_rate = max(min(learning_rate, max_learning_rate), min_learning_rate)
        return learning_rate

    def _convergence_metric(self, dx, x_old, x_new):
        assert x_old.keys() == x_new.keys() == dx.keys()
        abs_dx = {k : np.abs(dx[k]) for k in dx}
        abs_mag = {k : 1. + np.maximum(np.abs(x_new[k]), np.abs(x_old[k])) for k in x_new}
        max_step = max([(abs_dx[k] / abs_mag[k]).max() for k in abs_dx])
        return max_step

    def _convergence_met(self, convergence_metric, xtol):
        condition = (convergence_metric <= xtol)
        return condition

    def _compute_prior_guess(self):
        state = self.model.return_state()
        x_old = {k : v for k, v in state.items() if v.size > 0}
        return x_old
    
    def _compute_next_guess(self):
        state = self.model.return_state()
        x_new = {k : v for k, v in state.items() if v.size > 0}
        return x_new

    def _compute_guess_difference(self, x_old, x_new):
        assert x_old.keys() == x_new.keys()
        dx = {k : x_new[k] - x_old[k] for k in x_new}
        return dx

    def _set_states(self, x_old, x_new, learning_rate):
        model = self.model
        for state_name in x_new:
            prior_state = x_old[state_name]
            next_state = x_new[state_name]
            updated_state = (1 - learning_rate) * prior_state + (learning_rate) * next_state
            setattr(model, state_name, updated_state)

    def __on_step_start__(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
                          first_time=False, implicit=True, banded=None, first_iter=True,
                          num_iter=0, rtol=None, atol=None, head_tol=0.0015):
       self.x_old = self._compute_prior_guess() 
       self.success = False
    
    def __on_step_end__(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
                        first_time=False, implicit=True, banded=None, first_iter=True,
                        num_iter=1, rtol=None, atol=None, head_tol=0.0015):
        # Perform fixed-point iteration until convergence
        iter_elapsed = 1
        if (num_iter > 0):
            self.convergence_queue = []
            self.learning_queue = []
            self.x_new = self._compute_next_guess()
            self.dx = self._compute_guess_difference(self.x_old, self.x_new)
            self.convergence_metric = self._convergence_metric(self.dx, self.x_old, self.x_new)
            self.success = self._convergence_met(self.convergence_metric, self.xtol)
            if not self.success:
                for _ in range(num_iter):
                    # TODO: Rename this to step count
                    self.model.iter_count -= 1
                    self.model.t -= dt
                    self.x_old = self._compute_prior_guess()
                    # Enforce minimum depth
                    self.model.H_j = np.maximum(self.model.H_j, self.model._z_inv_j + self.model.min_depth)
                    self.model.h_Ik = np.maximum(self.model.h_Ik, self.model.min_depth)
                    try:
                        self.model._setup_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                            first_time=first_time, implicit=implicit, banded=banded,
                                            first_iter=False)
                        self.model._solve_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                            first_time=first_time, implicit=implicit, banded=banded,
                                            first_iter=False)
                    except:
                        self.model.iter_elapsed = iter_elapsed 
                        #self.model.load_state()
                        raise
                    self.x_new = self._compute_next_guess()
                    self.dx = self._compute_guess_difference(self.x_old, self.x_new)
                    self.step_ratio = self._compute_step_ratio(self.dx)
                    self.learning_rate = self._compute_learning_rate(self.step_ratio)
                    self._set_states(self.x_old, self.x_new, self.learning_rate)
                    self.convergence_metric = self._convergence_metric(self.dx, self.x_old, self.x_new)
                    self.success = self._convergence_met(self.convergence_metric, self.xtol)
                    self.convergence_queue.append(self.convergence_metric)
                    self.learning_queue.append(self.learning_rate)
                    iter_elapsed += 1
                    if self.success:
                        break
        self.model.iter_elapsed = iter_elapsed 
        # Enforce minimum depth
        self.model.H_j = np.maximum(self.model.H_j, self.model._z_inv_j + self.model.min_depth)
        self.model.h_Ik = np.maximum(self.model.h_Ik, self.model.min_depth)


class ExperimentalConvergenceTracker(ConvergenceTracker):

    def _compute_step_ratio(self, dx):
        # TODO: Need to add weirs, orifices, pumps
        model = self.model
        _D_Ik = model._D_Ik
        _Q_ik = model.Q_ik
        _Q_uk = model.Q_uk
        _Q_dk = model.Q_dk
        _Q_o = model.Q_o
        _Q_w = model.Q_w
        _Q_p = model.Q_p
        _kI = model._kI
        _forward_I_i = model.forward_I_i
        _backward_I_i = model.backward_I_i
        _is_start = model._is_start
        _is_end = model._is_end
        _D_j = model.b
        _J_uk = model._J_uk
        _J_dk = model._J_dk
        _J_uo = model._J_uo
        _J_do = model._J_do
        _J_uw = model._J_uw
        _J_dw = model._J_dw
        _J_up = model._J_up
        _J_dp = model._J_dp
        default = np.array([], dtype=np.float64)
        _dQ_ik = self.dx.get('Q_ik', default)
        _dQ_uk = self.dx.get('Q_uk', default)
        _dQ_dk = self.dx.get('Q_dk', default)
        _dQ_o = self.dx.get('Q_o', default)
        _dQ_w = self.dx.get('Q_w', default)
        _dQ_p = self.dx.get('Q_p', default)
        num_I = junction_numerator(_dQ_ik, _dQ_uk, _dQ_dk, _kI, 
                                   _forward_I_i, _backward_I_i, _is_start, _is_end)
        denom_I = junction_denominator(_D_Ik, _Q_ik, _Q_uk, _Q_dk, _kI, 
                                       _forward_I_i, _backward_I_i, _is_start, _is_end)
        num_j = superjunction_numerator(_D_j, _dQ_uk, _dQ_dk, _dQ_o, _dQ_w, _dQ_p, 
                                        _J_uk, _J_dk, _J_uo, _J_do, _J_uw, _J_dw, _J_up, _J_dp)
        denom_j = superjunction_denominator(_D_j, _Q_uk, _Q_dk, _Q_o, _Q_w, _Q_p, 
                                            _J_uk, _J_dk, _J_uo, _J_do, _J_uw, _J_dw, _J_up, _J_dp)
        beta_I = (num_I / denom_I).max()
        beta_j = (num_j / denom_j).max()
        step_ratio = max(beta_I, beta_j)
        return step_ratio

    def _compute_learning_rate(self, step_ratio):
        max_learning_rate = self.max_learning_rate
        min_learning_rate = self.min_learning_rate
        beta = self.beta
        if step_ratio > 0:
            learning_rate = min(1 / step_ratio / beta, 1.)
        else:
            learning_rate = 1.
        learning_rate = max(min(learning_rate, max_learning_rate), min_learning_rate)
        return learning_rate


class LegacyConvergenceTracker(BaseCallback):
    def __init__(self, model):
        self.model = model
        self.prior_guess = 0.
        self.next_guess = 0.
        self.learning_rate = 0.5
        self.min_depth = 1e-5

    def _convergence_met(self, prior_guess, next_guess, head_tol=0.0015):
        e = np.abs(next_guess - prior_guess)
        condition = e.max() <= head_tol
        return condition

    def _compute_prior_guess(self):
        return self.model.H_j
    
    def _compute_next_guess(self):
        return self.model.H_j

    def _set_states(self, prior_states, next_states):
        model = self.model
        alpha = self.learning_rate
        for state_name in next_states:
            prior_state = prior_states[state_name]
            next_state = next_states[state_name]
            updated_state = (1 - alpha) * prior_state + (alpha) * next_state
            setattr(model, state_name, updated_state)

    def __on_step_end__(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
                        first_time=False, implicit=True, banded=None, first_iter=True,
                        num_iter=1, rtol=None, atol=None, head_tol=0.0015):
        self.prior_guess = self._compute_prior_guess() 

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
                    self.prior_states = self.model.return_state()
                    self.model.H_j = np.maximum(self.model.H_j, self.model._z_inv_j + self.min_depth)
                    self.model.h_Ik = np.maximum(self.model.h_Ik, self.min_depth)
                    try:
                        self.model._setup_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                            first_time=first_time, implicit=implicit, banded=banded,
                                            first_iter=False)
                        self.model._solve_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                            first_time=first_time, implicit=implicit, banded=banded,
                                            first_iter=False)
                    except:
                        self.model.load_state()
                        raise
                    self.next_guess = self._compute_next_guess()
                    self.next_states = self.model.return_state()
                    iter_elapsed += 1
                    convergence_met = self._convergence_met(self.prior_guess, self.next_guess, head_tol=head_tol)
                    self._set_states(self.prior_states, self.next_states)
                    if convergence_met:
                        break
        self.model.iter_elapsed = iter_elapsed 

def continuity_error_j(model, dt):
    """
    err = [A_sj (dH_j / dt) + Q_dk - Q_uk] - [Q_in] 
    """
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
    #error -= Q_bc
    error[bc] = 0.
    return error

def continuity_magnitude_j(model, dt):
    mag = np.zeros(model.M)
    H_j_next = model.H_j
    H_j_prev = model.states['H_j']
    mag += model.A_sj / dt
    np.add.at(mag, model._J_uk, model._B_uk * model._dx_uk * model._theta_uk / 2 / dt)
    np.add.at(mag, model._J_dk, model._B_dk * model._dx_dk * model._theta_dk / 2 / dt)
    mag *= np.maximum(np.abs(H_j_next), np.abs(H_j_prev))
    return mag

def continuity_increment_j(model, dt):
    inc = np.zeros(model.M)
    H_j_next = model.H_j
    H_j_prev = model.states['H_j']
    inc += model.A_sj / dt
    np.add.at(inc, model._J_uk, model._B_uk * model._dx_uk * model._theta_uk / 2 / dt)
    np.add.at(inc, model._J_dk, model._B_dk * model._dx_dk * model._theta_dk / 2 / dt)
    inc *= (H_j_next - H_j_prev)
    return inc

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
    Q_1k_next = model.Q_ik[model._i_1k]
    Q_uk_next = model.Q_uk
    h_uk_next = model._h_uk
    H_juk_next = model.H_j[model._J_uk]
    b_uk = model._b_uk
    c_uk = model._c_uk
    P_uk = model._P_uk
    A_uk = model._A_uk
    LHS = b_uk * Q_uk_next + c_uk * Q_1k_next
    RHS = P_uk + g * A_uk * (H_juk_next - h_uk_next)
    error = LHS - RHS
    return error

# To be used with tsuperlink
def momentum_error_uk_3(model, dt):
    error = np.zeros(model.NK)
    Q_uk_next = model.Q_uk
    h_uk_next = model._h_uk
    H_juk_next = model.H_j[model._J_uk]
    a_uk = model._a_uk
    b_uk = model._b_uk
    c_uk = model._c_uk
    P_uk = model._P_uk
    LHS = a_uk * H_juk_next + b_uk * Q_uk_next + c_uk * h_uk_next
    RHS = P_uk
    error = LHS - RHS
    return error

def momentum_magnitude_uk(model, dt):
    Q_uk_next = model.Q_uk
    Q_uk_prev = model.states['Q_uk']
    mag = np.maximum(np.abs(Q_uk_next), np.abs(Q_uk_prev))
    return mag

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
    Q_nk_next = model.Q_ik[model._i_nk]
    h_dk_next = model._h_dk
    H_jdk_next = model.H_j[model._J_dk]
    b_dk = model._b_dk
    a_dk = model._a_dk
    A_dk = model._A_dk
    P_dk = model._P_dk
    # Friction and local losses
    LHS = b_dk * Q_dk_next + a_dk * Q_nk_next
    RHS = P_dk + g * A_dk * (h_dk_next - H_jdk_next)
    error = LHS - RHS
    return error

# To be used with tsuperlink
def momentum_error_dk_3(model, dt):
    error = np.zeros(model.NK)
    Q_dk_next = model.Q_dk
    h_dk_next = model._h_dk
    H_jdk_next = model.H_j[model._J_dk]
    a_dk = model._a_dk
    b_dk = model._b_dk
    c_dk = model._c_dk
    P_dk = model._P_dk
    LHS = a_dk * h_dk_next + b_dk * Q_dk_next + c_dk * H_jdk_next
    RHS = P_dk
    error = LHS - RHS
    return error

def momentum_magnitude_dk(model, dt):
    Q_dk_next = model.Q_dk
    Q_dk_prev = model.states['Q_dk']
    mag = np.maximum(np.abs(Q_dk_next), np.abs(Q_dk_prev))
    return mag

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
    """
    err = [E_Ik h_Ik - Q_im1k + Q_ik] - [D_Ik] 
    """
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

def continuity_magnitude_Ik(model, dt):
    h_Ik_next = model.h_Ik
    h_Ik_prev = model.states['h_Ik']
    mag = model._E_Ik * dt * np.maximum(np.abs(h_Ik_next), np.abs(h_Ik_prev))
    return mag

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
    if _i_is_internal.any():
        error[_i_is_start] += model._c_ik[_i_is_start] * model.Q_ik[_i_1k + 1]
        error[_i_is_end] += model._a_ik[_i_is_end] * model.Q_ik[_i_nk - 1]
        error[_i_is_internal] += model._a_ik[_i_is_internal] * model.Q_ik[_im1]
        error[_i_is_internal] += model._c_ik[_i_is_internal] * model.Q_ik[_ip1]
    return error

# To be used with tsuperlink
def momentum_error_ik_2(model, dt):
    error = np.zeros(model._i.size)
    Q_ik_next = model.Q_ik
    h_Ik_next = model.h_Ik
    error -= model._P_ik
    error += model._a_ik * h_Ik_next[model._Ik]
    error += model._b_ik * Q_ik_next
    error += model._c_ik * h_Ik_next[model._Ip1k]
    return error

def momentum_magnitude_ik(model, dt):
    Q_ik_next = model.Q_ik
    Q_ik_prev = model.states['Q_ik']
    mag = np.maximum(np.abs(Q_ik_next), np.abs(Q_ik_prev))
    return mag

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

def max_if(arr):
    if arr.size > 0:
        return arr.max()
    else:
        return 0.

def min_if(arr):
    if arr.size > 0:
        return arr.min()
    else:
        return 0.