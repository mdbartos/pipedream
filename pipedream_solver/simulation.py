import time
import copy
import sys
from itertools import count
from collections import deque
import numpy as np
import pandas as pd
try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from pipedream_solver.nutils import interpolate_sample, _kalman_semi_implicit
else:
    from pipedream_solver.utils import interpolate_sample, _kalman_semi_implicit

eps = np.finfo(float).eps

class Simulation():
    """
    Class for managing and executing simulations.

    Inputs:
    -------
    model : pipedream_solver.Superlink instance
        Hydraulic model to simulate
    Q_in : pd.DataFrame (T x M)
        Flow input at each superjunction (m^3/s)
    H_bc : pd.DataFrame (T x M)
        Boundary stage at each superjunction (m)
    Q_Ik : pd.DataFrame (T x MK)
        Flow input at each internal junction (m^3/s)
    t_start : int
        Simulation start time (defaults to minimum of input data start times)
    t_end : int
        Simulation end time (defaults to maximum of input data end times)
    dt : float
        Default step size for simulation (seconds)
    max_iter : int
        Maximum allowable number of time steps in simulation
    min_dt : float
        Minimum allowable step size for simulation
    max_dt : float
        Maximum allowable step size for simulation
    tol : float
        Tolerance for truncation error when computing adaptive step size
    min_rel_change : float
        Minimum relative change for adaptive step size
    max_rel_change : float
        Maximum relative change for adaptive step size
    safety_factor : float
        Safety factor for adaptive step size
    Qcov : np.ndarray (M x M)
        Process noise covariance for Kalman Filter
    Rcov : np.ndarray (M x M)
        Measurement noise covariance for Kalman Filter
    C : np.ndarray (M x a)
        Signal-input matrix for Kalman Filter
    H : np.ndarray (b x M)
        Observation matrix for Kalman Filter
    interpolation_method : `linear` or `nearest`
        Interpolation method to use for sampling input forcings

    Methods:
    --------
    step : Advance model forward in time
    record_state : Record current simulation state
    load_state : Load recorded simulation state
    print_progress : Print progress bar
    filter_step_size : Compute adaptive step size
    kalman_filter : Fuse observed data using Kalman Filter

    Attributes:
    -----------
    states : pipedream_solver.simulation.States instance
        Class containing a collection of model states, including:
            H_j  - Superjunction heads (meters)
            Q_ik - Link flows (m^3/s)
            h_Ik - Junction depths (m)
            Q_uk - Flow into upstream ends of superlinks (m^3/s)
            Q_dk - Flow into downstream ends of superlinks (m^3/s)
    t : float
        Current simulation time
    """
    def __init__(self, model, Q_in=None, H_bc=None, Q_Ik=None, t_start=None,
                 t_end=None, dt=None, max_iter=None, min_dt=1, max_dt=200,
                 tol=0.01, min_rel_change=1e-10, max_rel_change=1e10, safety_factor=0.9,
                 Qcov=None, Rcov=None, C=None, H=None, interpolation_method='linear'):
        self.model = model
        if Q_in is not None:
            self.Q_in = Q_in.copy(deep=True)
            self.Q_in = self.Q_in.iloc[:, model.permutations]
            self.Q_in.index = self.Q_in.index.astype(float)
        else:
            self.Q_in = Q_in
        if H_bc is not None:
            self.H_bc = H_bc.copy(deep=True)
            self.H_bc = self.H_bc.iloc[:, model.permutations]
            self.H_bc.index = self.H_bc.index.astype(float)
        else:
            self.H_bc = H_bc
        if Q_Ik is not None:
            self.Q_Ik = Q_Ik.copy(deep=True)
            self.Q_Ik.index = self.Q_Ik.index.astype(float)
        else:
            self.Q_Ik = Q_Ik
        self.inputs = (self.Q_in, self.H_bc, self.Q_Ik)
        any_inputs = any(inp is not None for inp in self.inputs)
        if dt is None:
            dt = model._dt
        self.dt = dt
        self.max_iter = max_iter
        # Sample interpolation method
        if interpolation_method.lower() == 'linear':
            self.interpolation = 1
        elif interpolation_method.lower() == 'nearest':
            self.interpolation = 0
        else:
            raise ValueError('Argument `interpolation_method` must be one of `linear` or `nearest`.')
        # Adaptive step size handling
        self.err = None
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.tol = tol
        self.min_rel_change = min_rel_change
        self.max_rel_change = max_rel_change
        self.safety_factor = safety_factor
        # Add queue of dts
        self.dts = deque([dt], maxlen=3)
        self.errs = deque([eps], maxlen=3)
        self.h0100 = [1, 0, 0, 0, 0]
        self.h0211 = [1/2, 1/2, 0, 1/2, 0]
        self.h211  = [1/6, 1/6, 0, 0, 0]
        self.h0312 = [1/4, 1/2, 1/4, 3/4, 1/4]
        self.h312  = [1/18, 1/9, 1/18, 0, 0]
        self.h0321 = [5/4, 1/2, -3/4, -1/4, -3/4]
        self.h321  = [1/3, 1/18, -5/18, -5/6, -1/6]
        # Boundary conditions for convenience
        self.bc = self.model.bc
        # TODO: This needs to be generalized
        self.state_variables = {'H_j' : 'j',
                                'h_Ik' : 'Ik',
                                'Q_ik' : 'ik',
                                'Q_uk' : 'k',
                                'Q_dk' : 'k',
                                'Q_o' : 'o',
                                'Q_w' : 'w',
                                'Q_p' : 'p',
                                'x_Ik' : 'Ik'}
        if t_start is None:
            if any_inputs:
                self.t_start = min(i.index.min() for i in self.inputs if i is not None)
                self.model.t = self.t_start
            else:
                self.t_start = model.t
        else:
            self.t_start = t_start
        if t_end is None:
            if any_inputs:
                self.t_end = max(i.index.max() for i in self.inputs if i is not None)
            else:
                self.t_end = np.inf
        else:
            self.t_end = t_end
        # Configure kalman filtering
        if Rcov is None:
            self.Rcov = np.zeros((model.M, model.M))
        elif np.isscalar(Rcov):
            self.Rcov = Rcov * np.eye(model.M)
        elif (Rcov.shape[0] == Rcov.size):
            assert isinstance(Rcov, np.ndarray)
            self.Rcov = np.diag(Rcov)
        else:
            assert isinstance(Rcov, np.ndarray)
            self.Rcov = Rcov
        if Qcov is None:
            self.Qcov = np.zeros((model.M, model.M))
        elif np.isscalar(Qcov):
            self.Qcov = Qcov * np.eye(model.M)
        elif (Qcov.shape[0] == Qcov.size):
            assert isinstance(Qcov, np.ndarray)
            self.Qcov = np.diag(Qcov)
        else:
            assert isinstance(Qcov, np.ndarray)
            self.Qcov = Qcov
        if C is None:
            self.C = np.eye(model.M)
        elif np.isscalar(C):
            self.C = C * np.eye(model.M)
        elif (C.shape[0] == C.size):
            assert isinstance(C, np.ndarray)
            self.C = np.diag(C)
        else:
            assert isinstance(C, np.ndarray)
            self.C = C
        if H is None:
            self.H = np.eye(model.M)
        elif np.isscalar(H):
            self.H = H * np.eye(model.M)
        elif (H.shape[0] == H.size):
            assert isinstance(H, np.ndarray)
            self.H = np.diag(H)
        else:
            assert isinstance(H, np.ndarray)
            self.H = H
        self.P_x_k_k = self.C @ self.Qcov @ self.C.T
        # Progress bar checkpoints
        if np.isfinite(self.t_end):
            self._checkpoints = np.linspace(self.t_start, self.t_end)
        else:
            self._checkpoints = np.array([np.inf])
        self._checkpoint_num = 0
        self._iter_count = 0
        self._clock_start_time = 0
        self._clock_current_time = 0
        # Create a sequence iterator
        if max_iter is None:
            self.steps = count()
        else:
            self.steps = range(max_iter)
        self.states = States()
        for state in self.state_variables:
            if state in self.model.states:
                setattr(self.states, state, {})
                getattr(self.states, state).update({float(model.t) :
                                                    np.copy(model.states[state])})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # TODO: Should be able to choose what to record
        for state, state_type in self.state_variables.items():
            if hasattr(self.states, state):
                d = getattr(self.states, state)
                df = pd.DataFrame.from_dict(d, orient='index')
                if state_type == 'j':
                    df.columns = self.model.superjunction_names
                elif state_type == 'k':
                    df.columns = self.model.superlink_names
                setattr(self.states, state, df)

    @property
    def t(self):
        return self.model.t

    def load_state(self, state={}):
        """
        Load simulation state

        Inputs:
        -------
        state : dict
            Dict of state variables {variable_name : array}
        """
        self.model.load_state(state)

    def record_state(self, state_variables={}):
        """
        Save simulation state

        Inputs:
        -------
        state_variables : dict
            Dict of state variables {variable_name : indexing_scheme}
        """
        # TODO: Should be able to choose what to record
        # if not state_variables:
        model = self.model
        if not state_variables:
            state_variables = self.state_variables
        for state in state_variables:
            if state in self.model.states:
                if not hasattr(self.states, state):
                    setattr(self.states, state, {})
                getattr(self.states, state).update({float(model.t) :
                                                    np.copy(getattr(model, state))})
        # TODO: Add ability to record error and retry attempts

    def print_progress(self, use_checkpoints=True):
        """
        Print simulation progress

        Inputs:
        -------
        use_checkpoints : bool
            If True, use checkpoints to speed printing
        """
        # Import current and ending time
        t = self.t
        t_start = self.t_start
        t_end = self.t_end
        # Use checkpoints to avoid slowing down program with printing
        if use_checkpoints:
            checkpoints = self._checkpoints
            previous_checkpoint = self._checkpoint_num
            current_checkpoint = np.searchsorted(checkpoints, t)
            # If we haven't reached a new checkpoint, exit function
            if (current_checkpoint == previous_checkpoint):
                return None
            else:
                self._checkpoint_num = current_checkpoint
        # Get clock time
        elapsed_time = round(time.time() - self._clock_start_time, 2)
        # Prevent from going over 100%
        if t > t_end:
            t = t_end
        bar_len = 50
        progress_ratio = float(t - t_start) / float(t_end - t_start)
        progress_len = int(round(bar_len * progress_ratio))
        pct_finished = round(100.0 * progress_ratio, 1)
        bar = '=' * progress_len + '-' * (bar_len - progress_len)
        sys.stdout.write('\r[{0}] {1}{2} [{3} s]'.format(bar, pct_finished, '%', elapsed_time))
        sys.stdout.flush()

    def _scaled_error(self, err, atol=1e-6, rtol=1e-3):
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        current_state = self.model.H_j
        previous_state = self.model.states['H_j']
        # TODO: This should probably be absolute value
        max_state = np.maximum(current_state, previous_state)
        scaled_err = err / (atol + max_state * rtol)
        return scaled_err

    def _normed_error(self, scaled_err, norm=2):
        n = scaled_err.size
        if (norm == 0):
            normed_err = (scaled_err > 0).astype(int).sum() / n
        elif (norm == 1):
            normed_err = scaled_err.sum() / n
        elif (norm == 2):
            normed_err = np.sqrt((scaled_err**2).sum() / n)
        # TODO: should check for inf too
        elif (norm == -1):
            normed_err = np.max(np.abs(scaled_err))
        else:
            raise
        normed_err = max(abs(normed_err), eps)
        return normed_err

    def compute_step_size(self, dt=None, err=None, min_dt=None, max_dt=None, tol=None,
                          min_rel_change=None, max_rel_change=None,
                          safety_factor=None):
        # Get current step size
        if dt is None:
            dt = self.dt
        # If no error metric provided, return current step size
        if err is None:
            err = self.err
            if err is None:
                return dt
        # Set parameters if not given
        if min_dt is None:
            min_dt = self.min_dt
        if max_dt is None:
            max_dt = self.max_dt
        if tol is None:
            tol = self.tol
        if min_rel_change is None:
            min_rel_change = self.min_rel_change
        if max_rel_change is None:
            max_rel_change = self.max_rel_change
        if safety_factor is None:
            safety_factor = self.safety_factor
        # Compute new step size
        err = max(abs(err), eps)
        dt = safety_factor * dt * min(max(np.sqrt(tol / 2 / err),
                                          min_rel_change), max_rel_change)
        dt = min(max(dt, min_dt), max_dt)
        # Return new step size
        return dt

    def filter_step_size(self, tol=0.5, dts=None, errs=None, coeffs=[0.5, 0.5, 0, 0.5, 0],
                         min_dt=None, max_dt=None, min_rel_change=None, max_rel_change=None,
                         safety_factor=1.0, k=2):
        """
        Compute adaptive step size

        Recommended coeffs:
         k(b1)  k(b2)  k(b3)    a2     a3
        ----------------------------------
        [    1     0      0      0     0 ]  Elementary
        [  1/2   1/2      0    1/2     0 ]  H0211
        [  1/6   1/6      0      0     0 ]  H211 PI
        [  1/4   1/2    1/4    3/4   1/4 ]  H0312
        [ 1/18   1/9   1/18      0     0 ]  H312 PID
        [  5/4   1/2   -3/4   -1/4  -3/4 ]  H0321
        [  1/3  1/18  -5/18   -5/6  -1/6 ]  H321
        ----------------------------------

        Inputs:
        -------
        tol : float
            Allowable tolerance for scaled error
        dts : list
            Timestep sizes at previous n iterations (seconds)
        errs : list
            Scaled and normed errors at previous n iterations
        coeffs : list (length 6)
            Filter coefficients for computing stepsize
        min_dt : float
            Minimum allowed stepsize (seconds)
        max_dt : float
            Maximum allowed stepsize (seconds)
        min_rel_change : float
            Minimum relative change in stepsize
        max_rel_change : float
            Maximum relative change in stepsize
        safety_factor : float
            Safety factor for computing stepsize
        k : int
            Order of the truncation error
        """
        if dts is None:
            dts = self.dts
            if len(dts) < 3:
                return dts[0]
        if errs is None:
            errs = self.errs
        if min_dt is None:
            min_dt = self.min_dt
        if max_dt is None:
            max_dt = self.max_dt
        if min_rel_change is None:
            min_rel_change = self.min_rel_change
        if max_rel_change is None:
            max_rel_change = self.max_rel_change
        err_n, err_nm1, err_nm2 = errs
        dt_n, dt_nm1, dt_nm2 = dts
        beta_1, beta_2, beta_3, alpha_2, alpha_3 = coeffs
        t_0 = (tol / err_n) ** (beta_1 / k)
        t_1 = (tol / err_nm1) ** (beta_2 / k)
        t_2 = (tol / err_nm2) ** (beta_3 / k)
        t_3 = (dt_n / dt_nm1) ** (-alpha_2)
        t_4 = (dt_nm1 / dt_nm2) ** (-alpha_3)
        factor = t_0 * t_1 * t_2 * t_3 * t_4
        if np.isnan(factor):
            return dt_n
        factor = min(max(factor, min_rel_change), max_rel_change)
        factor = safety_factor * factor
        dt_np1 = factor * dt_n
        dt_np1 = min(max(dt_np1, min_dt), max_dt)
        return dt_np1

    def kalman_filter(self, Z, H=None, C=None, Qcov=None, Rcov=None, P_x_k_k=None,
                      dt=None, **kwargs):
        """
        Apply Kalman Filter to fuse observed data into model.

        Inputs:
        -------
        Z : np.ndarray (b x 1)
            Observed data
        H : np.ndarray (M x b)
            Observation matrix
        C : np.ndarray (a x M)
            Signal-input matrix
        Qcov : np.ndarray (M x M)
            Process noise covariance
        Rcov : np.ndarray (M x M)
            Measurement noise covariance
        P_x_k_k : np.ndarray (M x M)
            Posterior error covariance estimate at previous timestep
        dt : float
            Timestep (seconds)
        """
        if dt is None:
            dt = self.dt
        if P_x_k_k is None:
            P_x_k_k = self.P_x_k_k
        if H is None:
            H = self.H
        if C is None:
            C = self.C
        if Qcov is None:
            Qcov = self.Qcov
        if Rcov is None:
            Rcov = self.Rcov
        A_1, A_2, b = self.model._semi_implicit_system(_dt=dt)
        b_hat, P_x_k_k = _kalman_semi_implicit(Z, P_x_k_k, A_1, A_2, b, H, C,
                                               Qcov, Rcov)
        self.P_x_k_k = P_x_k_k
        self.model.b = b_hat
        self.model.iter_count -= 1
        self.model.t -= dt
        self.model._solve_step(dt=dt, **kwargs)

    def step(self, dt=None, subdivisions=1, retries=0, tol=1, norm=2,
             coeffs=[0.5, 0.5, 0, 0.5, 0], safety_factor=1.0, **kwargs):
        """
        Advance model forward to next timestep.

        Inputs:
        -------
        dt : float
            Timestep
        subdivisions : int
            Number of subdivisions for error estimation
        retries : int
            Number of retries if error exceeds allowed threshold
        tol : float
            Tolerance of scaled truncation error
        norm : float
            Norm to apply when computing total truncation error
        coeffs : list (length 6)
            Filter coefficients for computing adaptive step size
        safety_factor : float
            Safety factor for adaptive step size
        kwargs : **dict
            Keyword arguments passed to self.model.step
        """
        if (self._iter_count == 0):
            self._clock_start_time = time.time()
        if dt is None:
            dt = self.dt
        else:
            self.dt = dt
        # Advance model forward one large step
        self._step(dt=dt, **kwargs)
        # NOTE: This is stored after stepping, because of the way save state works
        if retries:
            initial_state = copy.deepcopy(self.model.states)
        else:
            initial_state = self.model.states
        err = None
        # If using adaptive time-stepping...
        if subdivisions > 1:
            # Copy coarse-stepped estimate of state
            states_coarse = np.copy(self.model.H_j)
            # Load previous state
            self.load_state(initial_state)
            # Advance model forward with number of steps given by subdivisions
            for _ in range(subdivisions):
                self._step(dt=dt / subdivisions, **kwargs)
            # Copy fine-stepped estimate of state
            states_fine = np.copy(self.model.H_j)
            # TODO: Is there a way to generalize this error metric?
            raw_err = states_coarse - states_fine
            scaled_err = self._scaled_error(raw_err)
            err = self._normed_error(scaled_err, norm=norm)
        # Set instance variables
        self.dt = dt
        self.dts.appendleft(dt)
        self.err = err
        self.errs.appendleft(err)
        # TODO: This will not save the dt needed for the next step
        if ((retries) and (err is not None)):
            min_dt = self.min_dt
            # dt = self.compute_step_size(dt, tol=tol, err=err)
            # dt = self.filter_step_size(tol=tol, coeffs=coeffs,
            #                            safety_factor=safety_factor)
            dt = 0.5 * dt
            if ((err > tol) or (not np.isfinite(err))) and (dt > min_dt):
                self.dts.popleft()
                self.errs.popleft()
                self.load_state(initial_state)
                self.step(dt=dt, subdivisions=subdivisions, retries=retries-1, **kwargs)
        assert np.isfinite(self.model.H_j).all()
        self._iter_count += 1

    def _step(self, dt=None, **kwargs):
        # Specify current timestamps
        t_next = self.t + dt
        # Import inputs
        interpolation_method = self.interpolation
        if not 'Q_in' in kwargs:
            Q_in = self.Q_in
            # Get superjunction runoff input
            if Q_in is not None:
                Q_in_index, Q_in_values = Q_in.index.values, Q_in.values
                Q_in_next = interpolate_sample(t_next, Q_in_index, Q_in_values,
                                               interpolation_method)
            else:
                Q_in_next = None
        else:
            Q_in_next = kwargs.pop('Q_in')
        if not 'H_bc' in kwargs:
            H_bc = self.H_bc
            # Get head boundary conditions
            if H_bc is not None:
                H_bc_index, H_bc_values = H_bc.index.values, H_bc.values
                H_bc_next = interpolate_sample(t_next, H_bc_index, H_bc_values,
                                               interpolation_method)
            else:
                H_bc_next = None
        else:
            H_bc_next = kwargs.pop('H_bc')
        if not 'Q_Ik' in kwargs:
            Q_Ik = self.Q_Ik
            # Get junction runoff input
            if Q_Ik is not None:
                Q_Ik_index, Q_Ik_values = Q_Ik.index.values, Q_Ik.values
                Q_Ik_next = interpolate_sample(t_next, Q_Ik_index, Q_Ik_values,
                                               interpolation_method)
            else:
                Q_Ik_next = None
        else:
            Q_Ik_next = kwargs.pop('Q_Ik')
        # Infer if system is banded
        if not 'banded' in kwargs:
            banded = self.model.banded
        else:
            banded = kwargs.pop('banded')
        # Step model forward with stepsize dt
        self.Q_in_next = Q_in_next
        self.H_bc_next = H_bc_next
        self.Q_Ik_next = Q_Ik_next
        self.model.step(Q_in=Q_in_next, H_bc=H_bc_next, Q_0Ik=Q_Ik_next, dt=dt, banded=banded,
                        **kwargs)

class States():
    """
    Class for holding model states
    """
    def __init__(self):
        pass

    def __repr__(self):
        string = '[' + str(', '.join(self.__dict__.keys())) + ']'
        return string
