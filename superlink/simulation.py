import copy
import sys
from itertools import count
import numpy as np
import pandas as pd
try:
    import numba
    _HAS_NUMBA = True
except:
    _HAS_NUMBA = False
if _HAS_NUMBA:
    from superlink.nutils import interpolate_sample
else:
    from superlink.utils import interpolate_sample

eps = np.finfo(float).eps

class Simulation():
    def __init__(self, model, Q_in=None, H_bc=None, Q_Ik=None, t_start=None,
                 t_end=None, dt=None, max_iter=None, min_dt=1, max_dt=200,
                 tol=0.01, min_rel_change=0.5, max_rel_change=1.5, safety_factor=0.9):
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
            self.Q_Ik = self.Q_Ik.iloc[:, model.permutations]
            self.Q_Ik.index = self.Q_Ik.index.astype(float)
        else:
            self.Q_Ik = Q_Ik
        self.inputs = (self.Q_in, self.H_bc, self.Q_Ik)
        any_inputs = any(inp is not None for inp in self.inputs)
        if dt is None:
            dt = model._dt
        self.dt = dt
        self.max_iter = max_iter
        # Adaptive step size handling
        self.err = None
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.tol = tol
        self.min_rel_change = min_rel_change
        self.max_rel_change = max_rel_change
        self.safety_factor = safety_factor
        # TODO: This needs to be generalized
        self.state_variables = {'H_j' : 'j',
                                '_h_Ik' : 'Ik',
                                '_Q_ik' : 'ik',
                                '_Q_uk' : 'k',
                                '_Q_dk' : 'k'}
        if t_start is None:
            if any_inputs:
                self.t_start = min(i.index.min() for i in self.inputs if i is not None)
                self.model.t = self.t_start
            else:
                self.t_start = model.t
        if t_end is None:
            if any_inputs:
                self.t_end = max(i.index.max() for i in self.inputs if i is not None)
            else:
                self.t_end = np.inf
        # Progress bar checkpoints
        if np.isfinite(self.t_end):
            self._checkpoints = np.linspace(self.t_start, self.t_end)
        else:
            self._checkpoints = np.array([np.inf])
        self._checkpoint_num = 0
        # Create a sequence iterator
        if max_iter is None:
            self.steps = count()
        else:
            self.steps = range(max_iter)
        self.states = States()
        for state in self.state_variables:
            setattr(self.states, state, {})
            getattr(self.states, state).update({float(model.t) :
                                                np.copy(model.states[state])})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # TODO: Should be able to choose what to record
        for state, state_type in self.state_variables.items():
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
        self.model.load_state(state)

    def record_state(self, state_variables=()):
        # TODO: Should be able to choose what to record
        # if not state_variables:
        model = self.model
        state_variables = self.state_variables
        for state in state_variables:
            if not hasattr(self.states, state):
                setattr(self.states, state, {})
            getattr(self.states, state).update({float(model.t) :
                                                np.copy(getattr(model, state))})
        # TODO: Add ability to record error and retry attempts

    def print_progress(self, use_checkpoints=True):
        # Import current and ending time
        t = self.t
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
        # Prevent from going over 100%
        if t > t_end:
            t = t_end
        bar_len = 50
        progress_ratio = float(t) / float(t_end)
        progress_len = int(round(bar_len * progress_ratio))
        pct_finished = round(100.0 * progress_ratio, 1)
        bar = '=' * progress_len + '-' * (bar_len - progress_len)
        sys.stdout.write('\r[{0}] {1}{2}'.format(bar, pct_finished, '%'))
        sys.stdout.flush()

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

    def step(self, dt=None, subdivisions=1, retries=0, **kwargs):
        if dt is None:
            dt = self.dt
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
            err = (np.abs(states_coarse - states_fine)).sum()
        self.err = err
        # TODO: This will not save the dt needed for the next step
        if (retries) and (err is not None):
            tol = self.tol
            min_dt = self.min_dt
            dt = self.compute_step_size(dt, err=err)
            if (err > tol) and (dt > min_dt):
                self.load_state(initial_state)
                self.step(dt=dt, subdivisions=subdivisions, retries=retries-1, **kwargs)

    def _step(self, dt=None, **kwargs):
        # Import inputs
        Q_in = self.Q_in
        H_bc = self.H_bc
        Q_Ik = self.Q_Ik
        # Specify current timestamps
        t_next = self.t + dt
        # Infer if system is banded
        if not 'banded' in kwargs:
            banded = self.model.banded
        else:
            banded = kwargs.pop('banded')
        # Get superjunction runoff input
        if Q_in is not None:
            Q_in_index, Q_in_values = Q_in.index.values, Q_in.values
            Q_in_next = interpolate_sample(t_next, Q_in_index, Q_in_values)
        else:
            Q_in_next = None
        # Get head boundary conditions
        if H_bc is not None:
            H_bc_index, H_bc_values = H_bc.index.values, H_bc.values
            H_bc_next = interpolate_sample(t_next, H_bc_index, H_bc_values)
        else:
            H_bc_next = None
        # Get junction runoff input
        if Q_Ik is not None:
            Q_Ik_index, Q_Ik_values = Q_Ik.index.values, Q_Ik.values
            Q_Ik_next = interpolate_sample(t_next, Q_Ik_index, Q_Ik_values)
        else:
            Q_Ik_next = None
        # Step model forward with stepsize dt
        self.model.step(Q_in=Q_in_next, H_bc=H_bc_next, dt=dt, banded=banded,
                        **kwargs)

class States():
    def __init__(self):
        pass

    def __repr__(self):
        string = '[' + str(', '.join(self.__dict__.keys())) + ']'
        return string
