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

class Simulation():
    def __init__(self, model, Q_in=None, H_bc=None, Q_Ik=None, t_start=None,
                 t_end=None, dt=None, max_iter=None):
        self.model = model
        self.Q_in = Q_in
        self.H_bc = H_bc
        self.Q_Ik = Q_Ik
        self.inputs = (self.Q_in, self.H_bc, self.Q_Ik)
        any_inputs = any(inp is not None for inp in self.inputs)
        if dt is None:
            dt = model._dt
        self.dt = dt
        self.max_iter = max_iter
        # TODO: This needs to be generalized
        self.state_variables = ('H_j', '_h_Ik', '_Q_ik')
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
        self.t = self.t_start
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
        for state in self.state_variables:
            d = getattr(self.states, state)
            df = pd.DataFrame.from_dict(d, orient='index')
            setattr(self.states, state, df)

    def load_state(self, state={}):
        self.model.load_state(state)
        self.t = model.t

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

    def print_progress(self):
        t = self.t
        t_end = self.t_end
        bar_len = 50
        progress_ratio = float(t) / float(t_end)
        progress_len = int(round(bar_len * progress_ratio))
        pct_finished = round(100.0 * progress_ratio, 1)
        bar = '=' * progress_len + '-' * (bar_len - progress_len)
        sys.stdout.write('\r[{0}] {1}{2}'.format(bar, pct_finished, '%'))
        sys.stdout.flush()

    def step(self, dt, **kwargs):
        Q_in = self.Q_in
        H_bc = self.H_bc
        Q_Ik = self.Q_Ik
        model = self.model
        t_end = self.t_end
        t_next = self.model.t + dt
        if Q_in is not None:
            Q_in_index, Q_in_values = Q_in.index.values, Q_in.values
            Q_in_next = interpolate_sample(t_next, Q_in_index, Q_in_values)
        else:
            Q_in_next = None
        if H_bc is not None:
            H_bc_index, H_bc_values = H_bc.index.values, H_bc.values
            H_bc_next = interpolate_sample(t_next, H_bc_index, H_bc_values)
        else:
            H_bc_next = None
        if Q_Ik is not None:
            Q_Ik_index, Q_Ik_values = Q_Ik.index.values, Q_Ik.values
            Q_Ik_next = interpolate_sample(t_next, Q_Ik_index, Q_Ik_values)
        else:
            Q_Ik_next = None
        self.model.step(Q_in=Q_in_next, H_bc=H_bc_next, dt=dt, **kwargs)
        self.t = model.t

class States():
    def __init__(self):
        pass

    def __repr__(self):
        string = '[' + str(', '.join(self.__dict__.keys())) + ']'
        return string
