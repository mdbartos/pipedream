import numpy as np
from pipedream_solver.callbacks import BaseCallback

class BaseConvergenceManager(BaseCallback):
    def __init__(self, model, num_iter):
        self.model = model
        self.num_iter = num_iter
        self.iter_elapsed = 0

    def __on_step_start__(self, *args, **kwargs):
        if (self.iter_elapsed == 0):
            self.model.save_state()

    def __on_step_end__(self, *args, **kwargs):
        dt = kwargs['dt']
        self.iter_elapsed += 1
        while (self.iter_elapsed < self.num_iter):
            condition_met = self.convergence_condition()
            if not condition_met:
                self.H_j_prev = self.model.H_j.copy()
                self.model.iter_count -= 1
                self.model.t -= dt
                self.model._setup_step(*args, **kwargs)
                self.model._solve_step(*args, **kwargs)
                self.H_j_next = self.model.H_j.copy()
                self.iter_elapsed += 1
            else:
                break
        self.iter_elapsed = 0

    def convergence_condition(self):
        return False

class DefaultConvergenceManager(BaseConvergenceManager):
    def __init__(self, model, head_tol, num_iter):
        self.model = model
        self.head_tol = head_tol
        self.num_iter = num_iter
        self.iter_elapsed = 0

    def __on_step_start__(self, *args, **kwargs):
        self.H_j_prev = self.model.H_j.copy()
        if (self.iter_elapsed == 0):
            self.model.save_state()

    def __on_step_end__(self, *args, **kwargs):
        dt = kwargs['dt']
        self.H_j_next = self.model.H_j.copy()
        self.iter_elapsed += 1
        while (self.iter_elapsed < self.num_iter):
            condition_met = self.convergence_condition()
            if not condition_met:
                self.H_j_prev = self.model.H_j.copy()
                self.model.iter_count -= 1
                self.model.t -= dt
                self.model._setup_step(*args, **kwargs)
                self.model._solve_step(*args, **kwargs)
                self.H_j_next = self.model.H_j.copy()
                self.iter_elapsed += 1
            else:
                break
        self.iter_elapsed = 0

    def convergence_condition(self):
        head_tol = self.head_tol
        H_j_prev = self.H_j_prev
        H_j_next = self.H_j_next
        residual = np.abs(H_j_next - H_j_prev)
        condition_met = (residual < head_tol).all()
        return condition_met
