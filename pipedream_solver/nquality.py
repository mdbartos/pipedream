import numpy as np
import pandas as pd
from numba import njit
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

class QualityBuilder():
    """
    An implicit, staggered-grid water quality solver based on the 1D
    advection-reaction-diffusion equations.

    Inputs:
    -----------
    hydraulics: pipedream_solver.hydraulics.SuperLink
        A SuperLink instance describing the system hydraulics.

    superjunction_params: pd.DataFrame
        Table containing superjunction water quality parameters.
        The following fields are required:

        |-------+-------+-------+-----------------------------------------------------------|
        | Field | Type  | Unit  | Description                                               |
        |-------+-------+-------+-----------------------------------------------------------|
        | K     | float | 1/s   | First order reaction constant in superjunction            |
        | c_0   | float | */m^3 | Initial contaminant concentration in superjunction        |
        | bc    | bool  | -     | Indicates contaminant boundary condition at superjunction |
        |-------+-------+-------+-----------------------------------------------------------|

    superlink_params: pd.DataFrame
        Table containing superlink water quality parameters.
        The following fields are required:

        |-------+-------+------+------------------------------------------------------------------|
        | Field | Type  | Unit | Description                                                      |
        |---------+-------+------+----------------------------------------------------------------|
        | dx_uk | float | m    | Distance from the upstream end of the superlink to mixing zone   |
        | dx_dk | float | m    | Distance from the downstream end of the superlink to mixing zone |
        | D_uk  | float | m    | Diffusion constant of upstream end of superlink                  |
        | D_dk  | float | m    | Diffusion constant of downstream end of superlink                |
        |---------+-------+-------+---------------------------------------------------------------|

        If internal link and junction parameters are not provided (arguments 3 and 4), the
        following fields are required:

        |-------+-------+-------+---------------------------------------------------------------|
        | Field | Type  | Unit  | Description                                                   |
        |-------+-------+-------+---------------------------------------------------------------|
        | K     | float | 1/s   | First order reaction constant in internal links/junctions     |
        | D     | float | m^2/s | Diffusion constant in internal links/junctions                |
        | c_0   | float | */m^3 | Initial contaminant concentration in internal links/junctions |
        |-------+-------+-------+---------------------------------------------------------------|

    junction_params: pd.DataFrame (optional)
        Table containing junction water quality parameters.
        The following fields are required.

        |-------+-------+-------+-----------------------------------------------|
        | Field | Type  | Unit  | Description                                   |
        |-------+-------+-------+-----------------------------------------------|
        | K     | float | 1/s   | First order reaction constant in junction     |
        | D     | float | m^2/s | Diffusion constant in junction                |
        | c_0   | float | g/m^3 | Initial contaminant concentration in junction |
        |-------+-------+-------+-----------------------------------------------|

    link_params: pd.DataFrame (optional)
        Table containing link water quality parameters.
        The following fields are required.

        |-------+-------+-------+-------------------------------------------|
        | Field | Type  | Unit  | Description                               |
        |-------+-------+-------+-------------------------------------------|
        | K     | float | 1/s   | First order reaction constant in links    |
        | D     | float | m^2/s | Diffusion constant in link                |
        | c_0   | float | g/m^3 | Initial contaminant concentration in link |
        |-------+-------+-------+-------------------------------------------|

    c_min: float
        Minimum allowed contaminant concentration (default 0).

    c_max: float
        Maximum allowed contaminant concentration (default np.inf).

    Methods:
    -----------
    step : Advance model to next time step, computing water quality states

    Attributes:
    -----------
    t        : Current time (s)
    c_j      : Contaminant concentration in superjunctions (*/m^3)
    c_Ik     : Contaminant concentration in junctions (*/m^3)
    c_ik     : Contaminant concentration in links (*/m^3)
    c_uk     : Contaminant concentration entering upstream end of superlinks (*/m^3)
    c_dk     : Contaminant concentration exiting downstream end of superlinks (*/m^3)
    """
    def __init__(self, hydraulics, superjunction_params, superlink_params,
                 junction_params=None, link_params=None, c_min=0.0, c_max=np.inf):
        self.hydraulics = hydraulics
        self._ki = self.hydraulics._ki
        self._kI = self.hydraulics._kI
        self._K_j = superjunction_params['K'].values.astype(float)
        self._c_j = superjunction_params['c_0'].values.astype(float)
        self.bc = superjunction_params['bc'].values.astype(bool)
        if junction_params is not None:
            self._K_Ik = junction_params['K'].values.astype(float)
            self._c_Ik = junction_params['c_0'].values.astype(float)
        else:
            self._K_Ik = superlink_params['K'].values[self._kI].astype(float)
            self._c_Ik = superlink_params['c_0'].values[self._kI].astype(float)
        if link_params is not None:
            self._D_ik = link_params['D'].values.astype(float)
            self._K_ik = link_params['K'].values.astype(float)
            self._c_ik = link_params['c_0'].values.astype(float)
        else:
            self._D_ik = superlink_params['D'].values[self._ki].astype(float)
            self._K_ik = superlink_params['K'].values[self._ki].astype(float)
            self._c_ik = superlink_params['c_0'].values[self._ki].astype(float)
        self._dx_uk = superlink_params['dx_uk'].values.astype(float)
        self._dx_dk = superlink_params['dx_dk'].values.astype(float)
        if 'D_uk' in superlink_params.columns:
            self._D_uk = superlink_params['D_uk'].values.astype(float)
        else:
            self._D_uk = superlink_params['D'].values.astype(float)
        if 'D_dk' in superlink_params.columns:
            self._D_dk = superlink_params['D_dk'].values.astype(float)
        else:
            self._D_dk = superlink_params['D'].values.astype(float)
        # Structural parameters of hydraulic model
        self._forward_I_i = self.hydraulics.forward_I_i       # Index of link after junction Ik
        self._backward_I_i = self.hydraulics.backward_I_i     # Index of link before junction Ik
        self._is_start = self.hydraulics._is_start
        self._is_end = self.hydraulics._is_end
        self._ik = self.hydraulics._ik
        self._I = self.hydraulics._I
        self._A_SIk = self.hydraulics._A_SIk                 # Surface area of junction Ik
        self._I_1k = self.hydraulics._I_1k                # Index of first junction in each superlink
        self._i_1k = self.hydraulics._i_1k                # Index of first link in each superlink
        self._I_Nk = self.hydraulics._I_Nk           # Index of penultimate junction in each superlink
        self._I_Np1k = self.hydraulics._I_Np1k       # Index of ultimate junction in each superlink
        self._i_nk = self.hydraulics._i_nk           # Index of last link in each superlink
        self.NK = self.hydraulics.NK
        self.M = self.hydraulics.M
        self.nk = self.hydraulics.nk
        self._k = self.hydraulics._k                     # Superlink indices
        self._J_uk = self.hydraulics._J_uk           # Index of superjunction upstream of superlink k
        self._J_dk = self.hydraulics._J_dk          # Index of superjunction downstream of superlink k
        self._J_uo = self.hydraulics._J_uo
        self._J_do = self.hydraulics._J_do
        self._J_uw = self.hydraulics._J_uw
        self._J_dw = self.hydraulics._J_dw
        self._J_up = self.hydraulics._J_up
        self._J_dp = self.hydraulics._J_dp
        self.n_o = self.hydraulics.n_o              # Number of orifices in system
        self.n_w = self.hydraulics.n_w              # Number of weirs in system
        self.n_p = self.hydraulics.n_p              # Number of pumps in system
        self._sparse = self.hydraulics._sparse        # Use sparse data structures (y/n)
        self.bandwidth = self.hydraulics.bandwidth
        # Import instantaneous states of hydraulic model
        # self.import_hydraulic_states()
        # New states
        self._alpha_ik = np.zeros(self._ik.size)
        self._beta_ik = np.zeros(self._ik.size)
        self._chi_ik = np.zeros(self._ik.size)
        self._gamma_ik = np.zeros(self._ik.size)
        self._kappa_Ik = np.zeros(self._I.size)
        self._lambda_Ik = np.zeros(self._I.size)
        self._mu_Ik = np.zeros(self._I.size)
        self._eta_Ik = np.zeros(self._I.size)
        self._T_ik = np.zeros(self._ik.size)
        self._U_Ik = np.zeros(self._I.size)
        self._V_Ik = np.zeros(self._I.size)
        self._O_ik = np.zeros(self._ik.size)
        self._X_Ik = np.zeros(self._I.size)
        self._Y_Ik = np.zeros(self._I.size)
        self._Z_Ik = np.zeros(self._I.size)
        self._W_Ik = np.zeros(self._I.size)
        self._X_uk = np.zeros(self.NK)
        self._Y_uk = np.zeros(self.NK)
        self._Z_uk = np.zeros(self.NK)
        self._U_dk = np.zeros(self.NK)
        self._V_dk = np.zeros(self.NK)
        self._W_dk = np.zeros(self.NK)
        self._rho_uk = np.zeros(self.NK)
        self._tau_uk = np.zeros(self.NK)
        self._omega_uk = np.zeros(self.NK)
        self._rho_dk = np.zeros(self.NK)
        self._tau_dk = np.zeros(self.NK)
        self._omega_dk = np.zeros(self.NK)
        self._F_jj = np.zeros(self.M, dtype=float)
        self._O_diag = np.zeros(self.M, dtype=float)
        self._W_diag = np.zeros(self.M, dtype=float)
        self._P_diag = np.zeros(self.M, dtype=float)
        self.A = np.zeros((self.M, self.M))
        self.O = np.zeros((self.M, self.M))
        self.W = np.zeros((self.M, self.M))
        self.P = np.zeros((self.M, self.M))
        self.D = np.zeros(self.M)
        self.b = np.zeros(self.M)
        self._c_uk = 0.5 * self._c_j[self._J_uk] + 0.5 * self._c_Ik[self._I_1k]
        self._c_dk = 0.5 * self._c_j[self._J_dk] + 0.5 * self._c_Ik[self._I_Np1k]
        self._c_1k = self._c_Ik[self._I_1k]
        self._c_Np1k = self._c_Ik[self._I_Np1k]
        self._c_min = c_min
        self._c_max = c_max
        self.step(dt=1e-6)

    # TODO: It might be safer to have these as @properties
    def import_hydraulic_states(self):
        self._H_j_next = self.hydraulics.H_j
        self._h_Ik_next = self.hydraulics.h_Ik
        self._Q_ik_next = self.hydraulics.Q_ik
        self._Q_uk_next = self.hydraulics.Q_uk
        self._Q_dk_next = self.hydraulics.Q_dk
        self._H_j_prev = self.hydraulics.states['H_j']
        self._h_Ik_prev = self.hydraulics.states['h_Ik']
        self._Q_ik_prev = self.hydraulics.states['Q_ik']
        self._Q_uk_prev = self.hydraulics.states['Q_uk']
        self._Q_dk_prev = self.hydraulics.states['Q_dk']
        if self.n_o:
            self._Q_o_next = self.hydraulics.Q_o
            self._Q_o_prev = self.hydraulics.states['Q_o']
        if self.n_w:
            self._Q_w_next = self.hydraulics.Q_w
            self._Q_w_prev = self.hydraulics.states['Q_w']
        if self.n_p:
            self._Q_p_next = self.hydraulics.Q_p
            self._Q_p_prev = self.hydraulics.states['Q_p']
        self._u_ik_next = self.hydraulics._u_ik
        self._u_Ik_next = self.hydraulics._u_Ik
        self._u_Ip1k_next = self.hydraulics._u_Ip1k
        self._dx_ik_next = self.hydraulics._dx_ik
        self._A_ik_next = self.hydraulics._A_ik
        self._B_ik_next = self.hydraulics._B_ik
        self._A_uk_next = self.hydraulics._A_uk
        self._A_dk_next = self.hydraulics._A_dk
        self._A_sj = self.hydraulics._A_sj
        self._V_sj = self.hydraulics._V_sj

    @property
    def c_j(self):
        return self._c_j

    @c_j.setter
    def c_j(self, value):
        self._c_j = np.asarray(value)

    @property
    def c_Ik(self):
        return self._c_Ik

    @c_Ik.setter
    def c_Ik(self, value):
        self._c_Ik = np.asarray(value)

    @property
    def c_ik(self):
        return self._c_ik

    @c_ik.setter
    def c_ik(self, value):
        self._c_ik = np.asarray(value)

    @property
    def c_uk(self):
        return self._c_uk

    @c_uk.setter
    def c_uk(self, value):
        self._c_uk = np.asarray(value)

    @property
    def c_dk(self):
        return self._c_dk

    @c_dk.setter
    def c_dk(self, value):
        self._c_dk = np.asarray(value)

    @property
    def _dt(self):
        return self.hydraulics.t - self.hydraulics.states['t']

    def link_coeffs(self, _dt=None, _u_j_frac=0.0, first_iter=True):
        """
        Compute link momentum coefficients: a_ik, b_ik, c_ik and P_ik.
        """
        # Import instance variables
        _u_Ik_next = self._u_Ik_next         # Flow velocity at junction Ik
        _u_Ip1k_next = self._u_Ip1k_next     # Flow velocity at junction I + 1k
        _dx_ik_next = self._dx_ik_next       # Length of link ik
        _D_ik = self._D_ik
        _K_ik = self._K_ik
        _c_ik = self._c_ik
        _Q_uk_next = self._Q_uk_next
        _A_uk_next = self._A_uk_next
        _Q_dk_next = self._Q_dk_next
        _A_dk_next = self._A_dk_next
        _u_ik_next = self._u_ik_next
        _D_uk = self._D_uk
        _D_dk = self._D_dk
        _dx_uk = self._dx_uk
        _dx_dk = self._dx_dk
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _I_1k = self._I_1k
        _I_Np1k = self._I_Np1k
        _c_uk = self._c_uk
        _c_dk = self._c_dk
        _alpha_ik = self._alpha_ik
        _beta_ik = self._beta_ik
        _chi_ik = self._chi_ik
        _gamma_ik = self._gamma_ik
        # If time step not specified, use instance time
        if _dt is None:
            _dt = self._dt
        # Compute link coefficients
        # TODO: This should happen in hydraulic solver
        _u_1k = (safe_divide_vec(_Q_uk_next, _A_uk_next)
                 + _u_ik_next[_i_1k]) / 2
        _u_Np1k = (safe_divide_vec(_Q_dk_next, _A_dk_next)
                   + _u_ik_next[_i_nk]) / 2
        # TODO: Redundant computations
        _u_Ik_next[np.cumsum(self.nk) - self.nk[0]] = _u_1k
        _u_Ip1k_next[np.cumsum(self.nk) - 1] = _u_Np1k
        _u_Ik_hat = -np.maximum(_u_Ik_next, 0.)
        _u_Ip1k_hat = -np.maximum(-_u_Ip1k_next, 0.)
        _u_1k_hat = -np.maximum(_u_1k, 0.)
        _u_Np1k_hat = -np.maximum(-_u_Np1k, 0.)
        _alpha_ik = alpha_ik(_u_Ik_hat, _dx_ik_next, _D_ik)
        _beta_ik = beta_ik(_dt, _D_ik, _dx_ik_next, _K_ik, _u_Ik_hat, _u_Ip1k_hat)
        _chi_ik = chi_ik(_u_Ip1k_hat, _dx_ik_next, _D_ik)
        _gamma_ik = gamma_ik(_dt, _c_ik)
        # Compute link coefficients for boundaries
        _alpha_uk = alpha_ik(_u_j_frac * _u_1k_hat, _dx_uk, _D_uk)
        _beta_uk = beta_ik(_dt, _D_uk, _dx_uk, 0.0, _u_j_frac * _u_1k_hat, _u_1k_hat)
        _chi_uk = chi_ik(_u_1k_hat, _dx_uk, _D_uk)
        _gamma_uk = gamma_ik(_dt, _c_uk)
        _alpha_dk = alpha_ik(_u_Np1k_hat, _dx_dk, _D_dk)
        _beta_dk = beta_ik(_dt, _D_dk, _dx_dk, 0.0, _u_Np1k_hat, _u_j_frac * _u_Np1k_hat)
        _chi_dk = chi_ik(_u_j_frac * _u_Np1k_hat, _dx_dk, _D_dk)
        _gamma_dk = gamma_ik(_dt, _c_dk)
        # Export to instance variables
        self._alpha_ik = _alpha_ik
        self._beta_ik = _beta_ik
        self._chi_ik = _chi_ik
        self._gamma_ik = _gamma_ik
        self._alpha_uk = _alpha_uk
        self._beta_uk = _beta_uk
        self._chi_uk = _chi_uk
        self._gamma_uk = _gamma_uk
        self._alpha_dk = _alpha_dk
        self._beta_dk = _beta_dk
        self._chi_dk = _chi_dk
        self._gamma_dk = _gamma_dk

    def node_coeffs(self, _Q_0Ik=None, _c_0Ik=None, _dt=None, first_iter=True):
        """
        Compute nodal continuity coefficients: D_Ik and E_Ik.
        """
        # Import instance variables
        _forward_I_i = self._forward_I_i       # Index of link after junction Ik
        _backward_I_i = self._backward_I_i     # Index of link before junction Ik
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
        _c_ik_prev = self._c_ik
        _B_ik_next = self._B_ik_next
        _A_ik_next = self._A_ik_next
        _K_ik = self._K_ik
        _dx_ik_next = self._dx_ik_next
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no nodal input specified, use zero input
        if _Q_0Ik is None:
            _Q_0Ik = np.zeros(_h_Ik_next.size)
        if _c_0Ik is None:
            _c_0Ik = np.zeros(_h_Ik_next.size)
        # Compute continuity coefficients
        numba_node_coeffs(_kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                          _Q_ik_next, _h_Ik_next, _h_Ik_prev, _c_Ik_prev, _c_ik_prev,
                          _Q_uk_next, _Q_dk_next, _c_0Ik, _Q_0Ik, _A_SIk, _K_Ik, _K_ik,
                          _B_ik_next, _A_ik_next, _dx_ik_next, _forward_I_i, _backward_I_i,
                          _is_start, _is_end, _kI, _dt)
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
        _kappa_Ik = self._kappa_Ik
        _lambda_Ik = self._lambda_Ik
        _mu_Ik = self._mu_Ik
        _eta_Ik = self._eta_Ik
        _I_1k = self._I_1k                # Index of first junction in each superlink
        _i_1k = self._i_1k                # Index of first link in each superlink
        _I_Nk = self._I_Nk                # Index of penultimate junction in each superlink
        _i_nk = self._i_nk                # Index of last link in each superlink
        NK = self.NK
        _X_uk = self._X_uk
        _Y_uk = self._Y_uk
        _Z_uk = self._Z_uk
        _U_dk = self._U_dk
        _V_dk = self._V_dk
        _W_dk = self._W_dk
        _alpha_uk = self._alpha_uk
        _beta_uk = self._beta_uk
        _chi_uk = self._chi_uk
        _gamma_uk = self._gamma_uk
        _alpha_dk = self._alpha_dk
        _beta_dk = self._beta_dk
        _chi_dk = self._chi_dk
        _gamma_dk = self._gamma_dk
        _rho_uk = self._rho_uk
        _tau_uk = self._tau_uk
        _omega_uk = self._omega_uk
        _rho_dk = self._rho_dk
        _tau_dk = self._tau_dk
        _omega_dk = self._omega_dk
        # Compute boundary coefficients
        numba_boundary_coefficients(_X_uk, _Y_uk, _Z_uk, _U_dk, _V_dk, _W_dk,
                                    _X_Ik, _Y_Ik, _Z_Ik, _U_Ik, _V_Ik, _W_Ik,
                                    _kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                                    NK, _I_1k, _I_Nk)
        # Compute boundary advection/diffusion coefficients
        _theta_uk = theta_uk(_beta_uk, _chi_uk)
        _sigma_uk = sigma_uk(_alpha_uk, _chi_uk)
        _xi_uk = xi_uk(_gamma_uk, _chi_uk)
        _theta_dk = theta_dk(_beta_dk, _alpha_dk)
        _sigma_dk = sigma_dk(_chi_dk, _alpha_dk)
        _xi_dk = xi_dk(_gamma_dk, _alpha_dk)
        # Compute boundary transport coefficients
        _D_k_star = D_k_star(_X_uk, _U_dk, _W_dk, _Z_uk, _theta_uk, _theta_dk)
        _rho_uk = rho_uk(_U_dk, _X_uk, _Z_uk, _W_dk, _theta_dk, _sigma_uk, _D_k_star)
        _tau_uk = tau_uk(_Z_uk, _sigma_dk, _D_k_star)
        _omega_uk = omega_uk(_Z_uk, _V_dk, _U_dk, _Y_uk, _X_uk, _W_dk,
                             _theta_dk, _xi_uk, _xi_dk, _D_k_star)
        _rho_dk = rho_dk(_W_dk, _sigma_uk, _D_k_star)
        _tau_dk = tau_dk(_X_uk, _U_dk, _W_dk, _Z_uk, _theta_uk, _sigma_dk, _D_k_star)
        _omega_dk = omega_dk(_W_dk, _Y_uk, _X_uk, _V_dk, _Z_uk, _U_dk,
                             _theta_uk, _xi_uk, _xi_dk, _D_k_star)
        # Export instance variables
        self._X_uk = _X_uk
        self._Y_uk = _Y_uk
        self._Z_uk = _Z_uk
        self._U_dk = _U_dk
        self._V_dk = _V_dk
        self._W_dk = _W_dk
        self._theta_uk = _theta_uk
        self._sigma_uk = _sigma_uk
        self._xi_uk = _xi_uk
        self._theta_dk = _theta_dk
        self._sigma_dk = _sigma_dk
        self._xi_dk = _xi_dk
        self._rho_uk = _rho_uk
        self._tau_uk = _tau_uk
        self._omega_uk = _omega_uk
        self._rho_dk = _rho_dk
        self._tau_dk = _tau_dk
        self._omega_dk = _omega_dk

    def sparse_matrix_equations(self, c_bc=None, _Q_0j=None, _c_0j=None, u=None, _dt=None,
                                implicit=True, first_time=False):
        """
        Construct sparse matrices A, O, W, P and b.
        """
        # Import instance variables
        _k = self._k                     # Superlink indices
        _J_uk = self._J_uk               # Index of superjunction upstream of superlink k
        _J_dk = self._J_dk               # Index of superjunction downstream of superlink k
        _rho_uk = self._rho_uk
        _tau_uk = self._tau_uk
        _omega_uk = self._omega_uk
        _rho_dk = self._rho_dk
        _tau_dk = self._tau_dk
        _omega_dk = self._omega_dk
        _F_jj = self._F_jj
        _A_sj = self._A_sj               # Surface area of superjunction j
        _V_sj = self._V_sj
        _c_j = self._c_j
        _Q_uk_next = self._Q_uk_next
        _Q_dk_next = self._Q_dk_next
        _K_j = self._K_j
        NK = self.NK
        n_o = self.n_o                   # Number of orifices in system
        n_w = self.n_w                   # Number of weirs in system
        n_p = self.n_p                   # Number of pumps in system
        A = self.A
        if n_o:
            O = self.O
            _J_uo = self._J_uo               # Index of superjunction upstream of orifice o
            _J_do = self._J_do               # Index of superjunction upstream of orifice o
            _O_diag = self._O_diag           # Diagonal elements of matrix O
            _Q_o_next = self._Q_o_next
        if n_w:
            W = self.W
            _J_uw = self._J_uw               # Index of superjunction upstream of weir w
            _J_dw = self._J_dw               # Index of superjunction downstream of weir w
            _W_diag = self._W_diag           # Diagonal elements of matrix W
            _Q_w_next = self._Q_w_next
        if n_p:
            P = self.P
            _J_up = self._J_up               # Index of superjunction upstream of pump p
            _J_dp = self._J_dp               # Index of superjunction downstream of pump p
            _P_diag = self._P_diag           # Diagonal elements of matrix P
            _Q_p_next = self._Q_p_next
        M = self.M                       # Number of superjunctions in system
        _H_j_next = self._H_j_next                   # Head at superjunction j
        _H_j_prev = self._H_j_prev                   # Head at superjunction j
        bc = self.bc                     # Superjunction j has a fixed boundary condition (y/n)
        D = self.D                       # Vector for storing chi coefficients
        b = self.b                       # Right-hand side vector
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no boundary head specified, use current superjunction head
        if c_bc is None:
            c_bc = self._c_j
        # If no flow input specified, assume zero external inflow
        if _Q_0j is None:
            _Q_0j = 0
        if _c_0j is None:
            _c_0j = 0
        # If no control input signal specified assume zero input
        if u is None:
            u = 0
        # Clear old data
        _F_jj.fill(0)
        D.fill(0)
        numba_clear_off_diagonals(A, bc, _J_uk, _J_dk, NK)
        # Create A matrix
        numba_create_A_matrix(A, _F_jj, bc, _J_uk, _J_dk, _rho_uk, _rho_dk, _tau_uk, _tau_dk,
                              _Q_uk_next, _Q_dk_next, _A_sj, _V_sj, _H_j_next, _dt, _K_j, M, NK)
        # Create D vector
        numba_add_at(D, _J_uk, -_omega_uk * _Q_uk_next)
        numba_add_at(D, _J_dk, _omega_dk * _Q_dk_next)
        # Compute control matrix
        if n_o:
            _omega_o = (_Q_o >= 0).astype(float)
            _O_diag.fill(0)
            numba_clear_off_diagonals(O, bc, _J_uo, _J_do, n_o)
            numba_create_OWP_matrix(O, _O_diag, bc, _J_uo, _J_do, _omega_o,
                                    _Q_o_next, M, n_o)
        if n_w:
            _omega_w = (_Q_w >= 0).astype(float)
            _W_diag.fill(0)
            numba_clear_off_diagonals(W, bc, _J_uw, _J_dw, n_w)
            numba_create_OWP_matrix(W, _W_diag, bc, _J_uw, _J_dw, _omega_w,
                                    _Q_w_next, M, n_w)
        if n_p:
            _omega_p = (_Q_p >= 0).astype(float)
            _P_diag.fill(0)
            numba_clear_off_diagonals(P, bc, _J_up, _J_dp, n_p)
            numba_create_OWP_matrix(P, _P_diag, bc, _J_up, _J_dp, _omega_p,
                                    _Q_p_next, M, n_p)
        b.fill(0)
        b = (_A_sj * _H_j_prev * _c_j / _dt) + (_Q_0j * _c_0j) + D
        # Ensure boundary condition is specified
        b[bc] = c_bc[bc]
        # Export instance variables
        self.D = D
        self.b = b

    def solve_sparse_matrix(self, u=None, implicit=True):
        """
        Solve sparse system Ax = b for superjunction heads at time t + dt.
        """
        # Import instance variables
        A = self.A                    # Superlink/superjunction matrix
        b = self.b                    # Right-hand side vector
        O = self.O                    # Orifice matrix
        W = self.W                    # Weir matrix
        P = self.P                    # Pump matrix
        n_o = self.n_o                # Number of orifices
        n_w = self.n_w                # Number of weirs
        n_p = self.n_p                # Number of pumps
        _sparse = self._sparse        # Use sparse data structures (y/n)
        _c_min = self._c_min
        _c_max = self._c_max
        # Does the system have control assets?
        has_control = n_o + n_w + n_p
        # Get right-hand size
        if has_control:
            if implicit:
                l = A + O + W + P
                r = b
            else:
                # TODO: Broken
                # l = A
                # r = b + np.squeeze(B @ u)
                raise NotImplementedError
        else:
            l = A
            r = b
        if _sparse:
            _c_j_next = scipy.sparse.linalg.spsolve(l, r)
        else:
            _c_j_next = scipy.linalg.solve(l, r)
        assert np.isfinite(_c_j_next).all()
        _c_j_next = np.maximum(_c_j_next, _c_min)
        _c_j_next = np.minimum(_c_j_next, _c_max)
        # Export instance variables
        self._c_j = _c_j_next

    def solve_banded_matrix(self, u=None, implicit=True):
        # Import instance variables
        A = self.A                    # Superlink/superjunction matrix
        b = self.b                    # Right-hand side vector
        O = self.O                    # Orifice matrix
        W = self.W                    # Weir matrix
        P = self.P                    # Pump matrix
        n_o = self.n_o                # Number of orifices
        n_w = self.n_w                # Number of weirs
        n_p = self.n_p                # Number of pumps
        _sparse = self._sparse        # Use sparse data structures (y/n)
        _c_min = self._c_min
        _c_max = self._c_max
        bandwidth = self.bandwidth
        M = self.M
        # Does the system have control assets?
        has_control = n_o + n_w + n_p
        # Get right-hand size
        if has_control:
            if implicit:
                l = A + O + W + P
                r = b
            else:
                raise NotImplementedError
        else:
            l = A
            r = b
        AB = numba_create_banded(l, bandwidth, M)
        _c_j_next = scipy.linalg.solve_banded((bandwidth, bandwidth), AB, r,
                                              check_finite=False, overwrite_ab=True)
        assert np.isfinite(_c_j_next).all()
        _c_j_next = np.maximum(_c_j_next, _c_min)
        _c_j_next = np.minimum(_c_j_next, _c_max)
        # Export instance variables
        self._c_j = _c_j_next

    def solve_boundary_states(self):
        """
        Solve for concentrations at superlink boundaries
        """
        _c_j = self._c_j
        _c_uk = self._c_uk
        _c_dk = self._c_dk
        _J_uk = self._J_uk               # Index of superjunction upstream of superlink k
        _J_dk = self._J_dk               # Index of superjunction downstream of superlink k
        _rho_uk = self._rho_uk
        _tau_uk = self._tau_uk
        _omega_uk = self._omega_uk
        _rho_dk = self._rho_dk
        _tau_dk = self._tau_dk
        _omega_dk = self._omega_dk
        _c_min = self._c_min
        _c_max = self._c_max
        # Solve for boundary flow concentrations
        _c_uk_next = _rho_uk * _c_j[_J_uk] + _tau_uk * _c_j[_J_dk] + _omega_uk
        _c_dk_next = _rho_dk * _c_j[_J_uk] + _tau_dk * _c_j[_J_dk] + _omega_dk
        _c_1k = 2 * _c_uk_next - _c_j[_J_uk]
        _c_Np1k = 2 * _c_dk_next - _c_j[_J_dk]
        # Enforce non-negative concentration
        _c_uk_next = np.maximum(_c_uk_next, _c_min)
        _c_dk_next = np.maximum(_c_dk_next, _c_min)
        _c_1k = np.maximum(_c_1k, _c_min)
        _c_Np1k = np.maximum(_c_Np1k, _c_min)
        _c_uk_next = np.minimum(_c_uk_next, _c_max)
        _c_dk_next = np.minimum(_c_dk_next, _c_max)
        _c_1k = np.minimum(_c_1k, _c_max)
        _c_Np1k = np.minimum(_c_Np1k, _c_max)
        # Export instance variables
        self._c_1k = _c_1k
        self._c_Np1k = _c_Np1k
        self._c_uk = _c_uk_next
        self._c_dk = _c_dk_next

    def solve_internals_backwards(self, subcritical_only=False):
        """
        Solve for internal states of each superlink in the backward direction.
        """
        # Import instance variables
        _I_1k = self._I_1k                  # Index of first junction in superlink k
        _i_1k = self._i_1k                  # Index of first link in superlink k
        nk = self.nk
        NK = self.NK
        _c_Ik = self._c_Ik                  # Depth at junction Ik
        _c_ik = self._c_ik                  # Flow rate at link ik
        _U_Ik = self._U_Ik                  # Forward recurrence coefficient
        _V_Ik = self._V_Ik                  # Forward recurrence coefficient
        _W_Ik = self._W_Ik                  # Forward recurrence coefficient
        _X_Ik = self._X_Ik                  # Backward recurrence coefficient
        _Y_Ik = self._Y_Ik                  # Backward recurrence coefficient
        _Z_Ik = self._Z_Ik                  # Backward recurrence coefficient
        _c_1k = self._c_1k                  # Depth at upstream end of superlink k
        _c_Np1k = self._c_Np1k              # Depth at downstream end of superlink k
        _c_min = self._c_min
        _c_max = self._c_max
        # Solve internals
        numba_solve_internals(_c_Ik, _c_ik, _c_1k, _c_Np1k, _U_Ik, _V_Ik, _W_Ik,
                              _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, nk, NK,
                              _c_min, _c_max, first_link_backwards=True)
        # TODO: Temporary
        assert np.isfinite(_c_Ik).all()
        assert np.isfinite(_c_ik).all()
        # Ensure non-negative depths?
        _c_Ik[_c_Ik < _c_min] = _c_min
        _c_ik[_c_ik < _c_min] = _c_min
        _c_Ik[_c_Ik > _c_max] = _c_max
        _c_ik[_c_ik > _c_max] = _c_max
        # Export instance variables
        self._c_Ik = _c_Ik
        self._c_ik = _c_ik

    def solve_internals_forwards(self, subcritical_only=False):
        """
        Solve for internal states of each superlink in the backward direction.
        """
        # Import instance variables
        _I_1k = self._I_1k                  # Index of first junction in superlink k
        _i_1k = self._i_1k                  # Index of first link in superlink k
        nk = self.nk
        NK = self.NK
        _c_Ik = self._c_Ik                  # Depth at junction Ik
        _c_ik = self._c_ik                  # Flow rate at link ik
        _U_Ik = self._U_Ik                  # Forward recurrence coefficient
        _V_Ik = self._V_Ik                  # Forward recurrence coefficient
        _W_Ik = self._W_Ik                  # Forward recurrence coefficient
        _X_Ik = self._X_Ik                  # Backward recurrence coefficient
        _Y_Ik = self._Y_Ik                  # Backward recurrence coefficient
        _Z_Ik = self._Z_Ik                  # Backward recurrence coefficient
        _c_1k = self._c_1k                  # Depth at upstream end of superlink k
        _c_Np1k = self._c_Np1k              # Depth at downstream end of superlink k
        _c_min = self._c_min
        _c_max = self._c_max
        # Solve internals
        numba_solve_internals(_c_Ik, _c_ik, _c_1k, _c_Np1k, _U_Ik, _V_Ik, _W_Ik,
                              _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, nk, NK,
                              _c_min, _c_max, first_link_backwards=False)
        # TODO: Temporary
        assert np.isfinite(_c_Ik).all()
        assert np.isfinite(_c_ik).all()
        # Ensure non-negative depths?
        _c_Ik[_c_Ik < _c_min] = _c_min
        _c_ik[_c_ik < _c_min] = _c_min
        _c_Ik[_c_Ik > _c_max] = _c_max
        _c_ik[_c_ik > _c_max] = _c_max
        # Export instance variables
        self._c_Ik = _c_Ik
        self._c_ik = _c_ik

    def link_concentrations_from_recurrence(self):
        # Import instance variables
        nk = self.nk
        _is_end = self._is_end
        _is_start = self._is_start
        _c_Ik = self._c_Ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _I_1k = self._I_1k
        _I_Np1k = self._I_Np1k
        _c_1k = _c_Ik[_I_1k]
        _c_Np1k = _c_Ik[_I_Np1k]
        # Compute internal flow estimates in both directions
        c_ik_b = c_i_b(_X_Ik[~_is_end], _c_Ik[~_is_end],
                       _Y_Ik[~_is_end], _Z_Ik[~_is_end],
                       np.repeat(_c_Np1k, nk))
        c_ik_f = c_i_f(_U_Ik[~_is_end], _c_Ik[~_is_start],
                       _V_Ik[~_is_end], _W_Ik[~_is_end],
                       np.repeat(_c_1k, nk))
        return c_ik_b, c_ik_f

    def step(self, dt=None, c_bc=None, c_0j=None, Q_0j=None, c_0Ik=None,
             Q_0Ik=None, u_j_frac=0.0):
        """
        Advance model forward to next time step, computing water quality states.

        Inputs:
        -------
        dt : float
            Time step to advance (s)
        c_bc : np.ndarray (M)
            Boundary concentration at each superjunction (*/m^3)
        c_0j : np.ndarray (M)
            Contaminant concentration in direction superjunction inflow `Q_in` (*/m^3).
            Defaults to 0.
        Q_0j : np.ndarray (M)
            Direct inflow at each superjunction (m^3/s).
            Defaults to `_Q_in` of underlying SuperLink model.
        c_0Ik : np.ndarray (M)
            Contaminant concentration in direction junction inflow `Q_0Ik` (*/m^3).
            Defaults to 0.
        Q_0Ik : np.ndarray (M)
            Direct inflow at each junction (m^3/s).
            Defaults to `_Q_0Ik` of underlying SuperLink model.
        u_j_frac : float
            (Deprecated).

        Returns:
        --------
        None
        """
        self.import_hydraulic_states()
        if dt is None:
            dt = self._dt
        if Q_0j is None:
            Q_0j = self.hydraulics._Q_in
        if Q_0Ik is None:
            Q_0Ik = self.hydraulics._Q_0Ik
        self.link_coeffs(_dt=dt, _u_j_frac=u_j_frac)
        self.node_coeffs(_Q_0Ik=Q_0Ik, _c_0Ik=c_0Ik, _dt=dt)
        self.forward_recurrence()
        self.backward_recurrence()
        self.boundary_coefficients()
        self.sparse_matrix_equations(c_bc=c_bc, _Q_0j=Q_0j,
                                     _c_0j=c_0j, _dt=dt)
        if self.hydraulics.banded:
            self.solve_banded_matrix()
        else:
            self.solve_sparse_matrix()
        self.solve_boundary_states()
        self.solve_internals_backwards()

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
    # Use upwind scheme
    t_0 = u_Ik / dx_ik
    t_1 = - 2 * D_ik / (dx_ik**2)
    return t_0 + t_1

@njit
def beta_ik(dt, D_ik, dx_ik, K_ik, u_Ik, u_Ip1k):
    t_0 = 1 / dt
    t_1 = 4 * D_ik / (dx_ik**2)
    t_2 = - K_ik
    t_3 = - u_Ik / dx_ik
    t_4 = - u_Ip1k / dx_ik
    return t_0 + t_1 + t_2 + t_3 + t_4

@njit
def chi_ik(u_Ip1k, dx_ik, D_ik):
    # Use upwind scheme
    t_0 = u_Ip1k / dx_ik
    t_1 = - 2 * D_ik / (dx_ik**2)
    return t_0 + t_1

@njit
def gamma_ik(dt, c_ik_prev):
    t_0 = c_ik_prev / dt
    return t_0

@njit
def kappa_Ik(Q_im1k_next, B_im1k, dx_im1k, h_Ik_next, K_im1k, A_im1k, dt):
    t_0 = - Q_im1k_next
    t_1 = dx_im1k * (B_im1k * h_Ik_next / dt + K_im1k * A_im1k) / 2
    return t_0 + t_1

@njit
def lambda_Ik(A_SIk, h_Ik_next, dt, K_Ik):
    t_0 = A_SIk * h_Ik_next * (1 / dt + K_Ik)
    return t_0

@njit
def mu_Ik(Q_ik_next, B_ik, dx_ik, h_Ik_next, K_ik, A_ik, dt):
    t_0 = Q_ik_next
    t_1 = dx_ik * (B_ik * h_Ik_next / dt + K_ik * A_ik) / 2
    return t_0 + t_1

@njit
def eta_Ik(c_0_Ik, Q_0_Ik, A_SIk, h_Ik_prev, c_Ik_prev, B_ik, B_im1k,
           dx_ik, dx_im1k, c_ik_prev, c_im1k_prev, dt):
    t_0 = c_0_Ik * Q_0_Ik
    t_1 = A_SIk * h_Ik_prev * c_Ik_prev / dt
    t_2 = B_ik * dx_ik * h_Ik_prev * c_ik_prev / 2 / dt
    t_3 = B_im1k * dx_im1k * h_Ik_prev * c_im1k_prev / 2 / dt
    return t_0 + t_1 + t_2 + t_3

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
def U_Ik(chi_ik, T_ik):
    return safe_divide(-chi_ik, T_ik)

@njit
def V_Ik(gamma_ik, T_ik, alpha_ik, eta_Ik, kappa_Ik, V_Im1k, lambda_Ik, U_Im1k):
    t_0 = safe_divide(gamma_ik, T_ik)
    t_1 = - alpha_ik * (eta_Ik - kappa_Ik * V_Im1k)
    # TODO: Note that this denominator is being computed 3 times
    t_2 = T_ik * (lambda_Ik + kappa_Ik * U_Im1k)
    return t_0 + safe_divide(t_1, t_2)

@njit
def W_Ik(alpha_ik, kappa_Ik, W_Im1k, T_ik, lambda_Ik, U_Im1k):
    t_0 = alpha_ik * kappa_Ik * W_Im1k
    t_1 = T_ik * (lambda_Ik + kappa_Ik * U_Im1k)
    return safe_divide(t_0, t_1)

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
def theta_uk(beta_uk, chi_uk):
    return safe_divide_vec(-beta_uk, chi_uk)

@njit
def sigma_uk(alpha_uk, chi_uk):
    return safe_divide_vec(-alpha_uk, chi_uk)

@njit
def xi_uk(gamma_uk, chi_uk):
    return safe_divide_vec(gamma_uk, chi_uk)

@njit
def theta_dk(beta_dk, alpha_dk):
    return safe_divide_vec(-beta_dk, alpha_dk)

@njit
def sigma_dk(chi_dk, alpha_dk):
    return safe_divide_vec(-chi_dk, alpha_dk)

@njit
def xi_dk(gamma_dk, alpha_dk):
    return safe_divide_vec(gamma_dk, alpha_dk)

@njit
def D_k_star(X_uk, U_dk, W_dk, Z_uk, theta_uk, theta_dk):
    """
    Compute superlink boundary condition coefficient 'D_k_star'.
    """
    t_0 = (1 - X_uk * theta_uk) * (1 - U_dk * theta_dk)
    t_1 = W_dk * Z_uk * theta_uk * theta_dk
    result = t_0 - t_1
    return result

@njit
def rho_uk(U_dk, X_uk, Z_uk, W_dk, theta_dk, sigma_uk, D_k_star):
    """
    Compute superlink boundary condition coefficient 'alpha' for upstream end
    of superlink k.
    """
    num = (1 - U_dk * theta_dk) * X_uk * sigma_uk + Z_uk * theta_dk * W_dk * sigma_uk
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def tau_uk(Z_uk, sigma_dk, D_k_star):
    """
    Compute superlink boundary condition coefficient 'beta' for upstream end
    of superlink k.
    """
    num = Z_uk * sigma_dk
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def omega_uk(Z_uk, V_dk, U_dk, Y_uk, X_uk, W_dk, theta_dk, xi_uk, xi_dk, D_k_star):
    """
    Compute superlink boundary condition coefficient 'chi' for upstream end
    of superlink k.
    """
    t_0 = (1 - U_dk * theta_dk) * (Y_uk + X_uk * xi_uk + Z_uk * xi_dk)
    t_1 = Z_uk * theta_dk * (V_dk + U_dk * xi_dk + W_dk * xi_uk)
    num = t_0 + t_1
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def rho_dk(W_dk, sigma_uk, D_k_star):
    """
    Compute superlink boundary condition coefficient 'alpha' for downstream end
    of superlink k.
    """
    num = W_dk * sigma_uk
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def tau_dk(X_uk, U_dk, W_dk, Z_uk, theta_uk, sigma_dk, D_k_star):
    """
    Compute superlink boundary condition coefficient 'beta' for downstream end
    of superlink k.
    """
    num = W_dk * theta_uk * Z_uk * sigma_dk + (1 - X_uk * theta_uk) * U_dk * sigma_dk
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def omega_dk(W_dk, Y_uk, X_uk, V_dk, Z_uk, U_dk, theta_uk, xi_uk, xi_dk, D_k_star):
    """
    Compute superlink boundary condition coefficient 'chi' for downstream end
    of superlink k.
    """
    t_0 = W_dk * theta_uk * (Y_uk + X_uk * xi_uk + Z_uk * xi_dk)
    t_1 = (1 - X_uk * theta_uk) * (V_dk + U_dk * xi_dk + W_dk * xi_uk)
    num = t_0 + t_1
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_node_coeffs(_kappa_Ik, _lambda_Ik, _mu_Ik, _eta_Ik,
                      _Q_ik_next, _h_Ik_next, _h_Ik_prev, _c_Ik_prev, _c_ik_prev,
                      _Q_uk_next, _Q_dk_next, _c_0Ik, _Q_0Ik, _A_SIk, _K_Ik, _K_ik,
                      _B_ik_next, _A_ik_next, _dx_ik_next, _forward_I_i, _backward_I_i,
                      _is_start, _is_end, _kI, _dt):
    N = _kI.size
    for I in range(N):
        if _is_start[I]:
            i = _forward_I_i[I]
            k = _kI[I]
            _kappa_Ik[I] = kappa_Ik(_Q_uk_next[k], 0.0, 0.0, _h_Ik_next[I], 0.0, 0.0, _dt)
            _lambda_Ik[I] = lambda_Ik(_A_SIk[I], _h_Ik_next[I], _dt, _K_Ik[I])
            _mu_Ik[I] = mu_Ik(_Q_ik_next[i], _B_ik_next[i], _dx_ik_next[i], _h_Ik_next[I],
                              _K_ik[i], _A_ik_next[i], _dt)
            _eta_Ik[I] = eta_Ik(_c_0Ik[I], _Q_0Ik[I], _A_SIk[I], _h_Ik_prev[I], _c_Ik_prev[I],
                                _B_ik_next[i], 0.0, _dx_ik_next[i], 0.0, _c_ik_prev[i], 0.0, _dt)
        elif _is_end[I]:
            im1 = _backward_I_i[I]
            k = _kI[I]
            _kappa_Ik[I] = kappa_Ik(_Q_ik_next[im1], _B_ik_next[im1], _dx_ik_next[im1], _h_Ik_next[I],
                                    _K_ik[im1], _A_ik_next[im1], _dt)
            _lambda_Ik[I] = lambda_Ik(_A_SIk[I], _h_Ik_next[I], _dt, _K_Ik[I])
            _mu_Ik[I] = mu_Ik(_Q_dk_next[k], 0.0, 0.0, _h_Ik_next[I], 0.0, 0.0, _dt)
            _eta_Ik[I] = eta_Ik(_c_0Ik[I], _Q_0Ik[I], _A_SIk[I], _h_Ik_prev[I],
                                _c_Ik_prev[I], 0.0, _B_ik_next[im1], 0.0, _dx_ik_next[im1],
                                0.0, _c_ik_prev[im1], _dt)
        else:
            i = _forward_I_i[I]
            im1 = i - 1
            _kappa_Ik[I] = kappa_Ik(_Q_ik_next[im1], _B_ik_next[im1], _dx_ik_next[im1], _h_Ik_next[I],
                                    _K_ik[im1], _A_ik_next[im1], _dt)
            _lambda_Ik[I] = lambda_Ik(_A_SIk[I], _h_Ik_next[I], _dt, _K_Ik[I])
            _mu_Ik[I] = mu_Ik(_Q_ik_next[i], _B_ik_next[i], _dx_ik_next[i], _h_Ik_next[I], _K_ik[i],
                              _A_ik_next[i], _dt)
            _eta_Ik[I] = eta_Ik(_c_0Ik[I], _Q_0Ik[I], _A_SIk[I], _h_Ik_prev[I], _c_Ik_prev[I],
                                _B_ik_next[i], _B_ik_next[im1], _dx_ik_next[i], _dx_ik_next[im1],
                                _c_ik_prev[i], _c_ik_prev[im1], _dt)
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
            _U_Ik[_I_next] = U_Ik(_chi_ik[_i_next], _T_ik[_i_next])
            _V_Ik[_I_next] = V_Ik(_gamma_ik[_i_next], _T_ik[_i_next], _alpha_ik[_i_next],
                                  _eta_Ik[_I_next], _kappa_Ik[_I_next], _V_Ik[_Im1_next],
                                  _lambda_Ik[_I_next], _U_Ik[_Im1_next])
            _W_Ik[_I_next] = W_Ik(_alpha_ik[_i_next], _kappa_Ik[_I_next], _W_Ik[_Im1_next],
                                  _T_ik[_i_next], _lambda_Ik[_I_next], _U_Ik[_Im1_next])
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
                                _X_Ik, _Y_Ik, _Z_Ik, _U_Ik, _V_Ik, _W_Ik,
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

@njit(fastmath=True)
def numba_add_at(a, indices, b):
    n = len(indices)
    for k in range(n):
        i = indices[k]
        a[i] += b[k]

@njit
def numba_clear_off_diagonals(A, bc, _J_uk, _J_dk, NK):
    for k in range(NK):
        _J_u = _J_uk[k]
        _J_d = _J_dk[k]
        _bc_u = bc[_J_u]
        _bc_d = bc[_J_d]
        if not _bc_u:
            A[_J_u, _J_d] = 0.0
        if not _bc_d:
            A[_J_d, _J_u] = 0.0

@njit(fastmath=True)
def numba_create_A_matrix(A, _F_jj, bc, _J_uk, _J_dk, _rho_uk, _rho_dk, _tau_uk, _tau_dk,
                          _Q_uk, _Q_dk, _A_sj, _V_sj, _H_j_next, _dt, _K_j, M, NK):
    numba_add_at(_F_jj, _J_uk, _rho_uk * _Q_uk)
    numba_add_at(_F_jj, _J_dk, -_tau_dk * _Q_dk)
    _F_jj += (_A_sj * _H_j_next / _dt) + (_K_j * _V_sj)
    # Set diagonal of A matrix
    for i in range(M):
        if bc[i]:
            A[i,i] = 1.0
        else:
            A[i,i] = _F_jj[i]
    for k in range(NK):
        _J_u = _J_uk[k]
        _J_d = _J_dk[k]
        _bc_u = bc[_J_u]
        _bc_d = bc[_J_d]
        if not _bc_u:
            A[_J_u, _J_d] += (_tau_uk[k] * _Q_uk[k])
        if not _bc_d:
            A[_J_d, _J_u] -= (_rho_dk[k] * _Q_dk[k])

@njit(fastmath=True)
def numba_create_OWP_matrix(X, diag, bc, _J_uc, _J_dc, _omega_c, _Q_c, M, NC):
    # Set diagonal
    numba_add_at(diag, _J_uc, _Q_c * _omega_c)
    numba_add_at(diag, _J_dc, -_Q_c * (1 - _omega_c))
    for i in range(M):
        if bc[i]:
            X[i,i] = 0.0
        else:
            X[i,i] = diag[i]
    # Set off-diagonal
    for c in range(NC):
        _J_u = _J_uc[c]
        _J_d = _J_dc[c]
        _bc_u = bc[_J_u]
        _bc_d = bc[_J_d]
        if not _bc_u:
            X[_J_u, _J_d] += (_Q_c[c] * (1 - _omega_uc[c]))
        if not _bc_d:
            X[_J_d, _J_u] -= (_Q_c[c] * _omega_c[c])

@njit
def numba_create_banded(l, bandwidth, M):
    AB = np.zeros((2*bandwidth + 1, M))
    for i in range(M):
        AB[bandwidth, i] = l[i, i]
    for n in range(bandwidth):
        for j in range(M - n - 1):
            AB[bandwidth - n - 1, -j - 1] = l[-j - 2 - n, -j - 1]
            AB[bandwidth + n + 1, j] = l[j + n + 1, j]
    return AB

@njit
def numba_solve_internals(_c_Ik, _c_ik, _c_1k, _c_Np1k, _U_Ik, _V_Ik, _W_Ik,
                          _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, nk, NK,
                          min_c, max_c, first_link_backwards=True):
    for k in range(NK):
        n = nk[k]
        i_1 = _i_1k[k]
        I_1 = _I_1k[k]
        i_n = i_1 + n - 1
        I_Np1 = I_1 + n
        I_N = I_Np1 - 1
        # Set boundary depths
        _c_1 = _c_1k[k]
        _c_Np1 = _c_Np1k[k]
        _c_Ik[I_1] = _c_1
        _c_Ik[I_Np1] = _c_Np1
        # Set max depth
        # max_depth = max_depth_k[k]
        # Compute internal depths and flows (except first link flow)
        for j in range(n - 1):
            I = I_N - j
            Ip1 = I + 1
            i = i_n - j
            _c_ik[i] = c_i_f(_c_Ik[Ip1], _c_1, _U_Ik[I], _V_Ik[I], _W_Ik[I])
            _c_Ik[I] = c_I_b(_c_ik[i], _c_Np1, _X_Ik[I], _Y_Ik[I], _Z_Ik[I])
            if _c_Ik[I] < min_c:
                _c_Ik[I] = min_c
            if _c_Ik[I] > max_c:
                _c_Ik[I] = max_c
            if _c_ik[i] < min_c:
                _c_ik[i] = min_c
            if _c_ik[i] > max_c:
                _c_ik[i] = max_c
        if first_link_backwards:
            _c_ik[i_1] = c_i_b(_c_Ik[I_1], _c_Np1, _X_Ik[I_1], _Y_Ik[I_1],
                            _Z_Ik[I_1])
        else:
            # Not theoretically correct, but seems to be more stable sometimes
            _c_ik[i_1] = c_i_f(_c_Ik[I_1 + 1], _c_1, _U_Ik[I_1], _V_Ik[I_1],
                            _W_Ik[I_1])
    return 1

@njit
def c_i_f(c_Ip1k, c_1k, U_Ik, V_Ik, W_Ik):
    t_0 = U_Ik * c_Ip1k
    t_1 = V_Ik
    t_2 = W_Ik * c_1k
    return t_0 + t_1 + t_2

@njit
def c_i_b(c_Ik, c_Np1k, X_Ik, Y_Ik, Z_Ik):
    t_0 = X_Ik * c_Ik
    t_1 = Y_Ik
    t_2 = Z_Ik * c_Np1k
    return t_0 + t_1 + t_2

@njit
def c_I_b(c_ik, c_Np1k, X_Ik, Y_Ik, Z_Ik):
    num = c_ik - Y_Ik - Z_Ik * c_Np1k
    den = X_Ik
    result = safe_divide(num, den)
    return result

