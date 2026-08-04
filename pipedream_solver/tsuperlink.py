import numpy as np

from pipedream_solver.nsuperlink import nSuperLink
from pipedream_solver._tsuperlink import tnumba_a_ik, tnumba_b_ik, tnumba_c_ik, tnumba_P_ik, tnumba_forward_recurrence, tnumba_backward_recurrence, tnumba_solve_internals

class tSuperLink(nSuperLink):
    def __init__(self, superlinks, superjunctions,
                 links=None, junctions=None,
                 transects={}, storages={},
                 orifices=None, weirs=None, pumps=None,
                 dt=60, sparse=False, min_depth=1e-5, method='b',
                 inertial_damping=False, bc_method='z',
                 exit_hydraulics=False, auto_permute=False,
                 end_length=None, end_method='b', internal_links=4, mobile_elements=False):
        super().__init__(superlinks, superjunctions,
                         links, junctions, transects, storages,
                         orifices, weirs, pumps, dt, sparse,
                         min_depth, method, inertial_damping,
                         bc_method, exit_hydraulics, auto_permute,
                         end_length, end_method, internal_links, mobile_elements)

    def link_coeffs(self, _dt=None, first_iter=True):
        """
        Compute link momentum coefficients: a_ik, b_ik, c_ik and P_ik.
        """
        # Import instance variables
        _u_ik = self._u_ik
        _u_uk = self._u_uk
        _u_dk = self._u_dk
        _dx_ik = self._dx_ik       # Length of link ik
        _Sf_method_ik = self._Sf_method_ik
        _n_ik = self._n_ik         # Manning's roughness of link ik
        _Q_ik_prev = np.copy(self.states['Q_ik'])
        _h_Ik_prev = np.copy(self.states['h_Ik'][self._Ik])
        _h_Ip1k_prev = np.copy(self.states['h_Ik'][self._Ip1k])
        _Q_ik_next = self._Q_ik         # Flow rate at link ik
        _A_ik = self._A_ik         # Flow area at link ik
        _B_ik = self._B_ik
        _R_ik = self._R_ik         # Hydraulic radius at link ik
        _S_o_ik = self._S_o_ik     # Channel bottom slope at link ik
        _C_ik = self._C_ik         # Discharge coefficient of control structure at link ik
        # Upstream parameters
        _n_uk = self._n_uk
        _Q_uk_next = self._Q_uk
        _Q_uk_prev = self.states['Q_uk']
        _A_uk = self._A_uk
        _B_uk = self._B_uk
        _R_uk = self._R_uk
        _dx_uk = self._dx_uk
        _Sf_method_uk = self._Sf_method_uk
        _C_uk = self._C_uk
        _S_o_uk = self._S_o_uk
        _theta_uk = self._theta_uk
        _z_inv_uk = self._z_inv_uk
        _J_uk = self._J_uk
        _I_1k = self._I_1k
        _h_juk_prev = _theta_uk * (self.states['H_j'][_J_uk] - _z_inv_uk)
        _h_1k_prev = np.copy(self.states['h_Ik'][_I_1k])
        # Downstream parameters
        _n_dk = self._n_dk
        _Q_dk_next = self._Q_dk
        _Q_dk_prev = self.states['Q_dk']
        _A_dk = self._A_dk
        _B_dk = self._B_dk
        _R_dk = self._R_dk
        _dx_dk = self._dx_dk
        _Sf_method_dk = self._Sf_method_dk
        _C_dk = self._C_dk
        _S_o_dk = self._S_o_dk
        _theta_dk = self._theta_dk
        _z_inv_dk = self._z_inv_dk
        _J_dk = self._J_dk
        _I_Np1k = self._I_Np1k
        _h_jdk_prev = _theta_dk * (self.states['H_j'][_J_dk] - _z_inv_dk)
        _h_Np1k_prev = np.copy(self.states['h_Ik'][_I_Np1k])
        # If time step not specified, use instance time
        if _dt is None:
            _dt = self._dt
        g = 9.81
        # Compute link coefficients
        _a_ik = tnumba_a_ik(_u_ik, _B_ik, _A_ik, _dx_ik, _dt, g)
        _c_ik = tnumba_c_ik(_u_ik, _B_ik, _A_ik, _dx_ik, _dt, g)
        _b_ik = tnumba_b_ik(_dx_ik, _dt, _n_ik, _Q_ik_next, _A_ik, _R_ik, _C_ik, _Sf_method_ik, g)
        _P_ik = tnumba_P_ik(_Q_ik_prev, _dx_ik, _dt, _u_ik, _B_ik, _A_ik, _h_Ik_prev, _h_Ip1k_prev, _S_o_ik, g)
        # Compute momentum coefficients for upstream boundary
        _a_uk = _theta_uk * tnumba_a_ik(_u_uk, _B_uk, _A_uk, _dx_uk, _dt, g)
        _c_uk = tnumba_c_ik(_u_uk, _B_uk, _A_uk, _dx_uk, _dt, g)
        _b_uk = tnumba_b_ik(_dx_uk, _dt, _n_uk, _Q_uk_next, _A_uk, _R_uk, _C_uk, _Sf_method_uk, g)
        _P_uk = tnumba_P_ik(_Q_uk_prev, _dx_uk, _dt, _u_uk, _B_uk, _A_uk, _h_juk_prev, _h_1k_prev, _S_o_uk, g)
        _P_uk += _theta_uk * _z_inv_uk * _a_uk 
        # Compute momentum coefficients for downstream boundary
        _a_dk = tnumba_a_ik(_u_dk, _B_dk, _A_dk, _dx_dk, _dt, g)
        _c_dk = _theta_dk * tnumba_c_ik(_u_dk, _B_dk, _A_dk, _dx_dk, _dt, g)
        _b_dk = tnumba_b_ik(_dx_dk, _dt, _n_dk, _Q_dk_next, _A_dk, _R_dk, _C_dk, _Sf_method_dk, g)
        _P_dk = tnumba_P_ik(_Q_dk_prev, _dx_dk, _dt, _u_dk, _B_dk, _A_dk, _h_Np1k_prev, _h_jdk_prev, _S_o_dk, g)
        _P_dk += _theta_dk * _z_inv_dk * _c_dk 
        # Export to instance variables
        self._a_ik = _a_ik
        self._b_ik = _b_ik
        self._c_ik = _c_ik
        self._P_ik = _P_ik
        self._a_uk = _a_uk
        self._b_uk = _b_uk
        self._c_uk = _c_uk
        self._P_uk = _P_uk
        self._a_dk = _a_dk
        self._b_dk = _b_dk
        self._c_dk = _c_dk
        self._P_dk = _P_dk

    def forward_recurrence(self):
        """
        Compute forward recurrence coefficients: T_ik, U_Ik, V_Ik, and W_Ik.
        """
        # Import instance variables
        _I_1k = self._I_1k                # Index of first junction in each superlink
        _i_1k = self._i_1k                # Index of first link in each superlink
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
        tnumba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _a_ik, _b_ik, _c_ik,
                                 _P_ik, _E_Ik, _D_Ik, NK, nk, _I_1k, _i_1k)
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
        tnumba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _a_ik, _b_ik, _c_ik,
                                    _P_ik, _E_Ik, _D_Ik, NK, nk, _I_Nk, _i_nk)
        # Export instance variables
        self._O_ik = _O_ik
        self._X_Ik = _X_Ik
        self._Y_Ik = _Y_Ik
        self._Z_Ik = _Z_Ik

    def superlink_boundary_flow_coefficients(self, _dt=None):
        _U_Ik = self._U_Ik                # Recurrence coefficient U_Ik
        _V_Ik = self._V_Ik                # Recurrence coefficient V_Ik
        _W_Ik = self._W_Ik                # Recurrence coefficient W_Ik
        _X_Ik = self._X_Ik                # Recurrence coefficient X_Ik
        _Y_Ik = self._Y_Ik                # Recurrence coefficient Y_Ik
        _Z_Ik = self._Z_Ik                # Recurrence coefficient Z_Ik
        _E_Ik = self._E_Ik
        _D_Ik = self._D_Ik
        _I_1k = self._I_1k
        _I_Nk = self._I_Nk
        _I_Np1k = self._I_Np1k
        a_uk = self._a_uk
        a_dk = self._a_dk
        b_uk = self._b_uk
        b_dk = self._b_dk
        c_uk = self._c_uk
        c_dk = self._c_dk
        P_uk = self._P_uk
        P_dk = self._P_dk
        # Get boundary coefficients
        U_Nk = _U_Ik[_I_Nk]
        Z_1k = _Z_Ik[_I_1k]
        W_Nk = _W_Ik[_I_Nk]
        X_1k = _X_Ik[_I_1k]
        E_1k = _E_Ik[_I_1k]
        E_Np1k = _E_Ik[_I_Np1k]
        D_1k = _D_Ik[_I_1k]
        D_Np1k = _D_Ik[_I_Np1k]
        Y_1k = _Y_Ik[_I_1k]
        V_Nk = _V_Ik[_I_Nk]
        # Formulate expressions
        a = U_Nk - E_Np1k
        b = Z_1k
        c = W_Nk
        d = X_1k + E_1k
        e = Y_1k - D_1k
        f = D_Np1k + V_Nk
        p = a*d - b*c
        q = -p
        r = (a_dk * d - b_dk * q)
        s = (b_uk * q - c_uk * a)
        # Create inverse matrix
        aa = -a_uk * q * r
        bb = -(b * c_uk * c_dk * q)
        cc = -(c * a_uk * a_dk * q)
        dd = c_dk * q * s
        ee = P_uk * (b_dk * q - a_dk * d) + c_uk * (b_dk * (b * f - a * e) - a_dk * e - P_dk * b)
        ff = P_dk * (b_uk * q - c_uk * a) + a_dk * (b_uk * (c * e -d * f) - c_uk * f - P_uk * c)
        denom =  (c * b * a_dk * c_uk) + (r * s)
        exog_denom = b_dk * b_uk * q - a_dk * b_uk * d - c_uk * b_dk * a - c_uk * a_dk
        # Compute coefficients
        alpha_uk = aa / denom
        beta_uk = bb / denom
        chi_uk = ee / exog_denom
        alpha_dk = cc / denom
        beta_dk = dd / denom
        chi_dk = ff / exog_denom
        # Store coefficients
        self._alpha_uk = alpha_uk
        self._beta_uk = beta_uk
        self._chi_uk = chi_uk
        self._alpha_dk = alpha_dk
        self._beta_dk = beta_dk
        self._chi_dk = chi_dk

    def solve_internals_backwards(self, subcritical_only=False):
        """
        Solve for internal states of each superlink in the backward direction.
        """
        # Import instance variables
        _I_1k = self._I_1k                  # Index of first junction in superlink k
        _I_Nk = self._I_Nk
        _i_1k = self._i_1k                  # Index of first link in superlink k
        nk = self.nk
        NK = self.NK
        _h_Ik = self._h_Ik                  # Depth at junction Ik
        _Q_ik = self._Q_ik                  # Flow rate at link ik
        _D_Ik = self._D_Ik                  # Continuity coefficient
        _E_Ik = self._E_Ik                  # Continuity coefficient
        _U_Ik = self._U_Ik                  # Forward recurrence coefficient
        _V_Ik = self._V_Ik                  # Forward recurrence coefficient
        _W_Ik = self._W_Ik                  # Forward recurrence coefficient
        _X_Ik = self._X_Ik                  # Backward recurrence coefficient
        _Y_Ik = self._Y_Ik                  # Backward recurrence coefficient
        _Z_Ik = self._Z_Ik                  # Backward recurrence coefficient
        _Q_uk = self._Q_uk                  # Flow rate at upstream end of superlink k
        _Q_dk = self._Q_dk                  # Flow rate at downstream end of superlink k
        _h_uk = self._h_uk                  # Depth at upstream end of superlink k
        _h_dk = self._h_dk                  # Depth at downstream end of superlink k
        min_depth = self.min_depth          # Minimum allowable water depth
        max_depth_k = self.max_depth_k
        # Solve internals
        tnumba_solve_internals(_h_Ik, _Q_ik, _h_uk, _h_dk, _U_Ik, _V_Ik, _W_Ik,
                              _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, _I_Nk, nk, NK)
        # TODO: Temporary
        assert np.isfinite(_h_Ik).all()
        # Export instance variables
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    def solve_internals_forwards(self, subcritical_only=False):
        """
        Solve for internal states of each superlink in the backward direction.
        """
        # Import instance variables
        _I_1k = self._I_1k                  # Index of first junction in superlink k
        _i_1k = self._i_1k                  # Index of first link in superlink k
        _I_Nk = self._I_Nk
        nk = self.nk
        NK = self.NK
        _h_Ik = self._h_Ik                  # Depth at junction Ik
        _Q_ik = self._Q_ik                  # Flow rate at link ik
        _D_Ik = self._D_Ik                  # Continuity coefficient
        _E_Ik = self._E_Ik                  # Continuity coefficient
        _U_Ik = self._U_Ik                  # Forward recurrence coefficient
        _V_Ik = self._V_Ik                  # Forward recurrence coefficient
        _W_Ik = self._W_Ik                  # Forward recurrence coefficient
        _X_Ik = self._X_Ik                  # Backward recurrence coefficient
        _Y_Ik = self._Y_Ik                  # Backward recurrence coefficient
        _Z_Ik = self._Z_Ik                  # Backward recurrence coefficient
        _Q_uk = self._Q_uk                  # Flow rate at upstream end of superlink k
        _Q_dk = self._Q_dk                  # Flow rate at downstream end of superlink k
        _h_uk = self._h_uk                  # Depth at upstream end of superlink k
        _h_dk = self._h_dk                  # Depth at downstream end of superlink k
        min_depth = self.min_depth          # Minimum allowable water depth
        max_depth_k = self.max_depth_k
        # Solve internals
        tnumba_solve_internals(_h_Ik, _Q_ik, _h_uk, _h_dk, _U_Ik, _V_Ik, _W_Ik,
                              _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, _I_Nk, nk, NK)
        # Export instance variables
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik