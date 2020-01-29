import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
from numba import njit, prange
import superlink.geometry
import superlink.ngeometry
import superlink.storage
from superlink.superlink import SuperLink

class NumbaLink(SuperLink):
    def __init__(self, superlinks, superjunctions,
                 links=None, junctions=None,
                 transects={}, storages={},
                 orifices=None, weirs=None, pumps=None,
                 dt=60, sparse=False, min_depth=1e-5, method='b',
                 inertial_damping=False, bc_method='z',
                 exit_hydraulics=False, end_length=None,
                 end_method='o'):
        super().__init__(superlinks, superjunctions,
                         links, junctions, transects, storages,
                         orifices, weirs, pumps, dt, sparse,
                         min_depth, method, inertial_damping,
                         bc_method, exit_hydraulics, end_length,
                         end_method)

    def link_hydraulic_geometry(self):
        """
        Compute hydraulic geometry for each link.
        """
        # Import instance variables
        _ik = self._ik                 # Link index
        _Ik = self._Ik                 # Junction index
        _Ip1k = self._Ip1k             # Index of next junction
        _h_Ik = self._h_Ik             # Depth at junction Ik
        _A_ik = self._A_ik             # Flow area at link ik
        _Pe_ik = self._Pe_ik           # Hydraulic perimeter at link ik
        _R_ik = self._R_ik             # Hydraulic radius at link ik
        _B_ik = self._B_ik             # Top width at link ik
        _dx_ik = self._dx_ik           # Length of link ik
        _g1_ik = self._g1_ik           # Geometry 1 of link ik (vertical)
        _g2_ik = self._g2_ik           # Geometry 2 of link ik (horizontal)
        _g3_ik = self._g3_ik           # Geometry 3 of link ik (other)
        _geom_codes = self._geom_codes
        _transect_factory = self._transect_factory
        _transect_indices = self._transect_indices
        _has_irregular = self._has_irregular
        # Compute hydraulic geometry for regular geometries
        numba_hydraulic_geometry(_A_ik, _Pe_ik, _R_ik, _B_ik, _h_Ik,
                                 _g1_ik, _g2_ik, _g3_ik, _geom_codes, _Ik, _ik)
        # Compute hydraulic geometry for irregular geometries
        if _has_irregular:
            for transect_name, generator in _transect_factory.items():
                _ik_g = _transect_indices.loc[[transect_name]].values
                _Ik_g = _Ik[_ik_g]
                _Ip1k_g = _Ip1k[_ik_g]
                _h_Ik_g = _h_Ik[_Ik_g]
                _h_Ip1k_g = _h_Ik[_Ip1k_g]
                _A_ik[_ik_g] = generator.A_ik(_h_Ik_g, _h_Ip1k_g)
                _Pe_ik[_ik_g] = generator.Pe_ik(_h_Ik_g, _h_Ip1k_g)
                _R_ik[_ik_g] = generator.R_ik(_h_Ik_g, _h_Ip1k_g)
                _B_ik[_ik_g] = generator.B_ik(_h_Ik_g, _h_Ip1k_g)
        # Export to instance variables
        self._A_ik = _A_ik
        self._Pe_ik = _Pe_ik
        self._R_ik = _R_ik
        self._B_ik = _B_ik

    def upstream_hydraulic_geometry(self, area='avg'):
        """
        Compute hydraulic geometry of upstream ends of superlinks.
        """
        # Import instance variables
        _ik = self._ik                 # Link index
        _Ik = self._Ik                 # Junction index
        _ki = self._ki                 # Superlink index containing link ik
        _h_Ik = self._h_Ik             # Depth at junction Ik
        _A_uk = self._A_uk             # Flow area at upstream end of superlink k
        _B_uk = self._B_uk             # Top width at upstream end of superlink k
        _dx_ik = self._dx_ik           # Length of link ik
        _g1_ik = self._g1_ik           # Geometry 1 of link ik (vertical)
        _g2_ik = self._g2_ik           # Geometry 2 of link ik (horizontal)
        _g3_ik = self._g3_ik           # Geometry 3 of link ik (other)
        _z_inv_uk = self._z_inv_uk     # Invert offset of upstream end of superlink k
        _J_uk = self._J_uk             # Index of junction upstream of superlink k
        H_j = self.H_j                 # Head at superjunction j
        _transect_factory = self._transect_factory
        _uk_transect_indices = self._uk_transect_indices
        _uk_has_irregular = self._uk_has_irregular
        _i_1k = self._i_1k
        _I_1k = self._I_1k
        _geom_codes = self._geom_codes
        # Compute hydraulic geometry for regular geometries
        numba_boundary_geometry(_A_uk, _B_uk, _h_Ik, H_j, _z_inv_uk,
                                _g1_ik, _g2_ik, _g2_ik, _geom_codes,
                                _i_1k, _I_1k, _J_uk)
        # Compute hydraulic geometry for irregular geometries
        if _uk_has_irregular:
            for transect_name, generator in _transect_factory.items():
                _ik_g = _uk_transect_indices.loc[[transect_name]].values
                _ki_g = _ki[_ik_g]
                _Ik_g = _Ik[_ik_g]
                _h_Ik_g = _h_Ik[_Ik_g]
                _H_j_g = H_j[_J_uk[_ki_g]] - _z_inv_uk[_ki_g]
                # TODO: Allow for max here like above
                _A_uk[_ki_g] = generator.A_ik(_h_Ik_g, _H_j_g)
                _B_uk[_ki_g] = generator.B_ik(_h_Ik_g, _H_j_g)
        # Export to instance variables
        self._A_uk = _A_uk

    def downstream_hydraulic_geometry(self, area='avg'):
        """
        Compute hydraulic geometry of downstream ends of superlinks.
        """
        # Import instance variables
        _ik = self._ik                 # Link index
        _Ip1k = self._Ip1k             # Next junction index
        _ki = self._ki                 # Superlink index containing link ik
        _h_Ik = self._h_Ik             # Depth at junction Ik
        _A_dk = self._A_dk             # Flow area at downstream end of superlink k
        _B_dk = self._B_dk             # Top width at downstream end of superlink k
        _dx_ik = self._dx_ik           # Length of link ik
        _g1_ik = self._g1_ik           # Geometry 1 of link ik (vertical)
        _g2_ik = self._g2_ik           # Geometry 2 of link ik (horizontal)
        _g3_ik = self._g3_ik           # Geometry 3 of link ik (other)
        _z_inv_dk = self._z_inv_dk     # Invert offset of downstream end of superlink k
        _J_dk = self._J_dk             # Index of junction downstream of superlink k
        H_j = self.H_j                 # Head at superjunction j
        _transect_factory = self._transect_factory
        _dk_transect_indices = self._dk_transect_indices
        _dk_has_irregular = self._dk_has_irregular
        _i_nk = self._i_nk
        _I_Np1k = self._I_Np1k
        _geom_codes = self._geom_codes
        # Compute hydraulic geometry for regular geometries
        numba_boundary_geometry(_A_dk, _B_dk, _h_Ik, H_j, _z_inv_dk,
                                _g1_ik, _g2_ik, _g2_ik, _geom_codes,
                                _i_nk, _I_Np1k, _J_dk)
        # Compute hydraulic geometry for irregular geometries
        if _dk_has_irregular:
            for transect_name, generator in _transect_factory.items():
                _ik_g = _dk_transect_indices.loc[[transect_name]].values
                _ki_g = _ki[_ik_g]
                _Ip1k_g = _Ip1k[_ik_g]
                _h_Ip1k_g = _h_Ik[_Ip1k_g]
                _H_j_g = H_j[_J_dk[_ki_g]] - _z_inv_dk[_ki_g]
                # TODO: Allow max here like above
                _A_dk[_ki_g] = generator.A_ik(_h_Ip1k_g, _H_j_g)
                _B_dk[_ki_g] = generator.B_ik(_h_Ip1k_g, _H_j_g)
        # Export to instance variables
        self._A_dk = _A_dk

    def node_velocities(self):
        """
        Compute velocity of flow at each link and junction.
        """
        # Import instance variables
        _Ip1k = self._Ip1k                   # Next junction index
        _A_ik = self._A_ik                   # Flow area at link ik
        _Q_ik = self._Q_ik                   # Flow rate at link ik
        _u_ik = self._u_ik
        _u_Ik = self._u_Ik                   # Flow velocity at junction Ik
        _u_Ip1k = self._u_Ip1k               # Flow velocity at junction I + 1k
        _dx_ik = self._dx_ik                 # Length of link ik
        _link_start = self._link_start
        _link_end = self._link_end
        # Determine start and end nodes
        # Compute link velocities
        numba_u_ik(_Q_ik, _A_ik, _u_ik)
        # Compute velocities for start nodes (1 -> Nk)
        numba_u_Ik(_dx_ik, _u_ik, _link_start, _u_Ik)
        # Compute velocities for end nodes (2 -> Nk+1)
        numba_u_Ip1k(_dx_ik, _u_ik, _link_end, _u_Ip1k)
        # Export to instance variables
        self._u_ik = _u_ik
        self._u_Ik = _u_Ik
        self._u_Ip1k = _u_Ip1k

    def forward_recurrence(self):
        """
        Compute forward recurrence coefficients: T_ik, U_Ik, V_Ik, and W_Ik.
        """
        # Import instance variables
        _I_1k = self._I_1k                # Index of first junction in each superlink
        _i_1k = self._i_1k                # Index of first link in each superlink
        _A_ik = self._A_ik                # Flow area in link ik
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
        numba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _a_ik, _b_ik, _c_ik,
                                 _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_1k, _i_1k)
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
        _A_ik = self._A_ik                # Flow area in link ik
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
        numba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _a_ik, _b_ik, _c_ik,
                                    _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_Nk, _i_nk)
        # Export instance variables
        self._O_ik = _O_ik
        self._X_Ik = _X_Ik
        self._Y_Ik = _Y_Ik
        self._Z_Ik = _Z_Ik

    def superlink_flow_coefficients(self):
        """
        Compute superlink flow coefficients: alpha_uk, beta_uk, chi_uk,
        alpha_dk, beta_dk, chi_dk.
        """
        # Import instance variables
        _I_1k = self._I_1k              # Index of first junction in superlink k
        _I_Nk = self._I_Nk              # Index of penultimate junction in superlink k
        _I_Np1k = self._I_Np1k          # Index of last junction in superlink k
        _D_Ik = self._D_Ik              # Continuity coefficient
        _E_Ik = self._E_Ik              # Continuity coefficient
        _X_Ik = self._X_Ik              # Backward recurrence coefficient X_Ik
        _Y_Ik = self._Y_Ik              # Backward recurrence coefficient Y_Ik
        _Z_Ik = self._Z_Ik              # Backward recurrence coefficient Z_Ik
        _U_Ik = self._U_Ik              # Forward recurrence coefficient U_Ik
        _V_Ik = self._V_Ik              # Forward recurrence coefficient V_Ik
        _W_Ik = self._W_Ik              # Forward recurrence coefficient W_Ik
        _kappa_uk = self._kappa_uk      # Upstream superlink head coefficient kappa_uk
        _kappa_dk = self._kappa_dk      # Downstream superlink head coefficient kappa_dk
        _lambda_uk = self._lambda_uk    # Upstream superlink head coefficient lambda_uk
        _lambda_dk = self._lambda_dk    # Downstream superlink head coefficient lambda_dk
        _mu_uk = self._mu_uk            # Upstream superlink head coefficient mu_uk
        _mu_dk = self._mu_dk            # Downstream superlink head coefficient mu_dk
        _J_uk = self._J_uk              # Superjunction upstream of superlink k
        _J_dk = self._J_dk              # Superjunction downstream of superlink k
        H_j = self.H_j                  # Head at superjunction j
        _z_inv_uk = self._z_inv_uk      # Invert offset of upstream end of superlink k
        _z_inv_dk = self._z_inv_dk      # Invert offset of downstream end of superlink k
        _z_inv_j = self._z_inv_j        # Invert elevation at superjunction j
        _end_method = self._end_method    # Method for computing flow at pipe ends
        _theta_uk = self._theta_uk      # Upstream indicator variable
        _theta_dk = self._theta_dk      # Downstream indicator variable
        if _end_method == 'o':
            _X_1k = _X_Ik[_I_1k]
            _Y_1k = _Y_Ik[_I_1k]
            _Z_1k = _Z_Ik[_I_1k]
            _U_Nk = _U_Ik[_I_Nk]
            _V_Nk = _V_Ik[_I_Nk]
            _W_Nk = _W_Ik[_I_Nk]
        else:
            _X_1k = _X_Ik[_I_1k] + _E_Ik[_I_1k]
            _Y_1k = _Y_Ik[_I_1k] - _D_Ik[_I_1k]
            _Z_1k = _Z_Ik[_I_1k]
            _U_Nk = _U_Ik[_I_Nk] - _E_Ik[_I_Np1k]
            _V_Nk = _V_Ik[_I_Nk] + _D_Ik[_I_Np1k]
            _W_Nk = _W_Ik[_I_Nk]
        # Compute theta indicator variables
        _H_juk = H_j[_J_uk]
        _H_jdk = H_j[_J_dk]
        _theta_uk = np.where(_H_juk >= _z_inv_uk, 1.0, 0.0)
        _theta_dk = np.where(_H_jdk >= _z_inv_dk, 1.0, 0.0)
        # _theta_uk = 1.
        # _theta_dk = 1.
        # Compute D_k_star
        _D_k_star = numba_D_k_star(_X_1k, _kappa_uk, _U_Nk,
                                   _kappa_dk, _Z_1k, _W_Nk)
        # Compute upstream superlink flow coefficients
        _alpha_uk = numba_alpha_uk(_U_Nk, _kappa_dk, _X_1k,
                                   _Z_1k, _W_Nk, _D_k_star,
                                   _lambda_uk, _theta_uk)
        _beta_uk = numba_beta_uk(_U_Nk, _kappa_dk, _Z_1k,
                                 _W_Nk, _D_k_star, _lambda_dk, _theta_dk)
        _chi_uk = numba_chi_uk(_U_Nk, _kappa_dk, _Y_1k,
                               _X_1k, _mu_uk, _Z_1k,
                               _mu_dk, _V_Nk, _W_Nk,
                               _D_k_star, _theta_uk, _theta_dk)
        # Compute downstream superlink flow coefficients
        _alpha_dk = numba_alpha_dk(_X_1k, _kappa_uk, _W_Nk,
                                   _D_k_star, _lambda_uk, _theta_uk)
        _beta_dk = numba_beta_dk(_X_1k, _kappa_uk, _U_Nk,
                                 _W_Nk, _Z_1k, _D_k_star,
                                 _lambda_dk, _theta_dk)
        _chi_dk = numba_chi_dk(_X_1k, _kappa_uk, _V_Nk,
                               _W_Nk, _mu_uk, _U_Nk,
                               _mu_dk, _Y_1k, _Z_1k,
                               _D_k_star, _theta_uk, _theta_dk)
        # Export instance variables
        self._D_k_star = _D_k_star
        self._alpha_uk = _alpha_uk
        self._beta_uk = _beta_uk
        self._chi_uk = _chi_uk
        self._alpha_dk = _alpha_dk
        self._beta_dk = _beta_dk
        self._chi_dk = _chi_dk
        self._theta_uk = _theta_uk
        self._theta_dk = _theta_dk

    def sparse_matrix_equations(self, H_bc=None, _Q_0j=None, u=None, _dt=None, implicit=True,
                                first_time=False):
        """
        Construct sparse matrices A, O, W, P and b.
        """
        # Import instance variables
        _k = self._k                     # Superlink indices
        _J_uk = self._J_uk               # Index of superjunction upstream of superlink k
        _J_dk = self._J_dk               # Index of superjunction downstream of superlink k
        _alpha_uk = self._alpha_uk       # Superlink flow coefficient
        _alpha_dk = self._alpha_dk       # Superlink flow coefficient
        _beta_uk = self._beta_uk         # Superlink flow coefficient
        _beta_dk = self._beta_dk         # Superlink flow coefficient
        _chi_uk = self._chi_uk           # Superlink flow coefficient
        _chi_dk = self._chi_dk           # Superlink flow coefficient
        _alpha_ukm = self._alpha_ukm     # Summation of superlink flow coefficients
        _beta_dkl = self._beta_dkl       # Summation of superlink flow coefficients
        _chi_ukl = self._chi_ukl         # Summation of superlink flow coefficients
        _chi_dkm = self._chi_dkm         # Summation of superlink flow coefficients
        _F_jj = self._F_jj
        _A_sj = self._A_sj               # Surface area of superjunction j
        NK = self.NK
        n_o = self.n_o                   # Number of orifices in system
        n_w = self.n_w                   # Number of weirs in system
        n_p = self.n_p                   # Number of pumps in system
        A = self.A
        if n_o:
            O = self.O
            _J_uo = self._J_uo               # Index of superjunction upstream of orifice o
            _J_do = self._J_do               # Index of superjunction upstream of orifice o
            _alpha_o = self._alpha_o         # Orifice flow coefficient
            _beta_o = self._beta_o           # Orifice flow coefficient
            _chi_o = self._chi_o             # Orifice flow coefficient
            _alpha_uom = self._alpha_uom     # Summation of orifice flow coefficients
            _beta_dol = self._beta_dol       # Summation of orifice flow coefficients
            _chi_uol = self._chi_uol         # Summation of orifice flow coefficients
            _chi_dom = self._chi_dom         # Summation of orifice flow coefficients
            _O_diag = self._O_diag           # Diagonal elements of matrix O
        if n_w:
            W = self.W
            _J_uw = self._J_uw               # Index of superjunction upstream of weir w
            _J_dw = self._J_dw               # Index of superjunction downstream of weir w
            _alpha_w = self._alpha_w         # Weir flow coefficient
            _beta_w = self._beta_w           # Weir flow coefficient
            _chi_w = self._chi_w             # Weir flow coefficient
            _alpha_uwm = self._alpha_uwm     # Summation of weir flow coefficients
            _beta_dwl = self._beta_dwl       # Summation of weir flow coefficients
            _chi_uwl = self._chi_uwl         # Summation of weir flow coefficients
            _chi_dwm = self._chi_dwm         # Summation of weir flow coefficients
            _W_diag = self._W_diag           # Diagonal elements of matrix W
        if n_p:
            P = self.P
            _J_up = self._J_up               # Index of superjunction upstream of pump p
            _J_dp = self._J_dp               # Index of superjunction downstream of pump p
            _alpha_p = self._alpha_p         # Pump flow coefficient
            _beta_p = self._beta_p           # Pump flow coefficient
            _chi_p = self._chi_p             # Pump flow coefficient
            _alpha_upm = self._alpha_upm     # Summation of pump flow coefficients
            _beta_dpl = self._beta_dpl       # Summation of pump flow coefficients
            _chi_upl = self._chi_upl         # Summation of pump flow coefficients
            _chi_dpm = self._chi_dpm         # Summation of pump flow coefficients
            _P_diag = self._P_diag           # Diagonal elements of matrix P
        _sparse = self._sparse           # Use sparse matrix data structures (y/n)
        M = self.M                       # Number of superjunctions in system
        H_j = self.H_j                   # Head at superjunction j
        bc = self.bc                     # Superjunction j has a fixed boundary condition (y/n)
        D = self.D                       # Vector for storing chi coefficients
        b = self.b                       # Right-hand side vector
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no boundary head specified, use current superjunction head
        if H_bc is None:
            H_bc = self.H_j
        # If no flow input specified, assume zero external inflow
        if _Q_0j is None:
            _Q_0j = 0
        # If no control input signal specified assume zero input
        if u is None:
            u = 0
        # Clear old data
        _F_jj.fill(0)
        D.fill(0)
        numba_clear_off_diagonals(A, bc, _J_uk, _J_dk, NK)
        # Create A matrix
        numba_create_A_matrix(A, _F_jj, bc, _J_uk, _J_dk, _alpha_uk,
                              _alpha_dk, _beta_uk, _beta_dk, _A_sj, _dt,
                              M, NK)
        # Create D vector
        numba_add_at(D, _J_uk, -_chi_uk)
        numba_add_at(D, _J_dk, _chi_dk)
        # Compute control matrix
        if n_o:
            _alpha_uo = _alpha_o
            _alpha_do = _alpha_o
            _beta_uo = _beta_o
            _beta_do = _beta_o
            _chi_uo = _chi_o
            _chi_do = _chi_o
            _O_diag.fill(0)
            numba_clear_off_diagonals(O, bc, _J_uo, _J_do, n_o)
            # Set diagonal
            numba_create_OWP_matrix(O, _O_diag, bc, _J_uo, _J_do, _alpha_uo,
                                    _alpha_do, _beta_uo, _beta_do, M, n_o)
            # Set right-hand side
            numba_add_at(D, _J_uk, -_chi_uo)
            numba_add_at(D, _J_dk, _chi_do)
        if n_w:
            _alpha_uw = _alpha_w
            _alpha_dw = _alpha_w
            _beta_uw = _beta_w
            _beta_dw = _beta_w
            _chi_uw = _chi_w
            _chi_dw = _chi_w
            _W_diag.fill(0)
            numba_clear_off_diagonals(W, bc, _J_uw, _J_dw, n_w)
            # Set diagonal
            numba_create_OWP_matrix(W, _W_diag, bc, _J_uw, _J_dw, _alpha_uw,
                                    _alpha_dw, _beta_uw, _beta_dw, M, n_w)
            # Set right-hand side
            numba_add_at(D, _J_uw, -_chi_uw)
            numba_add_at(D, _J_dw, _chi_dw)
        if n_p:
            _alpha_up = _alpha_p
            _alpha_dp = _alpha_p
            _beta_up = _beta_p
            _beta_dp = _beta_p
            _chi_up = _chi_p
            _chi_dp = _chi_p
            _P_diag.fill(0)
            numba_clear_off_diagonals(P, bc, _J_up, _J_dp, n_p)
            # Set diagonal
            numba_create_OWP_matrix(P, _P_diag, bc, _J_up, _J_dp, _alpha_up,
                                    _alpha_dp, _beta_up, _beta_dp, M, n_p)
            # Set right-hand side
            numba_add_at(D, _J_up, -_chi_up)
            numba_add_at(D, _J_dp, _chi_dp)
        b.fill(0)
        b = (_A_sj * H_j / _dt) + _Q_0j + D
        # Ensure boundary condition is specified
        b[bc] = H_bc[bc]
        # Export instance variables
        self.D = D
        self.b = b
        # self._beta_dkl = _beta_dkl
        # self._alpha_ukm = _alpha_ukm
        # self._chi_ukl = _chi_ukl
        # self._chi_dkm = _chi_dkm
        if first_time and _sparse:
            self.A = self.A.tocsr()

    def solve_banded_matrix(self, u=None, implicit=True):
        # Import instance variables
        A = self.A                    # Superlink/superjunction matrix
        b = self.b                    # Right-hand side vector
        B = self.B                    # External control matrix
        O = self.O                    # Orifice matrix
        W = self.W                    # Weir matrix
        P = self.P                    # Pump matrix
        n_o = self.n_o                # Number of orifices
        n_w = self.n_w                # Number of weirs
        n_p = self.n_p                # Number of pumps
        _z_inv_j = self._z_inv_j      # Invert elevation of superjunction j
        _sparse = self._sparse        # Use sparse data structures (y/n)
        min_depth = self.min_depth    # Minimum depth at superjunctions
        max_depth = self.max_depth    # Maximum depth at superjunctions
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
        H_j_next = scipy.linalg.solve_banded((bandwidth, bandwidth), AB, r,
                                             check_finite=False, overwrite_ab=True)
        # Constrain heads based on allowed maximum/minimum depths
        H_j_next = np.maximum(H_j_next, _z_inv_j + min_depth)
        H_j_next = np.minimum(H_j_next, _z_inv_j + max_depth)
        # Export instance variables
        self.H_j = H_j_next

    def solve_internals_ls_alt(self):
        NK = self.NK
        nk = self.nk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _kI = self._kI
        _ki = self._ki
        _i_1k = self._i_1k
        _I_1k = self._I_1k
        _k_1k = self._k_1k
        _I_Nk = self._I_Nk
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _is_start = self._is_start
        _is_end = self._is_end
        _is_penult = self._is_penult
        _link_start = self._link_start
        _link_end = self._link_end
        _kk = _ki[~_link_end]
        min_depth = self.min_depth
        # Solve non-negative least squares
        _X = _X_Ik[~_is_start & ~_is_end]
        _U = _U_Ik[~_is_penult & ~_is_end]
        t0 = _W_Ik[~_is_end] * _h_uk[_ki]
        t1 = _Z_Ik[~_is_end] * _h_dk[_ki]
        t2 = _Y_Ik[~_is_end]
        t3 = _V_Ik[~_is_end]
        _b = -t0 + t1 + t2 - t3
        _b[_link_start] += _X_Ik[_is_start] * _h_uk
        _b[_link_end] -= _U_Ik[_is_penult] * _h_dk
        # Call numba function
        _h_Ik = numba_solve_internals_ls(_h_Ik, NK, nk, _k_1k, _i_1k, _I_1k,
                                         _U, _X, _b)
        # Set depths at upstream and downstream ends
        _h_Ik[_is_start] = _h_uk
        _h_Ik[_is_end] = _h_dk
        # Set min depth
        _h_Ik[_h_Ik < min_depth] = min_depth
        # Solve for flows using new depths
        Q_ik_b, Q_ik_f = self.superlink_flow_error()
        _Q_ik = (Q_ik_b + Q_ik_f) / 2
        # Export instance variables
        self._Q_ik = _Q_ik
        self._h_Ik = _h_Ik

@njit
def numba_hydraulic_geometry(_A_ik, _Pe_ik, _R_ik, _B_ik, _h_Ik,
                             _g1_ik, _g2_ik, _g3_ik, _geom_codes, _Ik, _ik):
    n = len(_ik)
    for i in range(n):
        I = _Ik[i]
        Ip1 = I + 1
        geom_code = _geom_codes[i]
        h_I = _h_Ik[I]
        h_Ip1 = _h_Ik[Ip1]
        g1_i = _g1_ik[i]
        g2_i = _g2_ik[i]
        g3_i = _g3_ik[i]
        if geom_code:
            if geom_code == 1:
                _A_ik[i] = superlink.ngeometry.Circular_A_ik(h_I, h_Ip1, g1_i)
                _Pe_ik[i] = superlink.ngeometry.Circular_Pe_ik(h_I, h_Ip1, g1_i)
                _R_ik[i] = superlink.ngeometry.Circular_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = superlink.ngeometry.Circular_B_ik(h_I, h_Ip1, g1_i)
            elif geom_code == 2:
                _A_ik[i] = superlink.ngeometry.Rect_Closed_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = superlink.ngeometry.Rect_Closed_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = superlink.ngeometry.Rect_Closed_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = superlink.ngeometry.Rect_Closed_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 3:
                _A_ik[i] = superlink.ngeometry.Rect_Open_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = superlink.ngeometry.Rect_Open_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = superlink.ngeometry.Rect_Open_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = superlink.ngeometry.Rect_Open_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 4:
                _A_ik[i] = superlink.ngeometry.Triangular_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = superlink.ngeometry.Triangular_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = superlink.ngeometry.Triangular_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = superlink.ngeometry.Triangular_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 5:
                _A_ik[i] = superlink.ngeometry.Trapezoidal_A_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
                _Pe_ik[i] = superlink.ngeometry.Trapezoidal_Pe_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
                _R_ik[i] = superlink.ngeometry.Trapezoidal_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = superlink.ngeometry.Trapezoidal_B_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
    return 1

@njit
def numba_boundary_geometry(_A_bk, _B_bk, _h_Ik, _H_j, _z_inv_bk,
                            _g1_ik, _g2_ik, _g3_ik, _geom_codes,
                            _i_bk, _I_bk, _J_bk):
    n = len(_i_bk)
    for k in range(n):
        i = _i_bk[k]
        I = _I_bk[k]
        j = _J_bk[k]
        # TODO: does not handle "max" mode
        h_I = _h_Ik[I]
        h_Ip1 = _H_j[j] - _z_inv_bk[k]
        geom_code = _geom_codes[i]
        g1_i = _g1_ik[i]
        g2_i = _g2_ik[i]
        g3_i = _g3_ik[i]
        if geom_code:
            if geom_code == 1:
                _A_bk[k] = superlink.ngeometry.Circular_A_ik(h_I, h_Ip1, g1_i)
                _B_bk[k] = superlink.ngeometry.Circular_B_ik(h_I, h_Ip1, g1_i)
            elif geom_code == 2:
                _A_bk[k] = superlink.ngeometry.Rect_Closed_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = superlink.ngeometry.Rect_Closed_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 3:
                _A_bk[k] = superlink.ngeometry.Rect_Open_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = superlink.ngeometry.Rect_Open_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 4:
                _A_bk[k] = superlink.ngeometry.Triangular_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = superlink.ngeometry.Triangular_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 5:
                _A_bk[k] = superlink.ngeometry.Trapezoidal_A_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
                _B_bk[k] = superlink.ngeometry.Trapezoidal_B_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
    return 1

@njit
def numba_solve_internals_ls(_h_Ik, NK, nk, _k_1k, _i_1k, _I_1k, _U, _X, _b):
    for k in range(NK):
        nlinks = nk[k]
        lstart = _k_1k[k]
        rstart = _i_1k[k]
        jstart = _I_1k[k]
        _Ak = np.zeros((nlinks, nlinks - 1))
        for i in range(nlinks - 1):
            _Ak[i, i] = _U[lstart + i]
            _Ak[i + 1, i] = -_X[lstart + i]
        _bk = _b[rstart:rstart+nlinks]
        _AA = _Ak.T @ _Ak
        _Ab = _Ak.T @ _bk
        # If want to prevent singular matrix, set ( diag == 0 ) = 1
        for i in range(nlinks - 1):
            if (_AA[i, i] == 0.0):
                _AA[i, i] = 1.0
        _h_inner = np.linalg.solve(_AA, _Ab)
        _h_Ik[jstart+1:jstart+nlinks] = _h_inner
    return _h_Ik

@njit
def numba_u_ik(_Q_ik, _A_ik, _u_ik):
    n = _u_ik.size
    for i in range(n):
        _Q_i = _Q_ik[i]
        _A_i = _A_ik[i]
        if _A_i:
            _u_ik[i] = _Q_i / _A_i
        else:
            _u_ik[i] = 0
    return _u_ik

@njit
def numba_u_Ik(_dx_ik, _u_ik, _link_start, _u_Ik):
    n = _u_Ik.size
    for i in range(n):
        if _link_start[i]:
            _u_Ik[i] = _u_ik[i]
        else:
            im1 = i - 1
            num = _dx_ik[i] * _u_ik[im1] + _dx_ik[im1] * _u_ik[i]
            den = _dx_ik[i] + _dx_ik[im1]
            if den:
                _u_Ik[i] = num / den
            else:
                _u_Ik[i] = 0
    return _u_Ik

@njit
def numba_u_Ip1k(_dx_ik, _u_ik, _link_end, _u_Ip1k):
    n = _u_Ip1k.size
    for i in range(n):
        if _link_end[i]:
            _u_Ip1k[i] = _u_ik[i]
        else:
            ip1 = i + 1
            num = _dx_ik[i] * _u_ik[ip1] + _dx_ik[ip1] * _u_ik[i]
            den = _dx_ik[i] + _dx_ik[ip1]
            if den:
                _u_Ip1k[i] = num / den
            else:
                _u_Ip1k[i] = 0
    return _u_Ip1k

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
def U_1k(E_2k, c_1k, A_1k, T_1k, g=9.81):
    """
    Compute forward recurrence coefficient 'U' for node 1, superlink k.
    """
    num = E_2k * c_1k - g * A_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit
def V_1k(P_1k, D_2k, c_1k, T_1k, a_1k=0.0, D_1k=0.0):
    """
    Compute forward recurrence coefficient 'V' for node 1, superlink k.
    """
    num = P_1k - D_2k * c_1k + D_1k * a_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit
def W_1k(A_1k, T_1k, a_1k=0.0, E_1k=0.0, g=9.81):
    """
    Compute forward recurrence coefficient 'W' for node 1, superlink k.
    """
    num = g * A_1k - E_1k * a_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit
def T_1k(a_1k, b_1k, c_1k):
    """
    Compute forward recurrence coefficient 'T' for link 1, superlink k.
    """
    return a_1k + b_1k + c_1k

@njit
def U_Ik(E_Ip1k, c_ik, A_ik, T_ik, g=9.81):
    """
    Compute forward recurrence coefficient 'U' for node I, superlink k.
    """
    num = E_Ip1k * c_ik - g * A_ik
    den = T_ik
    result = safe_divide(num, den)
    return result

@njit
def V_Ik(P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ik, V_Im1k, U_Im1k, T_ik, g=9.81):
    """
    Compute forward recurrence coefficient 'V' for node I, superlink k.
    """
    t_0 = P_ik
    t_1 = a_ik * D_Ik
    t_2 = D_Ip1k * c_ik
    t_3 = (g * A_ik - E_Ik * a_ik)
    t_4 = V_Im1k + D_Ik
    t_5 = U_Im1k - E_Ik
    t_6 = T_ik
    # TODO: There is still a divide by zero here
    num = (t_0 + t_1 - t_2 - (t_3 * t_4 / t_5))
    den = t_6
    result = safe_divide(num, den)
    return  result

@njit
def W_Ik(A_ik, E_Ik, a_ik, W_Im1k, U_Im1k, T_ik, g=9.81):
    """
    Compute forward recurrence coefficient 'W' for node I, superlink k.
    """
    num = -(g * A_ik - E_Ik * a_ik) * W_Im1k
    den = (U_Im1k - E_Ik) * T_ik
    result = safe_divide(num, den)
    return result

@njit
def T_ik(a_ik, b_ik, c_ik, A_ik, E_Ik, U_Im1k, g=9.81):
    """
    Compute forward recurrence coefficient 'T' for link i, superlink k.
    """
    t_0 = a_ik + b_ik + c_ik
    t_1 = g * A_ik - E_Ik * a_ik
    t_2 = U_Im1k - E_Ik
    result = t_0 - safe_divide(t_1, t_2)
    return result

@njit
def X_Nk(A_nk, E_Nk, a_nk, O_nk, g=9.81):
    """
    Compute backward recurrence coefficient 'X' for node N, superlink k.
    """
    num = g * A_nk - E_Nk * a_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit
def Y_Nk(P_nk, D_Nk, a_nk, O_nk, c_nk=0.0, D_Np1k=0.0):
    """
    Compute backward recurrence coefficient 'Y' for node N, superlink k.
    """
    num = P_nk + D_Nk * a_nk - D_Np1k * c_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit
def Z_Nk(A_nk, O_nk, c_nk=0.0, E_Np1k=0.0, g=9.81):
    """
    Compute backward recurrence coefficient 'Z' for node N, superlink k.
    """
    num = E_Np1k * c_nk - g * A_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit
def O_nk(a_nk, b_nk, c_nk):
    """
    Compute backward recurrence coefficient 'O' for link n, superlink k.
    """
    return a_nk + b_nk + c_nk

@njit
def X_Ik(A_ik, E_Ik, a_ik, O_ik, g=9.81):
    """
    Compute backward recurrence coefficient 'X' for node I, superlink k.
    """
    num = g * A_ik - E_Ik * a_ik
    den = O_ik
    result = safe_divide(num, den)
    return result

@njit
def Y_Ik(P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ip1k, Y_Ip1k, X_Ip1k, O_ik, g=9.81):
    """
    Compute backward recurrence coefficient 'Y' for node I, superlink k.
    """
    t_0 = P_ik
    t_1 = a_ik * D_Ik
    t_2 = D_Ip1k * c_ik
    t_3 = (g * A_ik - E_Ip1k * c_ik)
    t_4 = D_Ip1k - Y_Ip1k
    t_5 = X_Ip1k + E_Ip1k
    t_6 = O_ik
    # TODO: There is still a divide by zero here
    num = (t_0 + t_1 - t_2 - (t_3 * t_4 / t_5))
    den = t_6
    result = safe_divide(num, den)
    return result

@njit
def Z_Ik(A_ik, E_Ip1k, c_ik, Z_Ip1k, X_Ip1k, O_ik, g=9.81):
    """
    Compute backward recurrence coefficient 'Z' for node I, superlink k.
    """
    num = (g * A_ik - E_Ip1k * c_ik) * Z_Ip1k
    den = (X_Ip1k + E_Ip1k) * O_ik
    result = safe_divide(num, den)
    return result

@njit
def O_ik(a_ik, b_ik, c_ik, A_ik, E_Ip1k, X_Ip1k, g=9.81):
    """
    Compute backward recurrence coefficient 'O' for link i, superlink k.
    """
    t_0 = a_ik + b_ik + c_ik
    t_1 = g * A_ik - E_Ip1k * c_ik
    t_2 = X_Ip1k + E_Ip1k
    result = t_0 + safe_divide(t_1, t_2)
    return result

@njit
def numba_D_k_star(X_1k, kappa_uk, U_Nk, kappa_dk, Z_1k, W_Nk):
    """
    Compute superlink boundary condition coefficient 'D_k_star'.
    """
    t_0 = (X_1k * kappa_uk - 1) * (U_Nk * kappa_dk - 1)
    t_1 = (Z_1k * kappa_dk) * (W_Nk * kappa_uk)
    result = t_0 - t_1
    return result

@njit
def numba_alpha_uk(U_Nk, kappa_dk, X_1k, Z_1k, W_Nk, D_k_star, lambda_uk, theta_uk):
    """
    Compute superlink boundary condition coefficient 'alpha' for upstream end
    of superlink k.
    """
    num = theta_uk * ((1 - U_Nk * kappa_dk) * X_1k * lambda_uk
                        + (Z_1k * kappa_dk * W_Nk * lambda_uk))
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_beta_uk(U_Nk, kappa_dk, Z_1k, W_Nk, D_k_star, lambda_dk, theta_dk):
    """
    Compute superlink boundary condition coefficient 'beta' for upstream end
    of superlink k.
    """
    num = theta_dk * ((1 - U_Nk * kappa_dk) * Z_1k * lambda_dk
            + (Z_1k * kappa_dk * U_Nk * lambda_dk))
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_chi_uk(U_Nk, kappa_dk, Y_1k, X_1k, mu_uk, Z_1k,
                    mu_dk, V_Nk, W_Nk, D_k_star, theta_uk, theta_dk):
    """
    Compute superlink boundary condition coefficient 'chi' for upstream end
    of superlink k.
    """
    t_0 = (1 - U_Nk * kappa_dk) * (Y_1k + theta_uk * X_1k * mu_uk + theta_dk * Z_1k * mu_dk)
    t_1 = (Z_1k * kappa_dk) * (V_Nk + theta_uk * W_Nk * mu_uk + theta_dk * U_Nk * mu_dk)
    num = t_0 + t_1
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_alpha_dk(X_1k, kappa_uk, W_Nk, D_k_star, lambda_uk, theta_uk):
    """
    Compute superlink boundary condition coefficient 'alpha' for downstream end
    of superlink k.
    """
    num = theta_uk * ((1 - X_1k * kappa_uk) * W_Nk * lambda_uk
            + (W_Nk * kappa_uk * X_1k * lambda_uk))
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_beta_dk(X_1k, kappa_uk, U_Nk, W_Nk, Z_1k, D_k_star, lambda_dk, theta_dk):
    """
    Compute superlink boundary condition coefficient 'beta' for downstream end
    of superlink k.
    """
    num = theta_dk * ((1 - X_1k * kappa_uk) * U_Nk * lambda_dk
            + (W_Nk * kappa_uk * Z_1k * lambda_dk))
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_chi_dk(X_1k, kappa_uk, V_Nk, W_Nk, mu_uk, U_Nk,
                    mu_dk, Y_1k, Z_1k, D_k_star, theta_uk, theta_dk):
    """
    Compute superlink boundary condition coefficient 'chi' for downstream end
    of superlink k.
    """
    t_0 = (1 - X_1k * kappa_uk) * (V_Nk + theta_uk * W_Nk * mu_uk + theta_dk * U_Nk * mu_dk)
    t_1 = (W_Nk * kappa_uk) * (Y_1k + theta_uk * X_1k * mu_uk + theta_dk * Z_1k * mu_dk)
    num = t_0 + t_1
    den = D_k_star
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _a_ik, _b_ik, _c_ik,
                             _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_1k, _i_1k):
    for k in range(NK):
        # Start at junction 1
        _I_1 = _I_1k[k]
        _i_1 = _i_1k[k]
        _I_2 = _I_1 + 1
        _i_2 = _i_1 + 1
        nlinks = nk[k]
        _T_ik[_i_1] = T_1k(_a_ik[_i_1], _b_ik[_i_1], _c_ik[_i_1])
        _U_Ik[_I_1] = U_1k(_E_Ik[_I_2], _c_ik[_i_1], _A_ik[_i_1], _T_ik[_i_1])
        _V_Ik[_I_1] = V_1k(_P_ik[_i_1], _D_Ik[_I_2], _c_ik[_i_1], _T_ik[_i_1],
                            _a_ik[_i_1], _D_Ik[_I_1])
        _W_Ik[_I_1] = W_1k(_A_ik[_i_1], _T_ik[_i_1], _a_ik[_i_1], _E_Ik[_I_1])
        # Loop from junction 2 -> Nk
        for i in range(nlinks - 1):
            _i_next = _i_2 + i
            _I_next = _I_2 + i
            _Im1_next = _I_next - 1
            _Ip1_next = _I_next + 1
            _T_ik[_i_next] = T_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                  _A_ik[_i_next], _E_Ik[_I_next], _U_Ik[_Im1_next])
            _U_Ik[_I_next] = U_Ik(_E_Ik[_Ip1_next], _c_ik[_i_next],
                                  _A_ik[_i_next], _T_ik[_i_next])
            _V_Ik[_I_next] = V_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                  _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                  _E_Ik[_I_next], _V_Ik[_Im1_next], _U_Ik[_Im1_next],
                                  _T_ik[_i_next])
            _W_Ik[_I_next] = W_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                  _W_Ik[_Im1_next], _U_Ik[_Im1_next], _T_ik[_i_next])
    return 1

@njit
def numba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _a_ik, _b_ik, _c_ik,
                              _P_ik, _A_ik, _E_Ik, _D_Ik, NK, nk, _I_Nk, _i_nk):
    for k in range(NK):
        _I_N = _I_Nk[k]
        _i_n = _i_nk[k]
        _I_Nm1 = _I_N - 1
        _i_nm1 = _i_n - 1
        _I_Np1 = _I_N + 1
        nlinks = nk[k]
        _O_ik[_i_n] = O_nk(_a_ik[_i_n], _b_ik[_i_n], _c_ik[_i_n])
        _X_Ik[_I_N] = X_Nk(_A_ik[_i_n], _E_Ik[_I_N], _a_ik[_i_n], _O_ik[_i_n])
        _Y_Ik[_I_N] = Y_Nk(_P_ik[_i_n], _D_Ik[_I_N], _a_ik[_i_n], _O_ik[_i_n],
                            _c_ik[_i_n], _D_Ik[_I_Np1])
        _Z_Ik[_I_N] = Z_Nk(_A_ik[_i_n], _O_ik[_i_n], _c_ik[_i_n], _E_Ik[_I_Np1])
        for i in range(nlinks - 1):
            _i_next = _i_nm1 - i
            _I_next = _I_Nm1 - i
            _Ip1_next = _I_next + 1
            _O_ik[_i_next] = O_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                  _A_ik[_i_next], _E_Ik[_Ip1_next], _X_Ik[_Ip1_next])
            _X_Ik[_I_next] = X_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                  _O_ik[_i_next])
            _Y_Ik[_I_next] = Y_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                  _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                  _E_Ik[_Ip1_next], _Y_Ik[_Ip1_next], _X_Ik[_Ip1_next],
                                  _O_ik[_i_next])
            _Z_Ik[_I_next] = Z_Ik(_A_ik[_i_next], _E_Ik[_Ip1_next], _c_ik[_i_next],
                                  _Z_Ik[_Ip1_next], _X_Ik[_Ip1_next], _O_ik[_i_next])
    return 1

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

@njit(fastmath=True)
def numba_add_at(a, indices, b):
    n = len(indices)
    for k in range(n):
        i = indices[k]
        a[i] += b[k]

@njit(fastmath=True)
def numba_add_at_2d(a, indices_0, indices_1, b):
    n = len(indices_0)
    for k in range(n):
        i = indices_0[k]
        j = indices_1[k]
        a[i,j] += b[k]

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
def numba_create_A_matrix(A, _F_jj, bc, _J_uk, _J_dk, _alpha_uk,
                          _alpha_dk, _beta_uk, _beta_dk, _A_sj, _dt,
                          M, NK):
    numba_add_at(_F_jj, _J_uk, _alpha_uk)
    numba_add_at(_F_jj, _J_dk, -_beta_dk)
    _F_jj += (_A_sj / _dt)
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
            A[_J_u, _J_d] += _beta_uk[k]
        if not _bc_d:
            A[_J_d, _J_u] -= _alpha_dk[k]

@njit(fastmath=True)
def numba_create_OWP_matrix(X, diag, bc, _J_uc, _J_dc, _alpha_uc,
                            _alpha_dc, _beta_uc, _beta_dc, M, NC):
    # Set diagonal
    numba_add_at(diag, _J_uc, _alpha_uc)
    numba_add_at(diag, _J_dc, -_beta_dc)
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
            X[_J_u, _J_d] += _beta_uc[c]
        if not _bc_d:
            X[_J_d, _J_u] -= _alpha_dc[c]

