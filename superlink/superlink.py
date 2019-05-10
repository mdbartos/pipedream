import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import superlink.geometry
import superlink.storage

class SuperLink():
    def __init__(self, superlinks, superjunctions, links, junctions,
                 transects={}, storages={}, orifices=None,
                 dt=60, sparse=False, min_depth=1e-5, method='b',
                 inertial_damping=False):
        self.superlinks = superlinks
        self.superjunctions = superjunctions
        self.links = links
        self.junctions = junctions
        self.transects = transects
        self.storages = storages
        self.orifices = orifices
        self._dt = dt
        self._sparse = sparse
        self._method = method
        self.inertial_damping = inertial_damping
        self.min_depth = min_depth
        self._I = junctions.index.values
        self._ik = links.index.values
        self._Ik = links['j_0'].values.astype(int)
        self._Ip1k = links['j_1'].values.astype(int)
        self._kI = junctions['k'].values.astype(int)
        self._ki = links['k'].values.astype(int)
        self.start_nodes = superlinks['j_0'].values.astype(int)
        self.end_nodes = superlinks['j_1'].values.astype(int)
        self._is_start = np.zeros(self._I.size, dtype=bool)
        self._is_end = np.zeros(self._I.size, dtype=bool)
        self._is_start[self.start_nodes] = True
        self._is_end[self.end_nodes] = True
        self.middle_nodes = self._I[(~self._is_start) & (~self._is_end)]
        # Create forward and backward indexers
        self.forward_I_I = np.copy(self._I)
        self.forward_I_I[self._Ik] = self._Ip1k
        self.backward_I_I = np.copy(self._I)
        self.backward_I_I[self._Ip1k] = self._Ik
        self.forward_I_i = np.copy(self._I)
        self.forward_I_i[self._Ik] = self._ik
        self.backward_I_i = np.copy(self._I)
        self.backward_I_i[self._Ip1k] = self._ik
        self.forward_I_i[self.end_nodes] = self.backward_I_i[self.start_nodes]
        self.backward_I_i[self.start_nodes] = self.forward_I_i[self.start_nodes]
        # Handle channel geometries
        self._shape_ik = links['shape']
        if transects:
            self._transect_ik = links['ts']
        else:
            self._transect_ik = None
        self._g1_ik = links['g1'].values.astype(float)
        self._g2_ik = links['g2'].values.astype(float)
        self._g3_ik = links['g3'].values.astype(float)
        self._Q_ik = links['Q_0'].values.astype(float)
        self._dx_ik = links['dx'].values.astype(float)
        self._n_ik = links['n'].values.astype(float)
        self._ctrl = links['ctrl'].values.astype(bool)
        self._A_c_ik = links['A_c'].values.astype(float)
        self._C_ik = links['C'].values.astype(float)
        self._storage_type = superjunctions['storage']
        if storages:
            self._storage_table = superjunctions['table']
        else:
            self._storage_table = None
        self._storage_a = superjunctions['a'].values.astype(float)
        self._storage_b = superjunctions['b'].values.astype(float)
        self._storage_c = superjunctions['c'].values.astype(float)
        if 'max_depth' in superjunctions:
            self.max_depth = superjunctions['max_depth'].values.astype(float)
        else:
            self.max_depth = np.full(len(superjunctions), np.inf, dtype=float)
        self._h_Ik = junctions.loc[self._I, 'h_0'].values.astype(float)
        self._A_SIk = junctions.loc[self._I, 'A_s'].values.astype(float)
        self._z_inv_Ik = junctions.loc[self._I, 'z_inv'].values.astype(float)
        self._S_o_ik = ((self._z_inv_Ik[self._Ik] - self._z_inv_Ik[self._Ip1k])
                        / self._dx_ik)
        # TODO: Allow specifying initial flows
        self._Q_0Ik = np.zeros(self._I.size, dtype=float)
        # Handle orifices
        if orifices is not None:
            self._J_uo = self.orifices['sj_0'].values.astype(int)
            self._J_do = self.orifices['sj_1'].values.astype(int)
            self._Ao = self.orifices['A'].values.astype(float)
            self.p = self.orifices.shape[0]
        else:
            self._J_uo = np.array([], dtype=int)
            self._J_do = np.array([], dtype=int)
            self._Ao = np.array([], dtype=float)
            self.p = 0
        # Enforce minimum depth
        self._h_Ik = np.maximum(self._h_Ik, self.min_depth)
        # Computational arrays
        self._A_ik = np.zeros(self._ik.size)
        self._Pe_ik = np.zeros(self._ik.size)
        self._R_ik = np.zeros(self._ik.size)
        self._B_ik = np.zeros(self._ik.size)
        # self._Q_ik_f = np.zeros_like(self._Q_ik)
        # self._Q_ik_b = np.zeros_like(self._Q_ik)
        # self._h_Ik_f = np.zeros_like(self._h_Ik)
        # self._h_Ik_b = np.zeros_like(self._h_Ik)
        # Node velocities
        self._u_Ik = np.zeros(self._Ik.size, dtype=float)
        self._u_Ip1k = np.zeros(self._Ip1k.size, dtype=float)
        # Node coefficients
        self._E_Ik = np.zeros(self._I.size)
        self._D_Ik = np.zeros(self._I.size)
        # Forward recurrence relations
        self._I_end = np.zeros(self._I.size, dtype=bool)
        self._I_end[self.end_nodes] = True
        self._I_1k = self.start_nodes
        self._I_2k = self.forward_I_I[self._I_1k]
        self._i_1k = self.forward_I_i[self._I_1k]
        self._T_ik = np.zeros(self._ik.size)
        self._U_Ik = np.zeros(self._I.size)
        self._V_Ik = np.zeros(self._I.size)
        self._W_Ik = np.zeros(self._I.size)
        # Backward recurrence relations
        self._I_start = np.zeros(self._I.size, dtype=bool)
        self._I_start[self.start_nodes] = True
        self._I_Np1k = self.end_nodes
        self._I_Nk = self.backward_I_I[self._I_Np1k]
        self._i_nk = self.backward_I_i[self._I_Np1k]
        self._O_ik = np.zeros(self._ik.size)
        self._X_Ik = np.zeros(self._I.size)
        self._Y_Ik = np.zeros(self._I.size)
        self._Z_Ik = np.zeros(self._I.size)
        # Head at superjunctions
        self._z_inv_j = self.superjunctions['z_inv'].values
        self.H_j = self.superjunctions['h_0'].values + self._z_inv_j
        # Enforce minimum depth
        self.H_j = np.maximum(self.H_j, self._z_inv_j + self.min_depth)
        # Coefficients for head at upstream ends of superlink k
        self._J_uk = self.superlinks['sj_0'].values.astype(int)
        self._z_inv_uk = self._z_inv_Ik[self._I_1k]
        # Coefficients for head at downstream ends of superlink k
        self._J_dk = self.superlinks['sj_1'].values.astype(int)
        self._z_inv_dk = self._z_inv_Ik[self._I_Np1k]
        # Sparse matrix coefficients
        self.M = len(superjunctions)
        self.NK = len(superlinks)
        self.nk = np.bincount(self._ki)
        if sparse:
            self.A = scipy.sparse.lil_matrix((self.M, self.M))
        else:
            self.A = np.zeros((self.M, self.M))
        self.b = np.zeros(self.M)
        self.bc = self.superjunctions['bc'].values.astype(bool)
        if sparse:
            self.B = scipy.sparse.lil_matrix((self.M, self.p))
        else:
            self.B = np.zeros((self.M, self.p))
        if sparse:
            self.O = scipy.sparse.lil_matrix((self.M, self.M))
        else:
            self.O = np.zeros((self.M, self.M))
        # TODO: Should these be size NK?
        self._alpha_ukm = np.zeros(self.M, dtype=float)
        self._beta_dkl = np.zeros(self.M, dtype=float)
        self._chi_ukl = np.zeros(self.M, dtype=float)
        self._chi_dkm = np.zeros(self.M, dtype=float)
        self._k = superlinks.index.values
        self._A_sj = np.zeros(self.M, dtype=float)
        # TODO: Allow initial input to be specified
        self._Q_0j = 0
        # Set upstream and downstream superlink variables
        self._Q_uk = self._Q_ik[self._i_1k]
        self._Q_dk = self._Q_ik[self._i_nk]
        self._h_uk = self._h_Ik[self._I_1k]
        self._h_dk = self._h_Ik[self._I_Np1k]
        # Other parameters
        # self._Qo_t = self.min_Qo(self._J_uo, self._J_do, self._Ao, self.H_j)
        self._Qo_t, _ = self.B_j(self._J_uo, self._J_do, self._Ao, self.H_j)
        # self._Qo_t_min = self.min_Qo(self._J_uo, self._J_do, self._Ao, self.H_j)
        self._O_diag = np.zeros(self.M)
        # Set up hydraulic geometry computations
        self.configure_storages()
        self.configure_hydraulic_geometry()
        # Get indexers for least squares
        self.lsq_indexers()
        # Initialize to stable state
        self.step(dt=1e-6, first_time=True)

    def lsq_indexers(self):
        """
        Initialize matrices and indices for least-squares computation.
        """
        _ik = self._ik
        _ki = self._ki
        _kI = self._kI
        _I = self._I
        _I_start = self._I_start
        _I_end = self._I_end
        _I_1k = self._I_1k
        _I_Nk = self._I_Nk
        _I_Np1k = self._I_Np1k
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _sparse = self._sparse
        # Create forward and backward indexers
        _I_f = np.ones(_I.size, dtype=bool)
        _I_b = np.ones(_I.size, dtype=bool)
        _I_f[_I_Nk] = False
        _I_f[_I_Np1k] = False
        _I_b[_I_1k] = False
        _I_b[_I_Np1k] = False
        # Create centered indexers
        _i_c = np.ones(_ik.size, dtype=bool)
        _i_c[_i_1k] = False
        _i_c[_i_nk] = False
        middle_links = np.flatnonzero(_i_c)
        # Set parameters
        _NK = self._k.size
        m = 2*_ik.size - 2*_NK
        n = 2*_ik.size - 3*_NK
        # Create sparse matrices
        if _sparse:
            _G = scipy.sparse.lil_matrix((m, n))
        else:
            _G = np.zeros((m, n))
        _h = np.zeros(m)
        # Create row and column indexers for sparse matrices
        _kn = np.bincount(_ki)
        _Ih = np.concatenate([np.arange(0, k - 1, dtype=int) for k in _kn])
        _ih = np.concatenate([np.arange(0, k - 2, dtype=int) for k in _kn])
        h_spacing = 2*(_kn) - 3
        l_spacing = np.cumsum(h_spacing) - h_spacing
        v_spacing = 2*(_kn) - 2
        r_spacing = _kn - 1
        _b_f = np.cumsum(v_spacing) - v_spacing
        _b_b = np.cumsum(v_spacing) - 1
        _Ih_k = _kI[(~_I_start) & (~_I_end)]
        _cols_l = _Ih + l_spacing[_Ih_k]
        _rows_f = np.arange(0, m, 2)
        _rows_b = np.arange(1, m, 2)
        # TODO: Check this section for more links
        _ih_k = _ki[_i_c]
        _cols_r = _ih + l_spacing[_ih_k] + r_spacing[_ih_k]
        _cols_r = np.repeat(_cols_r, 2)
        _rows_r = np.arange(m, dtype=int)
        _rows_r = np.delete(_rows_r, [_b_f, _b_b])
        _lbound = np.full(n, -np.inf, dtype=float)
        _ubound = np.full(n, np.inf, dtype=float)
        _kx = np.cumsum(_kn) - _kn
        _ix_h = np.concatenate([np.arange(l, r) for l, r
                                in zip(_kx, _kx + r_spacing)])
        _ix_q = np.ones(n, dtype=bool)
        _ix_q[_ix_h] = False
        _ix_q = np.flatnonzero(_ix_q)
        _lbound[_ix_h] = 0
        self.middle_links = middle_links
        self._G = _G
        self._h = _h
        self._I_f = _I_f
        self._I_b = _I_b
        self._b_f = _b_f
        self._b_b = _b_b
        self._rows_f = _rows_f
        self._rows_b = _rows_b
        self._rows_r = _rows_r
        self._cols_l = _cols_l
        self._cols_r = _cols_r
        self._lbound = _lbound
        self._ubound = _ubound
        self._ix_h = _ix_h
        self._ix_q = _ix_q

    def safe_divide(function):
        """
        Allow for division by zero. Division by zero will return zero.
        """
        def inner(*args, **kwargs):
            num, den = function(*args, **kwargs)
            cond = (den != 0)
            result = np.zeros(num.size)
            result[cond] = num[cond] / den[cond]
            return result
        return inner

    @safe_divide
    def u_ik(self, Q_ik, A_ik):
        """
        Compute velocity of flow for link i, superlink k.
        """
        num = Q_ik
        den = np.where(A_ik > 0, A_ik, 0)
        return num, den

    @safe_divide
    def u_Ip1k(self, dx_ik, u_ip1k, dx_ip1k, u_ik):
        """
        Compute approximate velocity of flow for node I+1, superlink k
        using interpolation.
        """
        num = dx_ik * u_ip1k + dx_ip1k * u_ik
        den = dx_ik + dx_ip1k
        return num, den

    @safe_divide
    def u_Ik(self, dx_ik, u_im1k, dx_im1k, u_ik):
        """
        Compute approximate velocity of flow for node I, superlink k
        using interpolation.
        """
        num = dx_ik * u_im1k + dx_im1k * u_ik
        den = dx_ik + dx_im1k
        return num, den

    @safe_divide
    def Fr(self, u_ik, A_ik, B_ik, g=9.81):
        num = np.abs(u_ik) * np.sqrt(B_ik)
        den = np.sqrt(g * A_ik)
        return num, den

    # Link coefficients for superlink k
    def a_ik(self, u_Ik, sigma_ik=1):
        """
        Compute link coefficient 'a' for link i, superlink k.
        """
        return -np.maximum(u_Ik, 0) * sigma_ik

    def c_ik(self, u_Ip1k, sigma_ik=1):
        """
        Compute link coefficient 'c' for link i, superlink k.
        """
        return -np.maximum(-u_Ip1k, 0) * sigma_ik

    def b_ik(self, dx_ik, dt, n_ik, Q_ik_t, A_ik, R_ik,
             A_c_ik, C_ik, a_ik, c_ik, ctrl, sigma_ik=1, g=9.81):
        """
        Compute link coefficient 'b' for link i, superlink k.
        """
        # TODO: Clean up
        cond = A_ik > 0
        t_0 = (dx_ik / dt) * sigma_ik
        t_1 = np.zeros(Q_ik_t.size)
        t_1[cond] = (g * n_ik[cond]**2 * np.abs(Q_ik_t[cond]) * dx_ik[cond]
                    / A_ik[cond] / R_ik[cond]**(4/3))
        t_2 = np.zeros(ctrl.size)
        cond = ctrl
        t_2[cond] = A_ik[cond] * np.abs(Q_ik_t[cond]) / A_c_ik[cond]**2 / C_ik[cond]**2
        t_3 = a_ik
        t_4 = c_ik
        return t_0 + t_1 + t_2 - t_3 - t_4

    def P_ik(self, Q_ik_t, dx_ik, dt, A_ik, S_o_ik, sigma_ik=1, g=9.81):
        """
        Compute link coefficient 'P' for link i, superlink k.
        """
        t_0 = (Q_ik_t * dx_ik / dt) * sigma_ik
        t_1 = g * A_ik * S_o_ik * dx_ik
        return t_0 + t_1

    # Node coefficients for superlink k
    def E_Ik(self, B_ik, dx_ik, B_im1k, dx_im1k, A_SIk, dt):
        """
        Compute node coefficient 'E' for node I, superlink k.
        """
        t_0 = B_ik * dx_ik / 2
        t_1 = B_im1k * dx_im1k / 2
        t_2 = A_SIk
        t_3 = dt
        return (t_0 + t_1 + t_2) / t_3

    def D_Ik(self, Q_0IK, B_ik, dx_ik, B_im1k, dx_im1k, A_SIk, h_Ik_t, dt):
        """
        Compute node coefficient 'D' for node I, superlink k.
        """
        t_0 = Q_0IK
        t_1 = B_ik * dx_ik / 2
        t_2 = B_im1k * dx_im1k / 2
        t_3 = A_SIk
        t_4 = h_Ik_t / dt
        return t_0 + ((t_1 + t_2 + t_3) * t_4)

    @safe_divide
    def U_1k(self, E_2k, c_1k, A_1k, T_1k, g=9.81):
        """
        Compute forward recurrence coefficient 'U' for node 1, superlink k.
        """
        num = E_2k * c_1k - g * A_1k
        den = T_1k
        return num, den

    @safe_divide
    def V_1k(self, P_1k, D_2k, c_1k, T_1k):
        """
        Compute forward recurrence coefficient 'V' for node 1, superlink k.
        """
        num = P_1k - D_2k * c_1k
        den = T_1k
        return num, den

    @safe_divide
    def W_1k(self, A_1k, T_1k, g=9.81):
        """
        Compute forward recurrence coefficient 'W' for node 1, superlink k.
        """
        num = g * A_1k
        den = T_1k
        return num, den

    def T_1k(self, a_1k, b_1k, c_1k):
        """
        Compute forward recurrence coefficient 'T' for link 1, superlink k.
        """
        return a_1k + b_1k + c_1k

    @safe_divide
    def U_Ik(self, E_Ip1k, c_ik, A_ik, T_ik, g=9.81):
        """
        Compute forward recurrence coefficient 'U' for node I, superlink k.
        """
        num = E_Ip1k * c_ik - g * A_ik
        den = T_ik
        return num, den

    @safe_divide
    def V_Ik(self, P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ik, V_Im1k, U_Im1k, T_ik, g=9.81):
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
        return  num, den

    @safe_divide
    def W_Ik(self, A_ik, E_Ik, a_ik, W_Im1k, U_Im1k, T_ik, g=9.81):
        """
        Compute forward recurrence coefficient 'W' for node I, superlink k.
        """
        num = -(g * A_ik - E_Ik * a_ik) * W_Im1k
        den = (U_Im1k - E_Ik) * T_ik
        return num, den

    def T_ik(self, a_ik, b_ik, c_ik, A_ik, E_Ik, U_Im1k, g=9.81):
        """
        Compute forward recurrence coefficient 'T' for link i, superlink k.
        """
        t_0 = a_ik + b_ik + c_ik
        t_1 = g * A_ik - E_Ik * a_ik
        t_2 = U_Im1k - E_Ik
        # TODO: Can't use decorator here
        cond = t_2 != 0
        result = np.zeros(t_0.size)
        # TODO: Not sure if ~cond should be zero
        result[cond] = t_0[cond] - (t_1[cond] / t_2[cond])
        return result

    # Reverse recurrence relation coefficients
    @safe_divide
    def X_Nk(self, A_nk, E_Nk, a_nk, O_nk, g=9.81):
        """
        Compute backward recurrence coefficient 'X' for node N, superlink k.
        """
        num = g * A_nk - E_Nk * a_nk
        den = O_nk
        return num, den

    @safe_divide
    def Y_Nk(self, P_nk, D_Nk, a_nk, O_nk):
        """
        Compute backward recurrence coefficient 'Y' for node N, superlink k.
        """
        num = P_nk + D_Nk * a_nk
        den = O_nk
        return num, den

    @safe_divide
    def Z_Nk(self, A_nk, O_nk, g=9.81):
        """
        Compute backward recurrence coefficient 'Z' for node N, superlink k.
        """
        num = - g * A_nk
        den = O_nk
        return num, den

    def O_nk(self, a_nk, b_nk, c_nk):
        """
        Compute backward recurrence coefficient 'O' for link n, superlink k.
        """
        return a_nk + b_nk + c_nk

    @safe_divide
    def X_Ik(self, A_ik, E_Ik, a_ik, O_ik, g=9.81):
        """
        Compute backward recurrence coefficient 'X' for node I, superlink k.
        """
        num = g * A_ik - E_Ik * a_ik
        den = O_ik
        return num, den

    @safe_divide
    def Y_Ik(self, P_ik, a_ik, D_Ik, D_Ip1k, c_ik, A_ik, E_Ip1k, Y_Ip1k, X_Ip1k, O_ik, g=9.81):
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
        num = (t_0 + t_1 - t_2 - (t_3 * t_4 / t_5))
        den = t_6
        return num, den

    @safe_divide
    def Z_Ik(self, A_ik, E_Ip1k, c_ik, Z_Ip1k, X_Ip1k, O_ik, g=9.81):
        """
        Compute backward recurrence coefficient 'Z' for node I, superlink k.
        """
        num = (g * A_ik - E_Ip1k * c_ik) * Z_Ip1k
        den = (X_Ip1k + E_Ip1k) * O_ik
        return num, den

    def O_ik(self, a_ik, b_ik, c_ik, A_ik, E_Ip1k, X_Ip1k, g=9.81):
        """
        Compute backward recurrence coefficient 'O' for link i, superlink k.
        """
        t_0 = a_ik + b_ik + c_ik
        t_1 = g * A_ik - E_Ip1k * c_ik
        t_2 = X_Ip1k + E_Ip1k
        cond = t_2 != 0
        result = np.zeros(t_0.size)
        # TODO: Not sure if ~cond should be zero
        result[cond] = t_0[cond] + (t_1[cond] / t_2[cond])
        return result

    @safe_divide
    def gamma_uk(self, Q_uk_t, C_uk, A_uk, g=9.81):
        """
        Compute flow coefficient 'gamma' for upstream end of superlink k
        """
        num = -np.abs(Q_uk_t)
        den = 2 * (C_uk**2) * (A_uk**2) * g
        return num, den

    @safe_divide
    def gamma_dk(self, Q_dk_t, C_dk, A_dk, g=9.81):
        """
        Compute flow coefficient 'gamma' for downstream end of superlink k
        """
        num = np.abs(Q_dk_t)
        den = 2 * (C_dk**2) * (A_dk**2) * g
        return num, den

    def D_k_star(self, X_1k, gamma_uk, U_Nk, gamma_dk, Z_1k, W_Nk):
        """
        Compute superlink boundary condition coefficient 'D_k_star'.
        """
        t_0 = (X_1k * gamma_uk - 1) * (U_Nk * gamma_dk - 1)
        t_1 = (Z_1k * gamma_dk) * (W_Nk * gamma_uk)
        return t_0 - t_1

    @safe_divide
    def alpha_uk(self, U_Nk, gamma_dk, X_1k, Z_1k, W_Nk, D_k_star):
        """
        Compute superlink boundary condition coefficient 'alpha' for upstream end
        of superlink k.
        """
        num = (1 - U_Nk * gamma_dk) * X_1k + (Z_1k * gamma_dk * W_Nk)
        den = D_k_star
        return num, den

    @safe_divide
    def beta_uk(self, U_Nk, gamma_dk, Z_1k, W_Nk, D_k_star):
        """
        Compute superlink boundary condition coefficient 'beta' for upstream end
        of superlink k.
        """
        num = (1 - U_Nk * gamma_dk) * Z_1k + (Z_1k * gamma_dk * U_Nk)
        den = D_k_star
        return num, den

    @safe_divide
    def chi_uk(self, U_Nk, gamma_dk, Y_1k, X_1k, z_inv_uk, Z_1k,
               z_inv_dk, V_Nk, W_Nk, D_k_star):
        """
        Compute superlink boundary condition coefficient 'chi' for upstream end
        of superlink k.
        """
        t_0 = (1 - U_Nk * gamma_dk) * (Y_1k - X_1k * z_inv_uk - Z_1k * z_inv_dk)
        t_1 = (Z_1k * gamma_dk) * (V_Nk - W_Nk * z_inv_uk - U_Nk * z_inv_dk)
        num = t_0 + t_1
        den = D_k_star
        return num, den

    @safe_divide
    def alpha_dk(self, X_1k, gamma_uk, W_Nk, D_k_star):
        """
        Compute superlink boundary condition coefficient 'alpha' for downstream end
        of superlink k.
        """
        num = (1 - X_1k * gamma_uk) * W_Nk + (W_Nk * gamma_uk * X_1k)
        den = D_k_star
        return num, den

    @safe_divide
    def beta_dk(self, X_1k, gamma_uk, U_Nk, W_Nk, Z_1k, D_k_star):
        """
        Compute superlink boundary condition coefficient 'beta' for downstream end
        of superlink k.
        """
        num = (1 - X_1k * gamma_uk) * U_Nk + (W_Nk * gamma_uk * Z_1k)
        den = D_k_star
        return num, den

    @safe_divide
    def chi_dk(self, X_1k, gamma_uk, V_Nk, W_Nk, z_inv_uk, U_Nk,
               z_inv_dk, Y_1k, Z_1k, D_k_star):
        """
        Compute superlink boundary condition coefficient 'chi' for downstream end
        of superlink k.
        """
        t_0 = (1 - X_1k * gamma_uk) * (V_Nk - W_Nk * z_inv_uk - U_Nk * z_inv_dk)
        t_1 = (W_Nk * gamma_uk) * (Y_1k - X_1k * z_inv_uk - Z_1k * z_inv_dk)
        num = t_0 + t_1
        den = D_k_star
        return num, den

    def F_jj(self, A_sj, dt, beta_dkl, alpha_ukm):
        """
        Compute diagonal elements of sparse solution matrix A.
        """
        t_0 = A_sj / dt
        t_1 = beta_dkl
        t_2 = alpha_ukm
        return t_0 - t_1 + t_2

    def G_j(self, A_sj, dt, H_j, Q_0j, chi_ukl, chi_dkm):
        """
        Compute solution vector b.
        """
        t_0 = A_sj * H_j / dt
        t_1 = Q_0j
        t_2 = chi_ukl
        t_3 = chi_dkm
        # chi signs are switched in original paper
        return t_0 + t_1 - t_2 + t_3

    def B_j(self, J_uo, J_do, Ao, H_j, Co=0.67, g=9.81):
        dH_u = H_j[J_uo] - H_j[J_do]
        dH_d = H_j[J_do] - H_j[J_uo]
        # TODO: Why are these reversed?
        Qo_d = Co * Ao * np.sign(dH_u) * np.sqrt(2 * g * np.abs(dH_u))
        Qo_u = Co * Ao * np.sign(dH_d) * np.sqrt(2 * g * np.abs(dH_d))
        return Qo_u, Qo_d

    @safe_divide
    def theta_o(self, u, Qo_d_t, Ao, Co=0.67, g=9.81):
        num = 2 * g * u**2 * Co**2 * Ao**2
        den = np.abs(Qo_d_t)
        return num, den

    def configure_hydraulic_geometry(self):
        # Import instance variables
        transects = self.transects
        _shape_ik = self._shape_ik
        _transect_ik = self._transect_ik
        # Set attributes
        _geom_factory = {}
        _transect_factory = {}
        _transect_indices = None
        # Handle regular geometries
        _is_irregular = _shape_ik.str.lower() == 'irregular'
        _has_irregular = _is_irregular.any()
        _unique_geom = set(_shape_ik.str.lower().unique())
        _unique_geom.discard('irregular')
        _regular_shapes = _shape_ik[~_is_irregular]
        _geom_indices = pd.Series(_regular_shapes.index,
                                  index=_regular_shapes.str.lower().values)
        for geom in _unique_geom:
            _ik_g = _geom_indices.loc[[geom]].values
            _geom_factory[geom] = _ik_g
        # Handle irregular geometries
        if _has_irregular:
            _irregular_transects = _transect_ik[_is_irregular]
            _transect_indices = pd.Series(_irregular_transects.index,
                                          index=_irregular_transects.values)
            for transect_name, transect in transects.items():
                _transect_factory[transect_name] = superlink.geometry.Irregular(**transect)
        self._has_irregular = _has_irregular
        self._geom_factory = _geom_factory
        self._transect_factory = _transect_factory
        self._transect_indices = _transect_indices

    def configure_storages(self):
        # Import instance variables
        storages = self.storages
        _storage_type = self._storage_type
        _storage_table = self._storage_table
        _storage_factory = {}
        _storage_indices = None
        # Separate storages into functional and tabular
        _functional = (_storage_type.str.lower() == 'functional').values
        _tabular = (_storage_type.str.lower() == 'tabular').values
        # All entries must either be function or tabular
        assert (_tabular.sum() + _functional.sum()) == _storage_type.shape[0]
        # Configure tabular storages
        if storages:
            _tabular_storages = _storage_table[_tabular]
            _storage_indices = pd.Series(_tabular_storages.index, _tabular_storages.values)
            for table_name, table in storages.items():
                if table_name in _storage_indices:
                    _storage_factory[table_name] = superlink.storage.Tabular(**table)
        # Export instance variables
        self._storage_indices = _storage_indices
        self._storage_factory = _storage_factory
        self._functional = _functional
        self._tabular = _tabular

    def link_hydraulic_geometry(self):
        # TODO: Should probably use forward_I_i instead of _ik directly
        _ik = self._ik
        _Ik = self._Ik
        _Ip1k = self._Ip1k
        _h_Ik = self._h_Ik
        _A_ik = self._A_ik
        _Pe_ik = self._Pe_ik
        _R_ik = self._R_ik
        _B_ik = self._B_ik
        _dx_ik = self._dx_ik
        _g1_ik = self._g1_ik
        _g2_ik = self._g2_ik
        _g3_ik = self._g3_ik
        _geom_factory = self._geom_factory
        _transect_factory = self._transect_factory
        _transect_indices = self._transect_indices
        _has_irregular = self._has_irregular
        # Compute hydraulic geometry for regular geometries
        for geom, indices in _geom_factory.items():
            Geom = geom.title()
            _ik_g = indices
            _Ik_g = _Ik[_ik_g]
            _Ip1k_g = _Ip1k[_ik_g]
            generator = getattr(superlink.geometry, Geom)
            _g1_g = _g1_ik[_ik_g]
            _g2_g = _g2_ik[_ik_g]
            _g3_g = _g3_ik[_ik_g]
            _h_Ik_g = _h_Ik[_Ik_g]
            _h_Ip1k_g = _h_Ik[_Ip1k_g]
            _A_ik[_ik_g] = generator.A_ik(_h_Ik_g, _h_Ip1k_g,
                                          g1=_g1_g, g2=_g2_g, g3=_g3_g)
            _Pe_ik[_ik_g] = generator.Pe_ik(_h_Ik_g, _h_Ip1k_g,
                                            g1=_g1_g, g2=_g2_g, g3=_g3_g)
            _R_ik[_ik_g] = generator.R_ik(_A_ik[_ik_g], _Pe_ik[_ik_g])
            _B_ik[_ik_g] = generator.B_ik(_h_Ik_g, _h_Ip1k_g,
                                          g1=_g1_g, g2=_g2_g, g3=_g3_g)
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

    def compute_storage_areas(self):
        _functional = self._functional
        _tabular = self._tabular
        _storage_factory = self._storage_factory
        _storage_indices = self._storage_indices
        _storage_a = self._storage_a
        _storage_b = self._storage_b
        _storage_c = self._storage_c
        H_j = self.H_j
        _z_inv_j = self._z_inv_j
        min_depth = self.min_depth
        _A_sj = self._A_sj
        # Compute storage areas
        _h_j = np.maximum(H_j - _z_inv_j, min_depth)
        if _functional.any():
            generator = getattr(superlink.storage, 'Functional')
            _A_sj[_functional] = generator.A_sj(_h_j[_functional],
                                                _storage_a[_functional],
                                                _storage_b[_functional],
                                                _storage_c[_functional])
        if _tabular.any():
            for storage_name, generator in _storage_factory.items():
                _j_g = _storage_indices.loc[[storage_name]].values
                _A_sj[_j_g] = generator.A_sj(_h_j[_j_g])
        # Export instance variables
        self._A_sj = _A_sj

    def node_velocities(self):
        """
        Compute link hydraulic geometries and velocities.
        """
        # Import instance variables for better readability
        # TODO: Should probably use forward_I_i instead of _ik directly
        _ik = self._ik
        _Ik = self._Ik
        _Ip1k = self._Ip1k
        _h_Ik = self._h_Ik
        _A_ik = self._A_ik
        _Pe_ik = self._Pe_ik
        _R_ik = self._R_ik
        _B_ik = self._B_ik
        _Q_ik = self._Q_ik
        _u_Ik = self._u_Ik
        _u_Ip1k = self._u_Ip1k
        backward_I_i = self.backward_I_i
        forward_I_i = self.forward_I_i
        _dx_ik = self._dx_ik
        # TODO: Watch this
        _is_start_Ik = self._is_start[_Ik]
        _is_end_Ip1k = self._is_end[_Ip1k]
        # Compute link velocities
        _u_ik = self.u_ik(_Q_ik, _A_ik)
        # Compute velocities for start nodes (1 -> Nk)
        _u_Ik[_is_start_Ik] = _u_ik[_is_start_Ik]
        backward = backward_I_i[_Ik[~_is_start_Ik]]
        center = _ik[~_is_start_Ik]
        _u_Ik[~_is_start_Ik] = self.u_Ik(_dx_ik[center], _u_ik[backward],
                                         _dx_ik[backward], _u_ik[center])
        # Compute velocities for end nodes (2 -> Nk+1)
        _u_Ip1k[_is_end_Ip1k] = _u_ik[_is_end_Ip1k]
        forward = forward_I_i[_Ip1k[~_is_end_Ip1k]]
        center = _ik[~_is_end_Ip1k]
        _u_Ip1k[~_is_end_Ip1k] = self.u_Ip1k(_dx_ik[center], _u_ik[forward],
                                             _dx_ik[forward], _u_ik[center])
        # Export to instance variables
        self._u_ik = _u_ik
        self._u_Ik = _u_Ik
        self._u_Ip1k = _u_Ip1k

    def compute_flow_regime(self):
        # Import instance variables
        _u_ik = self._u_ik
        _A_ik = self._A_ik
        _B_ik = self._B_ik
        _ki = self._ki
        _kI = self._kI
        NK = self.NK
        nk = self.nk
        # Compute Froude number for each superlink and link
        _Fr_k = np.zeros(NK)
        _Fr_ik = self.Fr(_u_ik, _A_ik, _B_ik)
        np.add.at(_Fr_k, _ki, _Fr_ik)
        _Fr_k /= nk
        # Determine if superlink is supercritical
        _supercritical = (_Fr_k >= 1)[_kI]
        # Compute sigma for inertial damping
        _sigma_ik = 2 * (1 - _Fr_ik)
        _sigma_ik[_Fr_ik < 0.5] = 1.
        _sigma_ik[_Fr_ik > 1] = 0.
        # Export instance variables
        self._Fr_k = _Fr_k
        self._Fr_ik = _Fr_ik
        self._supercritical = _supercritical
        self._sigma_ik = _sigma_ik

    def link_coeffs(self, _dt=None):
        # Import instance variables
        _u_Ik = self._u_Ik
        _u_Ip1k = self._u_Ip1k
        _dx_ik = self._dx_ik
        _n_ik = self._n_ik
        _Q_ik = self._Q_ik
        _A_ik = self._A_ik
        _R_ik = self._R_ik
        _S_o_ik = self._S_o_ik
        _A_c_ik = self._A_c_ik
        _C_ik = self._C_ik
        _ctrl = self._ctrl
        inertial_damping = self.inertial_damping
        if inertial_damping:
            _sigma_ik = self._sigma_ik
        else:
            _sigma_ik = 1
        if _dt is None:
            _dt = self._dt
        # Compute link coefficients
        _a_ik = self.a_ik(_u_Ik, _sigma_ik)
        _c_ik = self.c_ik(_u_Ip1k, _sigma_ik)
        _b_ik = self.b_ik(_dx_ik, _dt, _n_ik, _Q_ik, _A_ik, _R_ik,
                          _A_c_ik, _C_ik, _a_ik, _c_ik, _ctrl)
        _P_ik = self.P_ik(_Q_ik, _dx_ik, _dt, _A_ik, _S_o_ik,
                          _sigma_ik)
        # Export to instance variables
        self._a_ik = _a_ik
        self._b_ik = _b_ik
        self._c_ik = _c_ik
        self._P_ik = _P_ik

    def node_coeffs(self, _Q_0Ik=None, _dt=None):
        # Import instance variables
        _I = self._I
        start_nodes = self.start_nodes
        end_nodes = self.end_nodes
        middle_nodes = self.middle_nodes
        forward_I_i = self.forward_I_i
        backward_I_i = self.backward_I_i
        _B_ik = self._B_ik
        _dx_ik = self._dx_ik
        _A_SIk = self._A_SIk
        _h_Ik = self._h_Ik
        _E_Ik = self._E_Ik
        _D_Ik = self._D_Ik
        if _dt is None:
            _dt = self._dt
        if _Q_0Ik is None:
            _Q_0Ik = np.zeros(_I.size)
        # Compute E_Ik and D_Ik
        start_links = self.forward_I_i[start_nodes]
        end_links = self.backward_I_i[end_nodes]
        backward = self.backward_I_i[middle_nodes]
        forward = self.forward_I_i[middle_nodes]
        _E_Ik[start_nodes] = self.E_Ik(_B_ik[start_links], _dx_ik[start_links],
                                        _B_ik[start_links], _dx_ik[start_links],
                                        _A_SIk[start_nodes], _dt)
        _E_Ik[end_nodes] = self.E_Ik(_B_ik[end_links], _dx_ik[end_links],
                                        _B_ik[end_links], _dx_ik[end_links],
                                        _A_SIk[end_nodes], _dt)
        _E_Ik[middle_nodes] = self.E_Ik(_B_ik[forward], _dx_ik[forward],
                                        _B_ik[backward], _dx_ik[backward],
                                        _A_SIk[middle_nodes], _dt)
        _D_Ik[start_nodes] = self.D_Ik(_Q_0Ik[start_nodes], _B_ik[start_links],
                                        _dx_ik[start_links], _B_ik[start_links],
                                        _dx_ik[start_links], _A_SIk[start_nodes],
                                        _h_Ik[start_nodes], _dt)
        _D_Ik[end_nodes] = self.D_Ik(_Q_0Ik[end_nodes], _B_ik[end_links],
                                        _dx_ik[end_links], _B_ik[end_links],
                                        _dx_ik[end_links], _A_SIk[end_nodes],
                                        _h_Ik[end_nodes], _dt)
        _D_Ik[middle_nodes] = self.D_Ik(_Q_0Ik[middle_nodes], _B_ik[forward],
                                        _dx_ik[forward], _B_ik[backward],
                                        _dx_ik[backward], _A_SIk[middle_nodes],
                                        _h_Ik[middle_nodes], _dt)
        # Export instance variables
        self._E_Ik = _E_Ik
        self._D_Ik = _D_Ik

    def forward_recurrence(self):
        # Import instance variables
        backward_I_I = self.backward_I_I
        forward_I_I = self.forward_I_I
        forward_I_i = self.forward_I_i
        _I_end = self._I_end
        _I_1k = self._I_1k
        _I_2k = self._I_2k
        _i_1k = self._i_1k
        _A_ik = self._A_ik
        _E_Ik = self._E_Ik
        _D_Ik = self._D_Ik
        _a_ik = self._a_ik
        _b_ik = self._b_ik
        _c_ik = self._c_ik
        _P_ik = self._P_ik
        _T_ik = self._T_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        # Compute coefficients for starting nodes
        _E_2k = _E_Ik[_I_2k]
        _D_2k = _D_Ik[_I_2k]
        _T_1k = self.T_1k(_a_ik[_i_1k], _b_ik[_i_1k], _c_ik[_i_1k])
        _U_1k = self.U_1k(_E_2k, _c_ik[_i_1k], _A_ik[_i_1k], _T_1k)
        _V_1k = self.V_1k(_P_ik[_i_1k], _D_2k, _c_ik[_i_1k], _T_1k)
        _W_1k = self.W_1k(_A_ik[_i_1k], _T_1k)
        # I = 1, i = 1
        _T_ik[_i_1k] = _T_1k
        _U_Ik[_I_1k] = _U_1k
        _V_Ik[_I_1k] = _V_1k
        _W_Ik[_I_1k] = _W_1k
        # I = 2, i = 2
        _I_next = _I_2k[~_I_end[_I_2k]]
        # Loop from 2 -> Nk
        # print('Forward recurrence')
        while _I_next.size:
            _Im1_next = backward_I_I[_I_next]
            _Ip1_next = forward_I_I[_I_next]
            _i_next = forward_I_i[_I_next]
            # print('I = ', _I_next, 'i = ', _i_next, 'I-1 = ', _Im1_next, 'I+1 = ', _Ip1_next)
            _T_ik[_i_next] = self.T_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                       _A_ik[_i_next], _E_Ik[_I_next], _U_Ik[_Im1_next])
            _U_Ik[_I_next] = self.U_Ik(_E_Ik[_Ip1_next], _c_ik[_i_next],
                                       _A_ik[_i_next], _T_ik[_i_next])
            _V_Ik[_I_next] = self.V_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                       _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                       _E_Ik[_I_next], _V_Ik[_Im1_next], _U_Ik[_Im1_next],
                                       _T_ik[_i_next])
            _W_Ik[_I_next] = self.W_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                       _W_Ik[_Im1_next], _U_Ik[_Im1_next], _T_ik[_i_next])
            _I_next = _Ip1_next[~_I_end[_Ip1_next]]
        # Try resetting
        _T_ik[_i_1k] = _T_1k
        _U_Ik[_I_1k] = _U_1k
        _V_Ik[_I_1k] = _V_1k
        _W_Ik[_I_1k] = _W_1k
        # Export instance variables
        self._T_ik = _T_ik
        self._U_Ik = _U_Ik
        self._V_Ik = _V_Ik
        self._W_Ik = _W_Ik

    def backward_recurrence(self):
        # Import instance variables
        backward_I_I = self.backward_I_I
        forward_I_I = self.forward_I_I
        forward_I_i = self.forward_I_i
        _I_start = self._I_start
        _I_Nk = self._I_Nk
        _i_nk = self._i_nk
        _A_ik = self._A_ik
        _E_Ik = self._E_Ik
        _D_Ik = self._D_Ik
        _a_ik = self._a_ik
        _b_ik = self._b_ik
        _c_ik = self._c_ik
        _P_ik = self._P_ik
        _O_ik = self._O_ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        # Compute coefficients for starting nodes
        _E_Nk = _E_Ik[_I_Nk]
        _D_Nk = _D_Ik[_I_Nk]
        _O_nk = self.O_nk(_a_ik[_i_nk], _b_ik[_i_nk], _c_ik[_i_nk])
        _X_Nk = self.X_Nk(_A_ik[_i_nk], _E_Nk, _a_ik[_i_nk], _O_nk)
        _Y_Nk = self.Y_Nk(_P_ik[_i_nk], _D_Nk, _a_ik[_i_nk], _O_nk)
        _Z_Nk = self.Z_Nk(_A_ik[_i_nk], _O_nk)
        # I = Nk, i = nk
        _O_ik[_i_nk] = _O_nk
        _X_Ik[_I_Nk] = _X_Nk
        _Y_Ik[_I_Nk] = _Y_Nk
        _Z_Ik[_I_Nk] = _Z_Nk
        # I = Nk-1, i = nk-1
        _I_next = backward_I_I[_I_Nk[~_I_start[_I_Nk]]]
        # Loop from Nk - 1 -> 1
        # print('Backward recurrence')
        while _I_next.size:
            _Ip1_next = forward_I_I[_I_next]
            _i_next = forward_I_i[_I_next]
            # print('I = ', _I_next, 'i = ', _i_next, 'I+1 = ', _Ip1_next)
            _O_ik[_i_next] = self.O_ik(_a_ik[_i_next], _b_ik[_i_next], _c_ik[_i_next],
                                _A_ik[_i_next], _E_Ik[_Ip1_next], _X_Ik[_Ip1_next])
            _X_Ik[_I_next] = self.X_Ik(_A_ik[_i_next], _E_Ik[_I_next], _a_ik[_i_next],
                                       _O_ik[_i_next])
            _Y_Ik[_I_next] = self.Y_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                       _D_Ik[_Ip1_next], _c_ik[_i_next], _A_ik[_i_next],
                                       _E_Ik[_Ip1_next], _Y_Ik[_Ip1_next], _X_Ik[_Ip1_next],
                                       _O_ik[_i_next])
            _Z_Ik[_I_next] = self.Z_Ik(_A_ik[_i_next], _E_Ik[_Ip1_next], _c_ik[_i_next],
                                       _Z_Ik[_Ip1_next], _X_Ik[_Ip1_next], _O_ik[_i_next])
            _I_next = backward_I_I[_I_next[~_I_start[_I_next]]]
        # Try resetting
        _O_ik[_i_nk] = _O_nk
        _X_Ik[_I_Nk] = _X_Nk
        _Y_Ik[_I_Nk] = _Y_Nk
        _Z_Ik[_I_Nk] = _Z_Nk
        # Export instance variables
        self._O_ik = _O_ik
        self._X_Ik = _X_Ik
        self._Y_Ik = _Y_Ik
        self._Z_Ik = _Z_Ik

    def superlink_upstream_head_coefficients(self, full_solution=False):
        # Import instance variables
        _I_1k = self._I_1k
        _i_1k = self._i_1k
        _h_Ik = self._h_Ik
        _J_uk = self._J_uk
        _z_inv_uk = self._z_inv_uk
        _A_ik = self._A_ik
        _Q_ik = self._Q_ik
        # Placeholder discharge coefficient
        _C_uk = 0.67
        # Current upstream flows
        _Q_uk_t = _Q_ik[_i_1k]
        # Upstream area
        _A_uk = _A_ik[_i_1k]
        # Compute superlink upstream coefficients
        _gamma_uk = self.gamma_uk(_Q_uk_t, _C_uk, _A_uk)
        self._gamma_uk = _gamma_uk

    def superlink_downstream_head_coefficients(self):
        # Import instance variables
        _I_Np1k = self._I_Np1k
        _i_nk = self._i_nk
        _h_Ik = self._h_Ik
        _J_dk = self._J_dk
        _z_inv_dk = self._z_inv_dk
        _A_ik = self._A_ik
        _Q_ik = self._Q_ik
        # Placeholder discharge coefficient
        _C_dk = 0.67
        # Current downstream flows
        _Q_dk_t = _Q_ik[_i_nk]
        # Downstream area
        _A_dk = _A_ik[_i_nk]
        # Compute superlink downstream coefficients
        _gamma_dk = self.gamma_dk(_Q_dk_t, _C_dk, _A_dk)
        self._gamma_dk = _gamma_dk

    def superlink_flow_coefficients(self):
        # Import instance variables
        _I_1k = self._I_1k
        _I_Nk = self._I_Nk
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _gamma_uk = self._gamma_uk
        _gamma_dk = self._gamma_dk
        _z_inv_uk = self._z_inv_uk
        _z_inv_dk = self._z_inv_dk
        # Compute D_k_star
        _D_k_star = self.D_k_star(_X_Ik[_I_1k], _gamma_uk, _U_Ik[_I_Nk],
                                  _gamma_dk, _Z_Ik[_I_1k], _W_Ik[_I_Nk])
        # Compute upstream superlink flow coefficients
        _alpha_uk = self.alpha_uk(_U_Ik[_I_Nk], _gamma_dk, _X_Ik[_I_1k],
                                  _Z_Ik[_I_1k], _W_Ik[_I_Nk], _D_k_star)
        _beta_uk = self.beta_uk(_U_Ik[_I_Nk], _gamma_dk, _Z_Ik[_I_1k],
                                _W_Ik[_I_Nk], _D_k_star)
        _chi_uk = self.chi_uk(_U_Ik[_I_Nk], _gamma_dk, _Y_Ik[_I_1k],
                              _X_Ik[_I_1k], _z_inv_uk, _Z_Ik[_I_1k],
                              _z_inv_dk, _V_Ik[_I_Nk], _W_Ik[_I_Nk],
                              _D_k_star)
        # Compute downstream superlink flow coefficients
        _alpha_dk = self.alpha_dk(_X_Ik[_I_1k], _gamma_uk, _W_Ik[_I_Nk],
                                  _D_k_star)
        _beta_dk = self.beta_dk(_X_Ik[_I_1k], _gamma_uk, _U_Ik[_I_Nk],
                                _W_Ik[_I_Nk], _Z_Ik[_I_1k], _D_k_star)
        _chi_dk = self.chi_dk(_X_Ik[_I_1k], _gamma_uk, _V_Ik[_I_Nk],
                              _W_Ik[_I_Nk], _z_inv_uk, _U_Ik[_I_Nk],
                              _z_inv_dk, _Y_Ik[_I_1k], _Z_Ik[_I_1k],
                              _D_k_star)
        # Export instance variables
        self._D_k_star = _D_k_star
        self._alpha_uk = _alpha_uk
        self._beta_uk = _beta_uk
        self._chi_uk = _chi_uk
        self._alpha_dk = _alpha_dk
        self._beta_dk = _beta_dk
        self._chi_dk = _chi_dk

    def sparse_matrix_equations(self, H_bc=None, _Q_0j=None, u=None, _dt=None, implicit=True,
                                first_time=False):
        # TODO: May want to consider reconstructing A each time while debugging
        # Import instance variables
        _k = self._k
        _J_uk = self._J_uk
        _J_dk = self._J_dk
        _alpha_uk = self._alpha_uk
        _alpha_dk = self._alpha_dk
        _beta_uk = self._beta_uk
        _beta_dk = self._beta_dk
        _chi_uk = self._chi_uk
        _chi_dk = self._chi_dk
        _alpha_ukm = self._alpha_ukm
        _beta_dkl = self._beta_dkl
        _chi_ukl = self._chi_ukl
        _chi_dkm = self._chi_dkm
        _A_sj = self._A_sj
        _J_uo = self._J_uo
        _J_do = self._J_do
        _Ao = self._Ao
        _sparse = self._sparse
        _O_diag = self._O_diag
        M = self.M
        H_j = self.H_j
        bc = self.bc
        _Qo_t = self._Qo_t
        if _dt is None:
            _dt = self._dt
        if H_bc is None:
            H_bc = self.H_j
        if _Q_0j is None:
            _Q_0j = 0
        if u is None:
            u = 0
        # Compute F_jj
        _alpha_ukm.fill(0)
        _beta_dkl.fill(0)
        np.add.at(_alpha_ukm, _J_uk, _alpha_uk)
        np.add.at(_beta_dkl, _J_dk, _beta_dk)
        _F_jj = self.F_jj(_A_sj, _dt, _beta_dkl, _alpha_ukm)
        # Set diagonals
        i = np.arange(M)
        self.A[i[~bc], i[~bc]] = _F_jj[i[~bc]]
        self.A[i[bc], i[bc]] = 1
        # Compute off-diagonals
        bc_uk = bc[_J_uk]
        bc_dk = bc[_J_dk]
        self.A[_J_uk[~bc_uk], _J_dk[~bc_uk]] = _beta_uk[~bc_uk]
        self.A[_J_dk[~bc_dk], _J_uk[~bc_dk]] = -_alpha_dk[~bc_dk]
        # Compute G_j
        _chi_ukl.fill(0)
        _chi_dkm.fill(0)
        np.add.at(_chi_ukl, _J_uk, _chi_uk)
        np.add.at(_chi_dkm, _J_dk, _chi_dk)
        b = self.G_j(_A_sj, _dt, H_j, _Q_0j, _chi_ukl, _chi_dkm)
        b[bc] = H_bc[bc]
        # Compute control matrix
        if _J_uo.size:
            bc_uo = bc[_J_uo]
            bc_do = bc[_J_do]
            if implicit:
                # TODO: This will overwrite?
                # TODO: Ao should be dependent on height of water
                _theta_o = self.theta_o(u, _Qo_t, _Ao)
                _O_diag.fill(0)
                np.add.at(_O_diag, _J_uo[~bc_uo], _theta_o[~bc_uo])
                np.add.at(_O_diag, _J_do[~bc_do], _theta_o[~bc_do])
                np.fill_diagonal(self.O, _O_diag)
                self.O[_J_uo[~bc_uo], _J_do[~bc_uo]] = -_theta_o[~bc_uo]
                self.O[_J_do[~bc_do], _J_uo[~bc_do]] = -_theta_o[~bc_do]
            else:
                _Qo_u, _Qo_d = self.B_j(_J_uo, _J_do, _Ao, H_j)
                self.B[_J_uo[~bc_uo]] = _Qo_u[~bc_uo]
                self.B[_J_do[~bc_do]] = _Qo_d[~bc_do]
        # Export instance variables
        self.b = b
        self._beta_dkl = _beta_dkl
        self._alpha_ukm = _alpha_ukm
        self._chi_ukl = _chi_ukl
        self._chi_dkm = _chi_dkm
        if first_time and _sparse:
            self.A = self.A.tocsr()

    def solve_sparse_matrix(self, u=None, implicit=True):
        # Import instance variables
        A = self.A
        b = self.b
        B = self.B
        O = self.O
        _z_inv_j = self._z_inv_j
        _sparse = self._sparse
        min_depth = self.min_depth
        max_depth = self.max_depth
        # Get right-hand size
        if u is not None:
            if implicit:
                l = A + O
                r = b
            else:
                l = A
                r = b + np.squeeze(B @ u)
        else:
            l = A
            r = b
        if _sparse:
            H_j_next = scipy.sparse.linalg.spsolve(l, r)
        else:
            H_j_next = scipy.linalg.solve(l, r)
        H_j_next = np.maximum(H_j_next, _z_inv_j + min_depth)
        H_j_next = np.minimum(H_j_next, _z_inv_j + max_depth)
        self.H_j = H_j_next

    def solve_superlink_flows(self):
        # Import instance variables
        _J_uk = self._J_uk
        _J_dk = self._J_dk
        _alpha_uk = self._alpha_uk
        _alpha_dk = self._alpha_dk
        _beta_uk = self._beta_uk
        _beta_dk = self._beta_dk
        _chi_uk = self._chi_uk
        _chi_dk = self._chi_dk
        H_j = self.H_j
        # Compute flow at next time step
        _Q_uk_next = _alpha_uk * H_j[_J_uk] + _beta_uk * H_j[_J_dk] + _chi_uk
        _Q_dk_next = _alpha_dk * H_j[_J_uk] + _beta_dk * H_j[_J_dk] + _chi_dk
        # Export instance variables
        self._Q_uk = _Q_uk_next
        self._Q_dk = _Q_dk_next

    def solve_orifice_flows(self, u=None):
        # Import instance variables
        _J_uo = self._J_uo
        _J_do = self._J_do
        _Ao = self._Ao
        H_j = self.H_j
        if u is None:
            u = 0
        # Compute orifice flows
        _, _Qo_t = self.B_j(_J_uo, _J_do, u * _Ao, H_j)
        # Export instance variables
        self._Qo_t = _Qo_t

    def solve_superlink_depths(self):
        # Import instance variables
        _J_uk = self._J_uk
        _J_dk = self._J_dk
        _gamma_uk = self._gamma_uk
        _gamma_dk = self._gamma_dk
        _z_inv_uk = self._z_inv_uk
        _z_inv_dk = self._z_inv_dk
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        H_j = self.H_j
        # Compute flow at next time step
        _h_uk_next = _gamma_uk * _Q_uk + H_j[_J_uk] - _z_inv_uk
        _h_dk_next = _gamma_dk * _Q_dk + H_j[_J_dk] - _z_inv_dk
        # Export instance variables
        self._h_uk = _h_uk_next
        self._h_dk = _h_dk_next

    def solve_superlink_depths_alt(self):
        # Import instance variables
        _I_1k = self._I_1k
        _I_Nk = self._I_Nk
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        H_j = self.H_j
        min_depth = self.min_depth
        # Compute flow at next time step
        det = (_X_Ik[_I_1k] * _U_Ik[_I_Nk]) - (_Z_Ik[_I_1k] * _W_Ik[_I_Nk])
        det[det == 0] = np.inf
        _h_uk_next = (_U_Ik[_I_Nk] * (_Q_uk - _Y_Ik[_I_1k])
                      - _Z_Ik[_I_1k] * (_Q_dk - _V_Ik[_I_Nk])) / det
        _h_dk_next = (-_W_Ik[_I_Nk] * (_Q_uk - _Y_Ik[_I_1k])
                      + _X_Ik[_I_1k] * (_Q_dk - _V_Ik[_I_Nk])) / det
        # Set minimum values
        _h_uk_next[_h_uk_next < min_depth] = min_depth
        _h_dk_next[_h_dk_next < min_depth] = min_depth
        # Export instance variables
        self._h_uk = _h_uk_next
        self._h_dk = _h_dk_next

    def solve_internals_lsq(self):
        # Import instance variables
        _I_1k = self._I_1k
        _I_Np1k = self._I_Np1k
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _kI = self._kI
        middle_nodes = self.middle_nodes
        middle_links = self.middle_links
        _G = self._G
        _h = self._h
        _I_f = self._I_f
        _I_b = self._I_b
        _b_f = self._b_f
        _b_b = self._b_b
        _rows_f = self._rows_f
        _rows_b = self._rows_b
        _rows_r = self._rows_r
        _cols_l = self._cols_l
        _cols_r = self._cols_r
        _lbound = self._lbound
        _ubound = self._ubound
        _ix_h = self._ix_h
        _ix_q = self._ix_q
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        min_depth = self.min_depth
        # Update solution matrix
        _G[_rows_f, _cols_l] = _U_Ik[_I_f]
        _G[_rows_b, _cols_l] = _X_Ik[_I_b]
        _G[_rows_r, _cols_r] = -1
        _h[::2] = -_V_Ik[_I_f] - _W_Ik[_I_f] * _h_uk[_kI[_I_f]]
        _h[1::2] = -_Y_Ik[_I_b] - _Z_Ik[_I_b] * _h_dk[_kI[_I_b]]
        _h[_b_f] += _Q_uk
        _h[_b_b] += _Q_dk
        # Solve constrained least squares
        result = scipy.optimize.lsq_linear(_G, _h, bounds=(_lbound, _ubound))
        x = result.x
        _h_Ik[middle_nodes] = x[_ix_h]
        _Q_ik[middle_links] = x[_ix_q]
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_uk
        _h_Ik[_h_Ik < min_depth] = min_depth
        # Set instance variables
        self.result = result
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    def solve_internals(self):
        self.solve_internals_forwards()
        self.solve_internals_backwards()
        _h_Ik_b = self._h_Ik_b
        _h_Ik_f = self._h_Ik_f
        _Q_ik_b = self._Q_ik_b
        _Q_ik_f = self._Q_ik_f
        # _Q_ik = (_Q_ik_b + _Q_ik_f) / 2
        # _h_Ik = (_h_Ik_b + _h_Ik_f) / 2
        _Q_ik = _Q_ik_b
        _h_Ik = _h_Ik_b
        self._Q_ik = _Q_ik
        self._h_Ik = _h_Ik

    def solve_internals_simultaneous(self):
        _I_1k = self._I_1k
        _I_2k = self._I_2k
        _I_Nk = self._I_Nk
        _I_Np1k = self._I_Np1k
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _I_start = self._I_start
        _I_end = self._I_end
        forward_I_I = self.forward_I_I
        backward_I_I = self.backward_I_I
        forward_I_i = self.forward_I_i
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        min_depth = self.min_depth
        # Set first elements
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Get rid of superlinks with one link (forwards)
        keep_f = (_I_2k != _I_Np1k)
        _Im1_next_f = _I_1k[keep_f]
        _I_next_f = _I_2k[keep_f]
        _I_1k_next_f = _I_1k[keep_f]
        _I_Nk_next_f = _I_Nk[keep_f]
        _I_Np1k_next_f = _I_Np1k[keep_f]
        # Get ride of superlinks with one link (backwards)
        keep_b = (_I_1k != _I_Nk)
        _Im1_next_b = _I_1k[keep_b]
        _I_next_b = _I_Nk[keep_b]
        _I_1k_next_b = _I_1k[keep_b]
        _I_2k_next_b = _I_2k[keep_b]
        _I_Np1k_next_b = _I_Np1k[keep_b]
        while (_I_next_f.size > 0) & (_I_next_b.size > 0):
            _i_next_f = forward_I_i[_I_next_f]
            _im1_next_f = forward_I_i[_Im1_next_f]
            _Ip1_next_f = forward_I_I[_I_next_f]
            _i_next_b = forward_I_i[_I_next_b]
            _Im1_next_b = backward_I_I[_I_next_b]
            _im1_next_b = forward_I_i[_Im1_next_b]
            # print('I_f =', _I_next_f, 'I_b =', _I_next_b)
            _h_Ik[_I_next_b] = self._h_Ik_next_b(_Q_ik[_i_next_b], _Y_Ik[_I_next_b],
                                                 _Z_Ik[_I_next_b], _h_Ik[_I_Np1k_next_b],
                                                 _X_Ik[_I_next_b])
            _h_Ik[_I_next_b[_h_Ik[_I_next_b] < min_depth]] = min_depth
            _Q_ik[_im1_next_b] = self._Q_im1k_next_b(_U_Ik[_Im1_next_b], _h_Ik[_I_next_b],
                                                       _V_Ik[_Im1_next_b], _W_Ik[_Im1_next_b],
                                                       _h_Ik[_I_1k_next_b])
            _h_Ik[_I_next_b] = (_h_Ik[_I_next_b] + self._h_Ik_next_f(_Q_ik[_im1_next_b],
                                                                     _V_Ik[_Im1_next_b],
                                                                     _W_Ik[_Im1_next_b],
                                                                     _h_Ik[_I_1k_next_b],
                                                                     _U_Ik[_Im1_next_b])) / 2
            _h_Ik[_I_next_b[_h_Ik[_I_next_b] < min_depth]] = min_depth
            _Q_ik[_im1_next_b] = (_Q_ik[_im1_next_b] + self._Q_im1k_next_b(_U_Ik[_Im1_next_b],
                                                                           _h_Ik[_I_next_b],
                                                                           _V_Ik[_Im1_next_b],
                                                                           _W_Ik[_Im1_next_b],
                                                                           _h_Ik[_I_1k_next_b])) / 2
            _h_Ik[_I_next_f] = self._h_Ik_next_f(_Q_ik[_im1_next_f], _V_Ik[_Im1_next_f],
                                                 _W_Ik[_Im1_next_f], _h_Ik[_I_1k_next_f],
                                                 _U_Ik[_Im1_next_f])
            _h_Ik[_I_next_f[_h_Ik[_I_next_f] < min_depth]] = min_depth
            _Q_ik[_i_next_f] = self._Q_i_next_f(_X_Ik[_I_next_f], _h_Ik[_I_next_f],
                                                _Y_Ik[_I_next_f], _Z_Ik[_I_next_f],
                                                _h_Ik[_I_Np1k_next_f])
            _h_Ik[_I_next_f] = (_h_Ik[_I_next_f] + self._h_Ik_next_b(_Q_ik[_i_next_f],
                                                                     _Y_Ik[_I_next_f],
                                                                     _Z_Ik[_I_next_f],
                                                                     _h_Ik[_I_Np1k_next_f],
                                                                     _X_Ik[_I_next_f])) / 2
            _h_Ik[_I_next_f[_h_Ik[_I_next_f] < min_depth]] = min_depth
            _Q_ik[_i_next_f] = (_Q_ik[_i_next_f] + self._Q_i_next_f(_X_Ik[_I_next_f],
                                                                    _h_Ik[_I_next_f],
                                                                    _Y_Ik[_I_next_f],
                                                                    _Z_Ik[_I_next_f],
                                                                    _h_Ik[_I_Np1k_next_f])) / 2
            keep_f = (_Ip1_next_f != _I_Np1k_next_f)
            _Im1_next_f = _I_next_f[keep_f]
            _I_next_f = _Ip1_next_f[keep_f]
            _I_1k_next_f = _I_1k_next_f[keep_f]
            _I_Nk_next_f = _I_Nk_next_f[keep_f]
            _I_Np1k_next_f = _I_Np1k_next_f[keep_f]
            keep_b = (_Im1_next_b != _I_1k_next_b)
            _I_next_b = _Im1_next_b[keep_b]
            _I_1k_next_b = _I_1k_next_b[keep_b]
            _I_Np1k_next_b = _I_Np1k_next_b[keep_b]
        _h_Ik[_h_Ik < min_depth] = min_depth
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    def solve_internals_forwards(self, supercritical_only=False):
        # Import instance variables
        _I_1k = self._I_1k
        _I_2k = self._I_2k
        _I_Nk = self._I_Nk
        _I_Np1k = self._I_Np1k
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _I_end = self._I_end
        forward_I_I = self.forward_I_I
        forward_I_i = self.forward_I_i
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        min_depth = self.min_depth
        # max_depth = 0.3048 * 10
        if supercritical_only:
            _supercritical = self._supercritical
        # Set first elements
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Get rid of superlinks with one link
        keep = (_I_2k != _I_Np1k)
        _Im1_next = _I_1k[keep]
        _I_next = _I_2k[keep]
        _I_1k_next = _I_1k[keep]
        _I_Nk_next = _I_Nk[keep]
        _I_Np1k_next = _I_Np1k[keep]
        # If only using subcritical superlinks
        if supercritical_only:
            keep = _supercritical[_I_next]
            _Im1_next = _Im1_next[keep]
            _I_next = _I_next[keep]
            _I_1k_next = _I_1k_next[keep]
            _I_Nk_next = _I_Nk_next[keep]
            _I_Np1k_next = _I_Np1k_next[keep]
        # Loop from 2 -> Nk
        # print('Solve internals forwards')
        while _I_next.size:
            _i_next = forward_I_i[_I_next]
            _im1_next = forward_I_i[_Im1_next]
            _Ip1_next = forward_I_I[_I_next]
            # print('I = ', _I_next, 'i = ', _i_next, 'I-1 = ', _Im1_next, 'I+1 = ', _Ip1_next)
            _h_Ik[_I_next] = self._h_Ik_next_f(_Q_ik[_im1_next], _V_Ik[_Im1_next],
                                                 _W_Ik[_Im1_next], _h_Ik[_I_1k_next],
                                                 _U_Ik[_Im1_next])
            _h_Ik[_I_next[_h_Ik[_I_next] < min_depth]] = min_depth
            # _h_Ik[_I_next[_h_Ik[_I_next] > max_depth]] = min_depth
            _Q_ik[_i_next] = self._Q_i_next_f(_X_Ik[_I_next], _h_Ik[_I_next],
                                                _Y_Ik[_I_next], _Z_Ik[_I_next],
                                                _h_Ik[_I_Np1k_next])
            keep = (_Ip1_next != _I_Np1k_next)
            _Im1_next = _I_next[keep]
            # _im1_next = _i_next[keep]
            _I_next = _Ip1_next[keep]
            _I_1k_next = _I_1k_next[keep]
            _I_Nk_next = _I_Nk_next[keep]
            _I_Np1k_next = _I_Np1k_next[keep]
        # TODO: Reset first elements
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Ensure non-negative depths
        _h_Ik[_h_Ik < min_depth] = min_depth
        # _h_Ik[_h_Ik > max_depth] = max_depth
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    @safe_divide
    def _h_Ik_next_f(self, Q_ik, V_Ik, W_Ik, h_1k, U_Ik):
        num = Q_ik - V_Ik - W_Ik * h_1k
        den = U_Ik
        return num, den

    def _Q_i_next_f(self, X_Ik, h_Ik, Y_Ik, Z_Ik, h_Np1k):
        t_0 = X_Ik * h_Ik
        t_1 = Y_Ik
        t_2 = Z_Ik * h_Np1k
        return t_0 + t_1 + t_2

    def solve_internals_backwards(self, subcritical_only=False):
        # Import instance variables
        _I_1k = self._I_1k
        _I_2k = self._I_2k
        _I_Nk = self._I_Nk
        _I_Np1k = self._I_Np1k
        _i_1k = self._i_1k
        _i_nk = self._i_nk
        _I_end = self._I_end
        backward_I_I = self.backward_I_I
        forward_I_i = self.forward_I_i
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _Q_uk = self._Q_uk
        _Q_dk = self._Q_dk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        min_depth = self.min_depth
        # max_depth = 0.3048 * 10
        if subcritical_only:
            _subcritical = ~self._supercritical
        # Set first elements
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Get rid of superlinks with one link
        keep = (_I_1k != _I_Nk)
        _Im1_next = _I_1k[keep]
        _I_next = _I_Nk[keep]
        _I_1k_next = _I_1k[keep]
        _I_2k_next = _I_2k[keep]
        _I_Np1k_next = _I_Np1k[keep]
        # If only using subcritical superlinks
        if subcritical_only:
            keep = _subcritical[_I_next]
            _Im1_next = _Im1_next[keep]
            _I_next = _I_next[keep]
            _I_1k_next = _I_1k_next[keep]
            _I_2k_next = _I_2k_next[keep]
            _I_Np1k_next = _I_Np1k_next[keep]
        # Loop from Nk -> 1
        # print('Solve internals backwards')
        while _I_next.size:
            _i_next = forward_I_i[_I_next]
            _Im1_next = backward_I_I[_I_next]
            _im1_next = forward_I_i[_Im1_next]
            # print('I = ', _I_next, 'i = ', _i_next, 'I-1 = ', _Im1_next, 'i-1 = ', _im1_next)
            _h_Ik[_I_next] = self._h_Ik_next_b(_Q_ik[_i_next], _Y_Ik[_I_next],
                                                 _Z_Ik[_I_next], _h_Ik[_I_Np1k_next],
                                                 _X_Ik[_I_next])
            # Ensure non-negative depths?
            _h_Ik[_I_next[_h_Ik[_I_next] < min_depth]] = min_depth
            # _h_Ik[_I_next[_h_Ik[_I_next] > max_depth]] = min_depth
            _Q_ik[_im1_next] = self._Q_im1k_next_b(_U_Ik[_Im1_next], _h_Ik[_I_next],
                                                     _V_Ik[_Im1_next], _W_Ik[_Im1_next],
                                                     _h_Ik[_I_1k_next])
            keep = (_Im1_next != _I_1k_next)
            _I_next = _Im1_next[keep]
            _I_1k_next = _I_1k_next[keep]
            _I_Np1k_next = _I_Np1k_next[keep]
        # Set upstream flow
        # TODO: May want to delete where this is set earlier
        _Q_ik[_i_1k] = _Q_uk
        _Q_ik[_i_nk] = _Q_dk
        _h_Ik[_I_1k] = _h_uk
        _h_Ik[_I_Np1k] = _h_dk
        # Ensure non-negative depths?
        _h_Ik[_h_Ik < min_depth] = min_depth
        # _h_Ik[_h_Ik > max_depth] = max_depth
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    @safe_divide
    def _h_Ik_next_b(self, Q_ik, Y_Ik, Z_Ik, h_Np1k, X_Ik):
        num = Q_ik - Y_Ik - Z_Ik * h_Np1k
        den = X_Ik
        return num, den

    def _Q_im1k_next_b(self, U_Im1k, h_Ik, V_Im1k, W_Im1k, h_1k):
        t_0 = U_Im1k * h_Ik
        t_1 = V_Im1k
        t_2 = W_Im1k * h_1k
        return t_0 + t_1 + t_2

    def step(self, H_bc=None, Q_in=None, u=None, dt=None, first_time=False, implicit=True):
        _method = self._method
        self.link_hydraulic_geometry()
        self.compute_storage_areas()
        self.node_velocities()
        self.compute_flow_regime()
        self.link_coeffs(_dt=dt)
        self.node_coeffs(_dt=dt)
        self.forward_recurrence()
        self.backward_recurrence()
        self.superlink_upstream_head_coefficients()
        self.superlink_downstream_head_coefficients()
        self.superlink_flow_coefficients()
        self.sparse_matrix_equations(H_bc=H_bc, _Q_0j=Q_in, u=u,
                                     first_time=first_time, _dt=dt,
                                     implicit=implicit)
        self.solve_sparse_matrix(u=u, implicit=implicit)
        self.solve_superlink_flows()
        self.solve_orifice_flows(u=u)
        self.solve_superlink_depths()
        # self.solve_superlink_depths_alt()
        if _method == 'lsq':
            self.solve_internals_lsq()
        elif _method == 'b':
            self.solve_internals_backwards()
        elif _method == 'f':
            self.solve_internals_forwards()
        elif _method == 's':
            self.solve_internals_backwards(subcritical_only=True)
            self.solve_internals_forwards(supercritical_only=True)
