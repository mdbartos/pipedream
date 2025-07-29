import numpy as np
from numba import njit, prange
from numba.types import float64, int64, uint32, uint16, uint8, boolean, UniTuple, Tuple, List, DictType, void
import pipedream_solver.ngeometry

SMALLEST_NORMAL = np.finfo(np.float64).smallest_normal

MIN_SJ_AREA = 1e-8

CIRCULAR = 1
RECT_CLOSED = 2
RECT_OPEN = 3
TRIANGULAR = 4
TRAPEZOIDAL = 5
PARABOLIC = 6
ELLIPTICAL = 7
WIDE = 8
FORCE_MAIN = 9
FLOODPLAIN = 10

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            int64[:], int64[:], int64[:]),
      cache=True)
def numba_hydraulic_geometry(_A_ik, _Pe_ik, _R_ik, _B_ik, _h_Ik,
                             _g1_ik, _g2_ik, _g3_ik, _g4_ik, _g5_ik, _g6_ik, _g7_ik,
                             _geom_codes, _Ik, _ik):
    n = len(_ik)
    for i in range(n):
        I = _Ik[i]
        Ip1 = I + 1
        geom_code = _geom_codes[i]
        h_I = _h_Ik[I]
        h_Ip1 = _h_Ik[Ip1]
        h_i = (h_I + h_Ip1) / 2
        g1_i = _g1_ik[i]
        g2_i = _g2_ik[i]
        g3_i = _g3_ik[i]
        g4_i = _g4_ik[i]
        g5_i = _g5_ik[i]
        g6_i = _g6_ik[i]
        g7_i = _g7_ik[i]
        if geom_code:
            if geom_code == CIRCULAR:
                _A_ik[i] = pipedream_solver.ngeometry.Circular_A_ik(h_i, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Circular_Pe_ik(h_i, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Circular_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Circular_B_ik(h_i, g1_i, g2_i)
            elif geom_code == RECT_CLOSED:
                _A_ik[i] = pipedream_solver.ngeometry.Rect_Closed_A_ik(h_i, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Rect_Closed_Pe_ik(h_i, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Rect_Closed_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Rect_Closed_B_ik(h_i, g1_i, g2_i, g3_i)
            elif geom_code == RECT_OPEN:
                _A_ik[i] = pipedream_solver.ngeometry.Rect_Open_A_ik(h_i, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Rect_Open_Pe_ik(h_i, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Rect_Open_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Rect_Open_B_ik(h_i, g1_i, g2_i)
            elif geom_code == TRIANGULAR:
                _A_ik[i] = pipedream_solver.ngeometry.Triangular_A_ik(h_i, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Triangular_Pe_ik(h_i, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Triangular_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Triangular_B_ik(h_i, g1_i, g2_i)
            elif geom_code == TRAPEZOIDAL:
                _A_ik[i] = pipedream_solver.ngeometry.Trapezoidal_A_ik(h_i, g1_i, g2_i, g3_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Trapezoidal_Pe_ik(h_i, g1_i, g2_i, g3_i)
                _R_ik[i] = pipedream_solver.ngeometry.Trapezoidal_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Trapezoidal_B_ik(h_i, g1_i, g2_i, g3_i)
            elif geom_code == PARABOLIC:
                _A_ik[i] = pipedream_solver.ngeometry.Parabolic_A_ik(h_i, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Parabolic_Pe_ik(h_i, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Parabolic_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Parabolic_B_ik(h_i, g1_i, g2_i)
            elif geom_code == ELLIPTICAL:
                raise NotImplementedError
            elif geom_code == WIDE:
                _A_ik[i] = pipedream_solver.ngeometry.Wide_A_ik(h_i, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Wide_Pe_ik(h_i, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Wide_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Wide_B_ik(h_i, g1_i, g2_i)
            elif geom_code == FORCE_MAIN:
                _A_ik[i] = pipedream_solver.ngeometry.Force_Main_A_ik(h_i, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Force_Main_Pe_ik(h_i, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Force_Main_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Force_Main_B_ik(h_i, g1_i, g2_i)
            elif geom_code == FLOODPLAIN:
                _A_ik[i] = pipedream_solver.ngeometry.Floodplain_A_ik(h_i, g1_i, g2_i,
                                                                      g3_i, g4_i, g5_i, g6_i, g7_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Floodplain_Pe_ik(h_i, g1_i, g2_i,
                                                                        g3_i, g4_i, g5_i, g6_i, g7_i)
                _R_ik[i] = pipedream_solver.ngeometry.Floodplain_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Floodplain_B_ik(h_i, g1_i, g2_i,
                                                                      g3_i, g4_i, g5_i, g6_i, g7_i)
    return 1

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            int64[:], int64[:], int64[:], int64[:]),
      cache=True)
def numba_boundary_geometry(_A_bk, _Pe_bk, _R_bk, _B_bk, _h_Ik, _H_j, _z_inv_bk, _theta_bk,
                            _g1_ik, _g2_ik, _g3_ik, _g4_ik, _g5_ik, _g6_ik, _g7_ik,
                            _geom_codes, _i_bk, _I_bk, _J_bk):
    n = len(_i_bk)
    for k in range(n):
        i = _i_bk[k]
        I = _I_bk[k]
        j = _J_bk[k]
        # TODO: If theta is zero, should return critical depth
        h_I = _h_Ik[I]
        h_Ip1 = _theta_bk[k] * (_H_j[j] - _z_inv_bk[k])
        h_i = (h_I + h_Ip1) / 2
        geom_code = _geom_codes[i]
        g1_i = _g1_ik[i]
        g2_i = _g2_ik[i]
        g3_i = _g3_ik[i]
        g4_i = _g4_ik[i]
        g5_i = _g5_ik[i]
        g6_i = _g6_ik[i]
        g7_i = _g7_ik[i]
        if geom_code:
            if geom_code == CIRCULAR:
                _A_bk[k] = pipedream_solver.ngeometry.Circular_A_ik(h_i, g1_i, g2_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Circular_Pe_ik(h_i, g1_i, g2_i)
                _R_bk[k] = pipedream_solver.ngeometry.Circular_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Circular_B_ik(h_i, g1_i, g2_i)
            elif geom_code == RECT_CLOSED:
                _A_bk[k] = pipedream_solver.ngeometry.Rect_Closed_A_ik(h_i, g1_i, g2_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Rect_Closed_Pe_ik(h_i, g1_i, g2_i)
                _R_bk[k] = pipedream_solver.ngeometry.Rect_Closed_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Rect_Closed_B_ik(h_i, g1_i, g2_i, g3_i)
            elif geom_code == RECT_OPEN:
                _A_bk[k] = pipedream_solver.ngeometry.Rect_Open_A_ik(h_i, g1_i, g2_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Rect_Open_Pe_ik(h_i, g1_i, g2_i)
                _R_bk[k] = pipedream_solver.ngeometry.Rect_Open_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Rect_Open_B_ik(h_i, g1_i, g2_i)
            elif geom_code == TRIANGULAR:
                _A_bk[k] = pipedream_solver.ngeometry.Triangular_A_ik(h_i, g1_i, g2_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Triangular_Pe_ik(h_i, g1_i, g2_i)
                _R_bk[k] = pipedream_solver.ngeometry.Triangular_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Triangular_B_ik(h_i, g1_i, g2_i)
            elif geom_code == TRAPEZOIDAL:
                _A_bk[k] = pipedream_solver.ngeometry.Trapezoidal_A_ik(h_i, g1_i, g2_i, g3_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Trapezoidal_Pe_ik(h_i, g1_i, g2_i, g3_i)
                _R_bk[k] = pipedream_solver.ngeometry.Trapezoidal_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Trapezoidal_B_ik(h_i, g1_i, g2_i, g3_i)
            elif geom_code == PARABOLIC:
                _A_bk[k] = pipedream_solver.ngeometry.Parabolic_A_ik(h_i, g1_i, g2_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Parabolic_Pe_ik(h_i, g1_i, g2_i)
                _R_bk[k] = pipedream_solver.ngeometry.Parabolic_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Parabolic_B_ik(h_i, g1_i, g2_i)
            elif geom_code == ELLIPTICAL:
                raise NotImplementedError
            elif geom_code == WIDE:
                _A_bk[k] = pipedream_solver.ngeometry.Wide_A_ik(h_i, g1_i, g2_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Wide_Pe_ik(h_i, g1_i, g2_i)
                _R_bk[k] = pipedream_solver.ngeometry.Wide_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Wide_B_ik(h_i, g1_i, g2_i)
            elif geom_code == FORCE_MAIN:
                _A_bk[k] = pipedream_solver.ngeometry.Force_Main_A_ik(h_i, g1_i, g2_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Force_Main_Pe_ik(h_i, g1_i, g2_i)
                _R_bk[k] = pipedream_solver.ngeometry.Force_Main_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Force_Main_B_ik(h_i, g1_i, g2_i)
            elif geom_code == FLOODPLAIN:
                _A_bk[k] = pipedream_solver.ngeometry.Floodplain_A_ik(h_i, g1_i, g2_i,
                                                                      g3_i, g4_i, g5_i, g6_i, g7_i)
                _Pe_bk[k] = pipedream_solver.ngeometry.Floodplain_Pe_ik(h_i, g1_i, g2_i,
                                                                        g3_i, g4_i, g5_i, g6_i, g7_i)
                _R_bk[k] = pipedream_solver.ngeometry.Floodplain_R_ik(_A_bk[k], _Pe_bk[k])
                _B_bk[k] = pipedream_solver.ngeometry.Floodplain_B_ik(h_i, g1_i, g2_i,
                                                                      g3_i, g4_i, g5_i, g6_i, g7_i)
    return 1

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            int64[:], int64),
      cache=True)
def numba_orifice_geometry(_Ao, h_eo, u_o, _g1_o, _g2_o, _g3_o, _geom_codes_o, n_o):
    for i in range(n_o):
        geom_code = _geom_codes_o[i]
        g1 = _g1_o[i]
        g2 = _g2_o[i]
        g3 = _g3_o[i]
        u = u_o[i]
        h_e = h_eo[i]
        if geom_code:
            if geom_code == CIRCULAR:
                _Ao[i] = pipedream_solver.ngeometry.Circular_A_ik(h_e, g1 * u, g2)
            elif geom_code == RECT_CLOSED:
                _Ao[i] = pipedream_solver.ngeometry.Rect_Closed_A_ik(h_e, g1 * u, g2)
            elif geom_code == RECT_OPEN:
                _Ao[i] = pipedream_solver.ngeometry.Rect_Open_A_ik(h_e, g1 * u, g2)
            elif geom_code == TRIANGULAR:
                _Ao[i] = pipedream_solver.ngeometry.Triangular_A_ik(h_e, g1 * u, g2)
            elif geom_code == TRAPEZOIDAL:
                _Ao[i] = pipedream_solver.ngeometry.Trapezoidal_A_ik(h_e, g1 * u, g2, g3)
            elif geom_code == PARABOLIC:
                _Ao[i] = pipedream_solver.ngeometry.Parabolic_A_ik(h_e, g1 * u, g2)
            elif geom_code == ELLIPTICAL:
                raise NotImplementedError
            elif geom_code == WIDE:
                _Ao[i] = pipedream_solver.ngeometry.Wide_A_ik(h_e, g1 * u, g2)
            elif geom_code == FORCE_MAIN:
                _Ao[i] = pipedream_solver.ngeometry.Force_Main_A_ik(h_e, g1 * u, g2)
            elif geom_code == FLOODPLAIN:
                # TODO: This can be implemented
                raise NotImplementedError
    return 1

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], boolean[:], float64[:],
            float64[:], float64[:], float64[:], float64[:],
            int64[:], int64[:], int64[:], int64[:], int64[:]),
      cache=True)
def numba_transect_geometry(_A_ik, _Pe_ik, _R_ik, _B_ik, _h_Ik, _is_irregular, _transect_zs,
                            _transect_As, _transect_Bs, _transect_Pes, _transect_Rs,
                            _transect_codes, _transect_inds, _transect_lens, _Ik, _ik):
    n = len(_ik)
    for i in range(n):
        is_irregular = _is_irregular[i]
        if is_irregular:
            I = _Ik[i]
            Ip1 = I + 1
            transect_code = _transect_codes[i]
            h_I = _h_Ik[I]
            h_Ip1 = _h_Ik[Ip1]
            h_i = (h_I + h_Ip1) / 2
            start = _transect_inds[transect_code]
            size = _transect_lens[transect_code]
            end = start + size
            _z_range = _transect_zs[start:end]
            _A_range = _transect_As[start:end]
            _B_range = _transect_Bs[start:end]
            _Pe_range = _transect_Pes[start:end]
            _R_range = _transect_Rs[start:end]
            _A_ik[i] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _A_range, 1)
            _B_ik[i] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _B_range, 1)
            _Pe_ik[i] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _Pe_range, 1)
            _R_ik[i] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _R_range, 1)
    return 1

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], boolean[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], int64[:], int64[:],
            int64[:], int64[:], int64[:], int64[:]),
      cache=True)
def numba_boundary_transect(_A_bk, _Pe_bk, _R_bk, _B_bk, _h_Ik, _H_j, _z_inv_bk,
                            _theta_bk, _is_irregular, _transect_zs, _transect_As, _transect_Bs,
                            _transect_Pes, _transect_Rs, _transect_codes, _transect_inds,
                            _transect_lens, _I_bk, _i_bk, _J_bk):
    n = len(_i_bk)
    for k in range(n):
        i = _i_bk[k]
        I = _I_bk[k]
        j = _J_bk[k]
        is_irregular_bk = _is_irregular[i]
        if is_irregular_bk:
            h_I = _h_Ik[I]
            # TODO: This should incorporate theta
            h_Ip1 = _theta_bk[k] * (_H_j[j] - _z_inv_bk[k])
            transect_code = _transect_codes[i]
            h_i = (h_I + h_Ip1) / 2
            start = _transect_inds[transect_code]
            size = _transect_lens[transect_code]
            end = start + size
            _z_range = _transect_zs[start:end]
            _A_range = _transect_As[start:end]
            _B_range = _transect_Bs[start:end]
            _Pe_range = _transect_Pes[start:end]
            _R_range = _transect_Rs[start:end]
            _A_bk[k] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _A_range, 1)
            _B_bk[k] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _B_range, 1)
            _Pe_bk[k] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _Pe_range, 1)
            _R_bk[k] = pipedream_solver.ngeometry.interpolate_geometry(h_i, _z_range, _R_range, 1)
    return 1

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], boolean[:]),
      cache=True)
def numba_compute_functional_storage_areas(h, A, a, b, c, _functional):
    M = h.size
    for j in range(M):
        if _functional[j]:
            if h[j] < 0:
                A[j] = 0
            else:
                A[j] = max(a[j] * (h[j]**b[j]) + c[j], MIN_SJ_AREA)
    return A

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], boolean[:]),
      cache=True)
def numba_compute_functional_storage_volumes(h, V, a, b, c, _functional):
    M = h.size
    for j in range(M):
        if _functional[j]:
            if h[j] < 0:
                V[j] = 0
            else:
                # TODO: Ensure this works when functional storage area is negative
                V[j] = (a[j] / (b[j] + 1)) * h[j] ** (b[j] + 1) + c[j] * h[j]
    return V

@njit
def numba_compute_tabular_storage_areas(h_j, A_sj, hs, As, sjs, sts, inds, lens):
    n = sjs.size
    for i in range(n):
        sj = sjs[i]
        st = sts[i]
        ind = inds[st]
        size = lens[st]
        h_range = hs[ind:ind+size]
        A_range = As[ind:ind+size]
        Amin = A_range.min()
        Amax = A_range.max()
        h_search = h_j[sj]
        ix = np.searchsorted(h_range, h_search)
        # NOTE: np.interp not supported in this version of numba
        # A_result = np.interp(h_search, h_range, A_range)
        # A_out[i] = A_result
        if (ix == 0):
            A_sj[sj] = Amin
        elif (ix >= size):
            A_sj[sj] = Amax
        else:
            dx_0 = h_search - h_range[ix - 1]
            dx_1 = h_range[ix] - h_search
            frac = dx_0 / (dx_0 + dx_1)
            A_sj[sj] = (1 - frac) * A_range[ix - 1] + (frac) * A_range[ix]
    return A_sj

@njit
def numba_compute_tabular_storage_volumes(h_j, V_sj, hs, As, Vs, sjs, sts, inds, lens):
    n = sjs.size
    for i in range(n):
        sj = sjs[i]
        st = sts[i]
        ind = inds[st]
        size = lens[st]
        h_range = hs[ind:ind+size]
        A_range = As[ind:ind+size]
        V_range = Vs[ind:ind+size]
        hmax = h_range.max()
        Vmin = V_range.min()
        Vmax = V_range.max()
        Amax = A_range.max()
        h_search = h_j[sj]
        ix = np.searchsorted(h_range, h_search)
        # NOTE: np.interp not supported in this version of numba
        # A_result = np.interp(h_search, h_range, A_range)
        # A_out[i] = A_result
        if (ix == 0):
            V_sj[sj] = Vmin
        elif (ix >= size):
            V_sj[sj] = Vmax + Amax * (h_search - hmax)
        else:
            dx_0 = h_search - h_range[ix - 1]
            dx_1 = h_range[ix] - h_search
            frac = dx_0 / (dx_0 + dx_1)
            V_sj[sj] = (1 - frac) * V_range[ix - 1] + (frac) * V_range[ix]
    return V_sj

@njit(float64(float64, float64, float64, float64, float64, float64, float64))
def friction_slope(Q_ik_t, dx_ik, A_ik, R_ik, n_ik, Sf_method_ik, g=9.81):
    if A_ik > 0:
        # Chezy-Manning eq.
        if Sf_method_ik == 0:
            t_1 = (g * n_ik**2 * np.abs(Q_ik_t) * dx_ik
                   / A_ik / R_ik**(4/3))
        # Hazen-Williams eq.
        elif Sf_method_ik == 1:
            t_1 = (1.354 * g * np.abs(Q_ik_t)**0.85 * dx_ik
                   / A_ik**0.85 / n_ik**1.85 / R_ik**1.1655)
        # Darcy-Weisbach eq.
        elif Sf_method_ik == 2:
            # kinematic viscosity(meter^2/sec), we can consider this is constant.
            nu = 0.0000010034
            Re = (np.abs(Q_ik_t) / A_ik) * 4 * R_ik / nu
            if Re > 2000:
                f = 0.25 / (np.log10(n_ik / (3.7 * 4 * R_ik) + 5.74 / (Re**0.9)))**2
            elif (Re > 0) and (Re <= 2000):
                f = 64 / Re
            elif (Re == 0):
                f = 0.1
            else:
                raise ValueError('Reynolds number outside allowable range')
            t_1 = (0.01274 * g * f * np.abs(Q_ik_t) * dx_ik
                   / (A_ik * R_ik))
        else:
            raise ValueError('Invalid friction method.')
        return t_1
    else:
        return 0.

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64),
      cache=True)
def tnumba_a_ik(u_ik, B_ik, A_ik, dx_ik, dt, g=9.81):
    t_0 = -u_ik * B_ik * dx_ik / dt
    t_1 = u_ik**2 * B_ik
    t_2 = -g * A_ik
    result = t_0 + t_1 + t_2
    return result

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64, float64),
      cache=True)
def tnumba_c_ik(u_ik, B_ik, A_ik, dx_ik, dt, g=9.81):
    t_0 = -u_ik * B_ik * dx_ik / dt
    t_1 = -u_ik**2 * B_ik
    t_2 = g * A_ik
    result = t_0 + t_1 + t_2
    return result

@njit(float64[:](float64[:], float64, float64[:], float64[:],
                 float64[:], float64[:], float64[:], int64[:], float64),
      cache=True)
def tnumba_b_ik(dx_ik, dt, n_ik, Q_ik_t, 
               A_ik, R_ik, C_ik, Sf_method_ik, g=9.81):
    t_0 = dx_ik / dt
    t_1 = np.zeros(Q_ik_t.size)
    k = len(Sf_method_ik)
    for n in range(k):
        t_1[n] = friction_slope(Q_ik_t[n], dx_ik[n], A_ik[n], R_ik[n],
                                n_ik[n], Sf_method_ik[n], g)
    t_2 = C_ik * np.abs(Q_ik_t) / 2 / g
    result = t_0 + t_1 + t_2
    return result

@njit(float64[:](float64[:], float64[:], float64, float64[:], float64[:], float64[:],
                 float64[:], float64[:], float64[:], float64),
      cache=True)
def tnumba_P_ik(Q_ik_t, dx_ik, dt, u_ik, B_ik, A_ik, 
         h_Ik_t, h_Ip1k_t, S_o_ik, g=9.81):
    t0 = Q_ik_t * dx_ik / dt
    t1 = - u_ik * B_ik * dx_ik * (h_Ik_t + h_Ip1k_t) / dt
    t2 = g * A_ik * S_o_ik * dx_ik
    result = t0 + t1 + t2
    return result

@njit(float64(float64, float64, float64, float64, float64, float64),
      cache=True)
def E_Ik(B_ik, dx_ik, B_im1k, dx_im1k, A_SIk, dt):
    """
    Compute node coefficient 'E' for node I, superlink k.
    """
    t_0 = B_ik * dx_ik / 2
    t_1 = B_im1k * dx_im1k / 2
    t_2 = A_SIk
    t_3 = dt
    return (t_0 + t_1 + t_2) / t_3

@njit(float64(float64, float64, float64, float64, float64, float64, float64, float64),
      cache=True)
def D_Ik(Q_0IK, B_ik, dx_ik, B_im1k, dx_im1k, A_SIk, h_Ik_t, dt):
    """
    Compute node coefficient 'D' for node I, superlink k.
    """
    t_0 = Q_0IK
    t_1 = B_ik * dx_ik / 2
    t_2 = B_im1k * dx_im1k / 2
    t_3 = A_SIk
    t_4 = h_Ik_t / dt
    return t_0 + ((t_1 + t_2 + t_3) * t_4)

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], float64[:], int64[:], float64,
            int64[:], int64[:], boolean[:], boolean[:]),
      cache=True)
def numba_node_coeffs(_D_Ik, _E_Ik, _Q_0Ik, _B_ik, _h_Ik, _dx_ik, _A_SIk,
                      _B_uk, _B_dk, _dx_uk, _dx_dk, _kI, _dt,
                      _forward_I_i, _backward_I_i, _is_start, _is_end):
    N = _h_Ik.size
    for I in range(N):
        k = _kI[I]
        if _is_start[I]:
            i = _forward_I_i[I]
            _E_Ik[I] = E_Ik(_B_ik[i], _dx_ik[i], _B_uk[k], _dx_uk[k], _A_SIk[I], _dt)
            _D_Ik[I] = D_Ik(_Q_0Ik[I], _B_ik[i], _dx_ik[i], _B_uk[k], _dx_uk[k], _A_SIk[I],
                            _h_Ik[I], _dt)
        elif _is_end[I]:
            im1 = _backward_I_i[I]
            _E_Ik[I] = E_Ik(_B_dk[k], _dx_dk[k], _B_ik[im1], _dx_ik[im1],
                            _A_SIk[I], _dt)
            _D_Ik[I] = D_Ik(_Q_0Ik[I], _B_dk[k], _dx_dk[k], _B_ik[im1],
                            _dx_ik[im1], _A_SIk[I], _h_Ik[I], _dt)
        else:
            i = _forward_I_i[I]
            im1 = i - 1
            _E_Ik[I] = E_Ik(_B_ik[i], _dx_ik[i], _B_ik[im1], _dx_ik[im1],
                            _A_SIk[I], _dt)
            _D_Ik[I] = D_Ik(_Q_0Ik[I], _B_ik[i], _dx_ik[i], _B_ik[im1],
                            _dx_ik[im1], _A_SIk[I], _h_Ik[I], _dt)
    return 1

@njit(float64(float64, float64),
      cache=True)
def safe_divide(num, den):
    return (num + SMALLEST_NORMAL) / (den + SMALLEST_NORMAL)

# TODO: This is really just meant for orifice, weir, and pump equations
# Where division by zero implies entire expression is zero
@njit(float64[:](float64[:], float64[:]),
      cache=True)
def safe_divide_vec(num, den):
    result = np.zeros_like(num)
    is_safe = (np.abs(den) > SMALLEST_NORMAL)
    result[is_safe] = num[is_safe] / den[is_safe]
    return result

@njit(float64(float64, float64, float64, float64, float64),
      cache=True)
def Q_i_f(h_Ip1k, h_1k, U_Ik, V_Ik, W_Ik):
    t_0 = U_Ik * h_Ip1k
    t_1 = V_Ik
    t_2 = W_Ik * h_1k
    return t_0 + t_1 + t_2

@njit(float64(float64, float64, float64, float64, float64),
      cache=True)
def Q_i_b(h_Ik, h_Np1k, X_Ik, Y_Ik, Z_Ik):
    t_0 = X_Ik * h_Ik
    t_1 = Y_Ik
    t_2 = Z_Ik * h_Np1k
    return t_0 + t_1 + t_2

@njit(float64(float64, float64, float64, float64, float64),
      cache=True)
def h_i_b(Q_ik, h_Np1k, X_Ik, Y_Ik, Z_Ik):
    num = Q_ik - Y_Ik - Z_Ik * h_Np1k
    den = X_Ik
    result = safe_divide(num, den)
    return result

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], int64[:], int64[:], int64[:], int64,
            float64, float64[:], boolean),
      cache=True)
def numba_solve_internals(_h_Ik, _Q_ik, _h_uk, _h_dk, _U_Ik, _V_Ik, _W_Ik,
                          _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, nk, NK,
                          min_depth, max_depth_k, first_link_backwards=True):
    for k in range(NK):
        n = nk[k]
        i_1 = _i_1k[k]
        I_1 = _I_1k[k]
        i_n = i_1 + n - 1
        I_Np1 = I_1 + n
        I_N = I_Np1 - 1
        # Set boundary depths
        _h_1k = _h_uk[k]
        _h_Np1k = _h_dk[k]
        _h_Ik[I_1] = _h_1k
        _h_Ik[I_Np1] = _h_Np1k
        # Set max depth
        max_depth = max_depth_k[k]
        # Compute internal depths and flows (except first link flow)
        for j in range(n - 1):
            I = I_N - j
            Ip1 = I + 1
            i = i_n - j
            _Q_ik[i] = Q_i_f(_h_Ik[Ip1], _h_1k, _U_Ik[I], _V_Ik[I], _W_Ik[I])
            _h_Ik[I] = h_i_b(_Q_ik[i], _h_Np1k, _X_Ik[I], _Y_Ik[I], _Z_Ik[I])
            #if _h_Ik[I] < min_depth:
            #    _h_Ik[I] = min_depth
            #if _h_Ik[I] > max_depth:
            #    _h_Ik[I] = max_depth
        if first_link_backwards:
            _Q_ik[i_1] = Q_i_b(_h_Ik[I_1], _h_Np1k, _X_Ik[I_1], _Y_Ik[I_1],
                            _Z_Ik[I_1])
        else:
            # Not theoretically correct, but seems to be more stable sometimes
            _Q_ik[i_1] = Q_i_f(_h_Ik[I_1 + 1], _h_1k, _U_Ik[I_1], _V_Ik[I_1],
                            _W_Ik[I_1])
    return 1


@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], int64[:], int64[:], int64[:], int64,
            float64, float64[:], boolean),
      cache=True)
def numba_solve_internals_matrix(_h_Ik, _Q_ik, _h_uk, _h_dk, _U_Ik, _V_Ik, _W_Ik,
                          _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, nk, NK,
                          min_depth, max_depth_k, first_link_backwards=True):
    for k in range(NK):
        n = nk[k]
        i_1 = _i_1k[k]
        I_1 = _I_1k[k]
        i_n = i_1 + n - 1
        I_2 = I_1 + 1
        I_Np1 = I_1 + n
        I_N = I_Np1 - 1
        # Set boundary depths
        _h_1k = _h_uk[k]
        _h_Np1k = _h_dk[k]
        _h_Ik[I_1] = _h_1k
        _h_Ik[I_Np1] = _h_Np1k
        # Set up matrix
        dim = 2 * n - 1
        A = np.zeros((dim, dim))
        b = np.zeros(dim)
        # Fill linear system
        #### h indices
        for j, i in enumerate(range(1, dim, 2)):
            A[i, i] = -_X_Ik[I_2+j]
            A[i, i+1] = 1.
            b[i] = _Y_Ik[I_2+j] + _h_Np1k * _Z_Ik[I_2+j]
        #### Q indices
        for j, i in enumerate(range(2, dim, 2)):
            A[i, i] = 1.
            if i < dim - 1:
                A[i, i+1] = -_U_Ik[I_2+j]
            b[i] = _V_Ik[I_2+j] + _h_1k * _W_Ik[I_2+j]
        #### Other indices
        A[0 ,0] = 1.
        A[0, 1] = -_X_Ik[I_1]
        b[0] = _Y_Ik[I_1] + _h_Np1k * _Z_Ik[I_1]
        b[-1] += _h_1k * _U_Ik[I_N]
        # Solve linear system
        x = np.linalg.solve(A, b)
        # Write to output arrays
        _Q_ik[i_1:i_1+n] = x[::2]
        _h_Ik[I_2:I_2+n-1] = x[1::2]
    return 1


@njit(float64[:](float64[:], int64, int64[:], int64[:], int64[:], int64[:], float64[:], float64[:], float64[:]),
      cache=True)
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
        _AkT = _Ak.T.copy()
        _bk = _b[rstart:rstart+nlinks].copy()
        _AA = _AkT @ _Ak
        _Ab = _AkT @ _bk
        # If want to prevent singular matrix, set ( diag == 0 ) = 1
        for i in range(nlinks - 1):
            if (_AA[i, i] == 0.0):
                _AA[i, i] = 1.0
        _h_inner = np.linalg.solve(_AA, _Ab)
        _h_Ik[jstart+1:jstart+nlinks] = _h_inner
    return _h_Ik

@njit(float64[:](float64[:], float64[:], float64[:]),
      cache=True)
def numba_u_ik(_Q_ik, _A_ik, _u_ik):
    n = _u_ik.size
    for i in range(n):
        _Q_i = _Q_ik[i]
        _A_i = _A_ik[i]
        if _A_i > SMALLEST_NORMAL:
            _u_ik[i] = _Q_i / _A_i
        else:
            _u_ik[i] = 0.
    return _u_ik

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], boolean[:], int64[:], float64[:]),
      cache=True)
def numba_u_Ik(_dx_ik, _u_ik, _dx_uk, _u_uk, _link_start, _ki, _u_Ik):
    n = _u_Ik.size
    for i in range(n):
        k = _ki[i]
        if _link_start[i]:
            num = _dx_ik[i] * _u_uk[k] + _dx_uk[k] * _u_ik[i]
            den = _dx_ik[i] + _dx_uk[k]
            if den > SMALLEST_NORMAL:
                _u_Ik[i] = num / den
            else:
                _u_Ik[i] = 0.
        else:
            im1 = i - 1
            num = _dx_ik[i] * _u_ik[im1] + _dx_ik[im1] * _u_ik[i]
            den = _dx_ik[i] + _dx_ik[im1]
            if den > SMALLEST_NORMAL:
                _u_Ik[i] = num / den
            else:
                _u_Ik[i] = 0.
    return _u_Ik

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], boolean[:], int64[:], float64[:]),
      cache=True)
def numba_u_Ip1k(_dx_ik, _u_ik, _dx_dk, _u_dk, _link_end, _ki, _u_Ip1k):
    n = _u_Ip1k.size
    for i in range(n):
        k = _ki[i]
        if _link_end[i]:
            num = _dx_ik[i] * _u_dk[k] + _dx_dk[k] * _u_ik[i]
            den = _dx_ik[i] + _dx_dk[k]
            if den > SMALLEST_NORMAL:
                _u_Ip1k[i] = num / den
            else:
                _u_Ip1k[i] = 0.
        else:
            ip1 = i + 1
            num = _dx_ik[i] * _u_ik[ip1] + _dx_ik[ip1] * _u_ik[i]
            den = _dx_ik[i] + _dx_ik[ip1]
            if den > SMALLEST_NORMAL:
                _u_Ip1k[i] = num / den
            else:
                _u_Ip1k[i] = 0.
    return _u_Ip1k

@njit(float64(float64, float64),
      cache=True)
def U_1k(c_1k, T_1k):
    """
    Compute forward recurrence coefficient 'U' for node 1, superlink k.
    """
    num = -c_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64),
      cache=True)
def V_1k(P_1k, T_1k):
    """
    Compute forward recurrence coefficient 'V' for node 1, superlink k.
    """
    num = P_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64),
      cache=True)
def W_1k(a_1k, T_1k):
    """
    Compute forward recurrence coefficient 'W' for node 1, superlink k.
    """
    num = -a_1k
    den = T_1k
    result = safe_divide(num, den)
    return result

@njit(float64(float64),
      cache=True)
def T_1k(b_1k):
    """
    Compute forward recurrence coefficient 'T' for link 1, superlink k.
    """
    return b_1k

@njit(float64(float64, float64, float64, float64),
      cache=True)
def U_Ik(E_Ik, c_ik, U_Im1k, T_ik):
    """
    Compute forward recurrence coefficient 'U' for node I, superlink k.
    """
    num = c_ik * (E_Ik - U_Im1k)
    den = T_ik
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64, float64, float64, float64, float64, float64),
      cache=True)
def V_Ik(P_ik, a_ik, D_Ik, E_Ik, V_Im1k, U_Im1k, T_ik):
    """
    Compute forward recurrence coefficient 'V' for node I, superlink k.
    """
    num = a_ik * (D_Ik + V_Im1k) - P_ik * (E_Ik - U_Im1k)
    den = T_ik
    result = safe_divide(num, den)
    return  result

@njit(float64(float64, float64, float64),
      cache=True)
def W_Ik(a_ik, W_Im1k, T_ik):
    """
    Compute forward recurrence coefficient 'W' for node I, superlink k.
    """
    num = a_ik * W_Im1k
    den = T_ik
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64, float64, float64),
      cache=True)
def T_ik(a_ik, b_ik, E_Ik, U_Im1k):
    """
    Compute forward recurrence coefficient 'T' for link i, superlink k.
    """
    result = a_ik - b_ik * (E_Ik - U_Im1k)
    return result

@njit(float64(float64, float64),
      cache=True)
def X_Nk(a_nk, O_nk):
    """
    Compute backward recurrence coefficient 'X' for node N, superlink k.
    """
    num = -a_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64),
      cache=True)
def Y_Nk(P_nk, O_nk):
    """
    Compute backward recurrence coefficient 'Y' for node N, superlink k.
    """
    num = P_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64),
      cache=True)
def Z_Nk(c_nk, O_nk):
    """
    Compute backward recurrence coefficient 'Z' for node N, superlink k.
    """
    num = -c_nk
    den = O_nk
    result = safe_divide(num, den)
    return result

@njit(float64(float64),
      cache=True)
def O_nk(b_nk):
    """
    Compute backward recurrence coefficient 'O' for link n, superlink k.
    """
    return b_nk

@njit(float64(float64, float64, float64, float64),
      cache=True)
def X_Ik(a_ik, E_Ip1k, X_Ip1k, O_ik):
    """
    Compute backward recurrence coefficient 'X' for node I, superlink k.
    """
    num = -a_ik * (E_Ip1k + X_Ip1k)
    den = O_ik
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64, float64, float64, float64, float64, float64),
      cache=True)
def Y_Ik(P_ik, c_ik, D_Ip1k, E_Ip1k, Y_Ip1k, X_Ip1k, O_ik):
    """
    Compute backward recurrence coefficient 'Y' for node I, superlink k.
    """
    num = -c_ik * (D_Ip1k - Y_Ip1k) + P_ik * (E_Ip1k + X_Ip1k)
    den = O_ik
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64, float64),
      cache=True)
def Z_Ik(c_ik, Z_Ip1k, O_ik):
    """
    Compute backward recurrence coefficient 'Z' for node I, superlink k.
    """
    num = c_ik * Z_Ip1k
    den = O_ik
    result = safe_divide(num, den)
    return result

@njit(float64(float64, float64, float64, float64),
      cache=True)
def O_ik(b_ik, c_ik, E_Ip1k, X_Ip1k):
    """
    Compute backward recurrence coefficient 'O' for link i, superlink k.
    """
    result = c_ik + b_ik * (E_Ip1k + X_Ip1k)
    return result

@njit(float64[:](float64[:], float64[:], float64[:], float64),
      cache=True)
def gamma_o(Q_o_t, Ao, Co, g=9.81):
    """
    Compute flow coefficient 'gamma' for orifice o.
    """
    num = 2 * g * Co**2 * Ao**2
    den = np.abs(Q_o_t)
    result = safe_divide_vec(num, den)
    return result

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]),
      cache=True)
def gamma_w(Q_w_t, H_w_t, L_w, s_w, Cwr, Cwt):
    """
    Compute flow coefficient 'gamma' for weir w.
    """
    num = (Cwr * L_w * H_w_t + Cwt * s_w * H_w_t**2)**2
    den = np.abs(Q_w_t)
    result = safe_divide_vec(num, den)
    return result

@njit(float64[:](float64[:], float64[:], float64[:], float64[:]),
      cache=True)
def gamma_p(Q_p_t, b_p, c_p, u):
    """
    Compute flow coefficient 'gamma' for pump p.
    """
    num = u
    den = b_p * np.abs(Q_p_t)**(c_p - 1)
    result = safe_divide_vec(num, den)
    return result

@njit(float64[:](float64[:], float64[:], float64[:]),
      cache=True)
def xi_uk(dx_uk, B_uk, theta_uk):
    result = (dx_uk * B_uk * theta_uk) / 2
    return result

@njit(float64[:](float64[:], float64[:], float64[:]),
      cache=True)
def xi_dk(dx_dk, B_dk, theta_dk):
    result = (dx_dk * B_dk * theta_dk) / 2
    return result

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], float64[:], float64[:], boolean[:],
             int64[:], int64[:]),
      cache=True)
def numba_orifice_flow_coefficients(_alpha_o, _beta_o, _chi_o, H_j, _Qo, u, _z_inv_j,
                                    _z_o, _tau_o, _Co, _Ao, _y_max_o, _unidir_o,
                                     _J_uo, _J_do):
    g = 9.81
    _H_uo = H_j[_J_uo]
    _H_do = H_j[_J_do]
    _z_inv_uo = _z_inv_j[_J_uo]
    # Create indicator functions
    _omega_o = np.zeros_like(_H_uo)
    _omega_o[_H_uo >= _H_do] = 1.0
    # Compute universal coefficients
    _gamma_o = gamma_o(_Qo, _Ao, _Co, g)
    # Create conditionals
    cond_0 = (_omega_o * _H_uo + (1 - _omega_o) * _H_do >
                _z_o + _z_inv_uo + (_tau_o * _y_max_o * u))
    cond_1 = ((1 - _omega_o) * _H_uo + _omega_o * _H_do >
                _z_o + _z_inv_uo + (_tau_o * _y_max_o * u / 2))
    cond_2 = (_omega_o * _H_uo + (1 - _omega_o) * _H_do >
                _z_o + _z_inv_uo)
    cond_3 = (_H_do >= _H_uo) & _unidir_o
    # Fill coefficient arrays
    # Submerged on both sides
    a = (cond_0 & cond_1)
    _alpha_o[a] = _gamma_o[a]
    _beta_o[a] = -_gamma_o[a]
    _chi_o[a] = 0.0
    # Submerged on one side
    b = (cond_0 & ~cond_1)
    _alpha_o[b] = _gamma_o[b] * _omega_o[b] * (-1)**(1 - _omega_o[b])
    _beta_o[b] = _gamma_o[b] * (1 - _omega_o[b]) * (-1)**(1 - _omega_o[b])
    _chi_o[b] = (_gamma_o[b] * (-1)**(1 - _omega_o[b])
                                    * (- _z_inv_uo[b] - _z_o[b] -
                                        _tau_o[b] * _y_max_o[b] * u[b] / 2))
    # Weir flow
    c = (~cond_0 & cond_2)
    _alpha_o[c] = _gamma_o[c] * _omega_o[c] * (-1)**(1 - _omega_o[c])
    _beta_o[c] = _gamma_o[c] * (1 - _omega_o[c]) * (-1)**(1 - _omega_o[c])
    _chi_o[c] = (_gamma_o[c] * (-1)**(1 - _omega_o[c])
                                    * (- _z_inv_uo[c] - _z_o[c]))
    # No flow
    d = (~cond_0 & ~cond_2) | cond_3
    _alpha_o[d] = 0.0
    _beta_o[d] = 0.0
    _chi_o[d] = 0.0
    return 1

@njit(float64[:](float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], float64[:], boolean[:],
            int64[:], int64[:], float64),
      cache=True)
def numba_solve_orifice_flows(H_j, u, _z_inv_j, _z_o,
                              _tau_o, _y_max_o, _Co, _Ao, _unidir_o, _J_uo, _J_do, g=9.81):
    # Specify orifice heads at previous timestep
    _H_uo = H_j[_J_uo]
    _H_do = H_j[_J_do]
    _z_inv_uo = _z_inv_j[_J_uo]
    # Create indicator functions
    _omega_o = np.zeros_like(_H_uo)
    _omega_o[_H_uo >= _H_do] = 1.0
    # Create arrays to store flow coefficients for current time step
    _alpha_o = np.zeros_like(_H_uo)
    _beta_o = np.zeros_like(_H_uo)
    _chi_o = np.zeros_like(_H_uo)
    # Compute universal coefficients
    _gamma_o = 2 * g * _Co**2 * _Ao**2
    # Create conditionals
    cond_0 = (_omega_o * _H_uo + (1 - _omega_o) * _H_do >
                _z_o + _z_inv_uo + (_tau_o * _y_max_o * u))
    cond_1 = ((1 - _omega_o) * _H_uo + _omega_o * _H_do >
                _z_o + _z_inv_uo + (_tau_o * _y_max_o * u / 2))
    cond_2 = (_omega_o * _H_uo + (1 - _omega_o) * _H_do >
                _z_o + _z_inv_uo)
    cond_3 = (_H_do >= _H_uo) & _unidir_o
    # Fill coefficient arrays
    # Submerged on both sides
    a = (cond_0 & cond_1)
    _alpha_o[a] = _gamma_o[a]
    _beta_o[a] = -_gamma_o[a]
    _chi_o[a] = 0.0
    # Submerged on one side
    b = (cond_0 & ~cond_1)
    _alpha_o[b] = _gamma_o[b] * _omega_o[b] * (-1)**(1 - _omega_o[b])
    _beta_o[b] = _gamma_o[b] * (1 - _omega_o[b]) * (-1)**(1 - _omega_o[b])
    _chi_o[b] = (_gamma_o[b] * (-1)**(1 - _omega_o[b])
                                    * (- _z_inv_uo[b] - _z_o[b]
                                        - _tau_o[b] * _y_max_o[b] * u[b] / 2))
    # Weir flow on one side
    c = (~cond_0 & cond_2)
    _alpha_o[c] = _gamma_o[c] * _omega_o[c] * (-1)**(1 - _omega_o[c])
    _beta_o[c] = _gamma_o[c] * (1 - _omega_o[c]) * (-1)**(1 - _omega_o[c])
    _chi_o[c] = (_gamma_o[c] * (-1)**(1 - _omega_o[c])
                                    * (- _z_inv_uo[c] - _z_o[c]))
    # No flow
    d = (~cond_0 & ~cond_2) | cond_3
    _alpha_o[d] = 0.0
    _beta_o[d] = 0.0
    _chi_o[d] = 0.0
    # Compute flow
    _Qo_next = (-1)**(1 - _omega_o) * np.sqrt(np.abs(
               _alpha_o * _H_uo + _beta_o * _H_do + _chi_o))
    # Export instance variables
    return _Qo_next

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int64[:], int64[:]),
      cache=True)
def numba_weir_flow_coefficients(_Hw, _Qw, _alpha_w, _beta_w, _chi_w, H_j, _z_inv_j, _z_w,
                                 _y_max_w, u, _L_w, _s_w, _Cwr, _Cwt, _J_uw, _J_dw):
    # Specify weir heads at previous timestep
    _H_uw = H_j[_J_uw]
    _H_dw = H_j[_J_dw]
    _z_inv_uw = _z_inv_j[_J_uw]
    # Create indicator functions
    _omega_w = np.zeros(_H_uw.size)
    _omega_w[_H_uw >= _H_dw] = 1.0
    # Create conditionals
    cond_0 = (_omega_w * _H_uw + (1 - _omega_w) * _H_dw >
                _z_w + _z_inv_uw + (1 - u) * _y_max_w)
    cond_1 = ((1 - _omega_w) * _H_uw + _omega_w * _H_dw >
                _z_w + _z_inv_uw + (1 - u) * _y_max_w)
    # Effective heads
    a = (cond_0 & cond_1)
    b = (cond_0 & ~cond_1)
    c = (~cond_0)
    _Hw[a] = _H_uw[a] - _H_dw[a]
    _Hw[b] = (_omega_w[b] * _H_uw[b] + (1 - _omega_w[b]) * _H_dw[b]
                    + (-_z_inv_uw[b] - _z_w[b] - (1 - u[b]) * _y_max_w[b]))
    _Hw[c] = 0.0
    _Hw = np.abs(_Hw)
    # Compute universal coefficients
    _gamma_w = gamma_w(_Qw, _Hw, _L_w, _s_w, _Cwr, _Cwt)
    # Fill coefficient arrays
    # Submerged on both sides
    a = (cond_0 & cond_1)
    _alpha_w[a] = _gamma_w[a]
    _beta_w[a] = -_gamma_w[a]
    _chi_w[a] = 0.0
    # Submerged on one side
    b = (cond_0 & ~cond_1)
    _alpha_w[b] = _gamma_w[b] * _omega_w[b] * (-1)**(1 - _omega_w[b])
    _beta_w[b] = _gamma_w[b] * (1 - _omega_w[b]) * (-1)**(1 - _omega_w[b])
    _chi_w[b] = (_gamma_w[b] * (-1)**(1 - _omega_w[b]) *
                                (- _z_inv_uw[b] - _z_w[b] - (1 - u[b]) * _y_max_w[b]))
    # No flow
    c = (~cond_0)
    _alpha_w[c] = 0.0
    _beta_w[c] = 0.0
    _chi_w[c] = 0.0
    return 1

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], int64[:], int64[:]),
      cache=True)
def numba_solve_weir_flows(_Hw, _Qw, H_j, _z_inv_j, _z_w, _y_max_w, u, _L_w,
                           _s_w, _Cwr, _Cwt, _J_uw, _J_dw):
    _H_uw = H_j[_J_uw]
    _H_dw = H_j[_J_dw]
    _z_inv_uw = _z_inv_j[_J_uw]
    # Create indicator functions
    _omega_w = np.zeros(_H_uw.size)
    _omega_w[_H_uw >= _H_dw] = 1.0
    # Create conditionals
    cond_0 = (_omega_w * _H_uw + (1 - _omega_w) * _H_dw >
                _z_w + _z_inv_uw + (1 - u) * _y_max_w)
    cond_1 = ((1 - _omega_w) * _H_uw + _omega_w * _H_dw >
                _z_w + _z_inv_uw + (1 - u) * _y_max_w)
    # TODO: Is this being recalculated for a reason?
    # Effective heads
    a = (cond_0 & cond_1)
    b = (cond_0 & ~cond_1)
    c = (~cond_0)
    _Hw[a] = _H_uw[a] - _H_dw[a]
    _Hw[b] = (_omega_w[b] * _H_uw[b] + (1 - _omega_w[b]) * _H_dw[b]
                    + (-_z_inv_uw[b] - _z_w[b] - (1 - u[b]) * _y_max_w[b]))
    _Hw[c] = 0.0
    _Hw = np.abs(_Hw)
    # Compute universal coefficient
    _gamma_ww = (_Cwr * _L_w * _Hw + _Cwt * _s_w * _Hw**2)**2
    # Compute flow
    _Qw_next = (-1)**(1 - _omega_w) * np.sqrt(_gamma_ww * _Hw)
    return _Qw_next

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            int64[:], int64[:]),
      cache=True)
def numba_pump_flow_coefficients(_alpha_p, _beta_p, _chi_p, H_j, _z_inv_j, _Qp, u,
                                 _z_p, _dHp_max, _dHp_min, _a_p, _b_p, _c_p,
                                 _J_up, _J_dp):
    # Get upstream and downstream heads and invert elevation
    _H_up = H_j[_J_up]
    _H_dp = H_j[_J_dp]
    _z_inv_up = _z_inv_j[_J_up]
    # Compute effective head
    _dHp = _H_dp - _H_up
    # Condition 0: Upstream head is above inlet height
    cond_0 = _H_up > _z_inv_up + _z_p
    # Condition 1: Head difference is within range of pump curve
    cond_1 = (_dHp > _dHp_min) & (_dHp < _dHp_max)
    _dHp[_dHp > _dHp_max] = _dHp_max[_dHp > _dHp_max]
    _dHp[_dHp < _dHp_min] = _dHp_min[_dHp < _dHp_min]
    # Compute universal coefficients
    _gamma_p = gamma_p(_Qp, _b_p, _c_p, u)
    # Fill coefficient arrays
    # Head in pump curve range
    a = (cond_0 & cond_1)
    _alpha_p[a] = _gamma_p[a]
    _beta_p[a] = -_gamma_p[a]
    _chi_p[a] = _gamma_p[a] * _a_p[a]
    # Head outside of pump curve range
    b = (cond_0 & ~cond_1)
    _alpha_p[b] = 0.0
    _beta_p[b] = 0.0
    _chi_p[b] = _gamma_p[b] * (_a_p[b] - _dHp[b])
    # Depth below inlet
    c = (~cond_0)
    _alpha_p[c] = 0.0
    _beta_p[c] = 0.0
    _chi_p[c] = 0.0
    return 1

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
                 float64[:], float64[:], float64[:], int64[:], int64[:]),
      cache=True)
def numba_solve_pump_flows(H_j, u, _z_inv_j, _z_p, _dHp_max, _dHp_min, _a_p, _b_p, _c_p,
                           _J_up, _J_dp):
    _H_up = H_j[_J_up]
    _H_dp = H_j[_J_dp]
    _z_inv_up = _z_inv_j[_J_up]
    # Create conditionals
    _dHp = _H_dp - _H_up
    _dHp[_dHp > _dHp_max] = _dHp_max[_dHp > _dHp_max]
    _dHp[_dHp < _dHp_min] = _dHp_min[_dHp < _dHp_min]
    cond_0 = _H_up > _z_inv_up + _z_p
    # Compute universal coefficients
    _Qp_next = (u / _b_p * (_a_p - _dHp))**(1 / _c_p)
    _Qp_next[~cond_0] = 0.0
    return _Qp_next

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], int64, int64[:], int64[:], int64[:]),
      cache=True)
def tnumba_forward_recurrence(_T_ik, _U_Ik, _V_Ik, _W_Ik, _a_ik, _b_ik, _c_ik,
                             _P_ik, _E_Ik, _D_Ik, NK, nk, _I_1k, _i_1k):
    for k in range(NK):
        # Start at junction 1
        _I_1 = _I_1k[k]
        _i_1 = _i_1k[k]
        _I_2 = _I_1 + 1
        _i_2 = _i_1 + 1
        nlinks = nk[k]
        _T_ik[_i_1] = T_1k(_b_ik[_i_1])
        _U_Ik[_I_1] = U_1k(_c_ik[_i_1], _T_ik[_i_1])
        _V_Ik[_I_1] = V_1k(_P_ik[_i_1], _T_ik[_i_1])
        _W_Ik[_I_1] = W_1k(_a_ik[_i_1], _T_ik[_i_1])
        # Loop from junction 2 -> Nk
        for i in range(nlinks - 1):
            _i_next = _i_2 + i
            _I_next = _I_2 + i
            _Im1_next = _I_next - 1
            _T_ik[_i_next] = T_ik(_a_ik[_i_next], _b_ik[_i_next], _E_Ik[_I_next], _U_Ik[_Im1_next])
            _U_Ik[_I_next] = U_Ik(_E_Ik[_I_next], _c_ik[_i_next],
                                  _U_Ik[_Im1_next], _T_ik[_i_next])
            _V_Ik[_I_next] = V_Ik(_P_ik[_i_next], _a_ik[_i_next], _D_Ik[_I_next],
                                  _E_Ik[_I_next], _V_Ik[_Im1_next], _U_Ik[_Im1_next],
                                  _T_ik[_i_next])
            _W_Ik[_I_next] = W_Ik(_a_ik[_i_next], _W_Ik[_Im1_next], _T_ik[_i_next])
    return 1

@njit(int64(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
            float64[:], float64[:], float64[:], int64, int64[:], int64[:], int64[:]),
      cache=True)
def tnumba_backward_recurrence(_O_ik, _X_Ik, _Y_Ik, _Z_Ik, _a_ik, _b_ik, _c_ik,
                              _P_ik, _E_Ik, _D_Ik, NK, nk, _I_Nk, _i_nk):
    for k in range(NK):
        _I_N = _I_Nk[k]
        _i_n = _i_nk[k]
        _I_Nm1 = _I_N - 1
        _i_nm1 = _i_n - 1
        _I_Np1 = _I_N + 1
        nlinks = nk[k]
        _O_ik[_i_n] = O_nk(_b_ik[_i_n])
        _X_Ik[_I_N] = X_Nk(_a_ik[_i_n], _O_ik[_i_n])
        _Y_Ik[_I_N] = Y_Nk(_P_ik[_i_n], _O_ik[_i_n])
        _Z_Ik[_I_N] = Z_Nk(_c_ik[_i_n], _O_ik[_i_n])
        for i in range(nlinks - 1):
            _i_next = _i_nm1 - i
            _I_next = _I_Nm1 - i
            _Ip1_next = _I_next + 1
            _O_ik[_i_next] = O_ik(_b_ik[_i_next], _c_ik[_i_next], _E_Ik[_Ip1_next], _X_Ik[_Ip1_next])
            _X_Ik[_I_next] = X_Ik(_a_ik[_i_next], _E_Ik[_Ip1_next], _X_Ik[_Ip1_next], _O_ik[_i_next])
            _Y_Ik[_I_next] = Y_Ik(_P_ik[_i_next], _c_ik[_i_next], _D_Ik[_Ip1_next], 
                                  _E_Ik[_Ip1_next], _Y_Ik[_Ip1_next], _X_Ik[_Ip1_next],
                                  _O_ik[_i_next])
            _Z_Ik[_I_next] = Z_Ik(_c_ik[_i_next], _Z_Ik[_Ip1_next], _O_ik[_i_next])
    return 1

@njit(float64[:,:](float64[:,:], int64, int64),
      cache=True)
def numba_create_banded(l, bandwidth, M):
    AB = np.zeros((2*bandwidth + 1, M))
    for i in range(M):
        AB[bandwidth, i] = l[i, i]
    for n in range(bandwidth):
        for j in range(M - n - 1):
            AB[bandwidth - n - 1, -j - 1] = l[-j - 2 - n, -j - 1]
            AB[bandwidth + n + 1, j] = l[j + n + 1, j]
    return AB

@njit(void(float64[:], int64[:], float64[:]),
      cache=True)
def numba_add_at(a, indices, b):
    n = len(indices)
    for k in range(n):
        i = indices[k]
        a[i] += b[k]

@njit(void(float64[:, :], boolean[:], int64[:], int64[:], int64),
      cache=True)
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

def numba_B_j(_B_j, _J_uk, _J_dk, _xi_uk, _xi_dk, _A_sj):
    numba_add_at(_B_j, _J_uk, _xi_uk)
    numba_add_at(_B_j, _J_dk, _xi_dk)
    _B_j += _A_sj
    return _B_j

@njit(void(float64[:, :], boolean[:], float64[:], float64, int64),
      cache=True)
def numba_create_J_matrix(J, bc, _A_sj, _dt, M):
    for i in range(M):
        if bc[i]:
            J[i,i] = 1.0
        else:
            J[i,i] = (_A_sj[i] / _dt)

@njit(void(float64[:, :], float64[:], boolean[:], int64[:], int64[:], float64[:],
           float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
           float64, int64, int64),
      cache=True)
def numba_create_K_matrix(K, _F_jj, bc, _J_uk, _J_dk, _alpha_uk,
                          _alpha_dk, _beta_uk, _beta_dk, _xi_uk, _xi_dk,
                          _A_sj, _dt, M, NK):
    numba_add_at(_F_jj, _J_uk, _alpha_uk)
    numba_add_at(_F_jj, _J_dk, -_beta_dk)
    numba_add_at(_F_jj, _J_uk, _xi_uk / _dt)
    numba_add_at(_F_jj, _J_dk, _xi_dk / _dt)
    # Set diagonal of K matrix
    for i in range(M):
        if bc[i]:
            K[i,i] = 0.0
        else:
            K[i,i] = _F_jj[i]
    for k in range(NK):
        _J_u = _J_uk[k]
        _J_d = _J_dk[k]
        _bc_u = bc[_J_u]
        _bc_d = bc[_J_d]
        if not _bc_u:
            K[_J_u, _J_d] += _beta_uk[k]
        if not _bc_d:
            K[_J_d, _J_u] -= _alpha_dk[k]

@njit(void(float64[:, :], float64[:], boolean[:], int64[:], int64[:], float64[:],
           float64[:], float64[:], float64[:], int64, int64),
      cache=True)
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

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], int64[:], int64[:], int64),
      cache=True)
def numba_Q_i_next_b(X_Ik, h_Ik, Y_Ik, Z_Ik, h_Np1k, _Ik, _ki, n):
    _Q_i = np.zeros(n)
    for i in range(n):
        I = _Ik[i]
        k = _ki[i]
        t_0 = X_Ik[I] * h_Ik[I]
        t_1 = Y_Ik[I]
        t_2 = Z_Ik[I] * h_Np1k[k]
        _Q_i[i] = t_0 + t_1 + t_2
    return _Q_i

@njit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], int64[:], int64[:], int64),
      cache=True)
def numba_Q_im1k_next_f(U_Ik, h_Ik, V_Ik, W_Ik, h_1k, _Ik, _ki, n):
    _Q_i = np.zeros(n)
    for i in range(n):
        I = _Ik[i]
        Ip1 = I + 1
        k = _ki[i]
        t_0 = U_Ik[I] * h_Ik[Ip1]
        t_1 = V_Ik[I]
        t_2 = W_Ik[I] * h_1k[k]
        _Q_i[i] = t_0 + t_1 + t_2
    return _Q_i

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
           float64[:], float64[:], float64[:], float64[:], int64[:], int64[:], int64[:],
           int64[:], int64[:], int64, boolean[:]),
      cache=True)
def numba_reposition_junctions(_x_Ik, _z_inv_Ik, _h_Ik, _dx_ik, _Q_ik, _H_dk,
                               _b0, _zc, _xc, _m, _elem_pos, _i_1k, _I_1k,
                               _I_Np1k, nk, NK, reposition):
    for k in range(NK):
        if reposition[k]:
            _i_1 = _i_1k[k]
            _I_1 = _I_1k[k]
            _I_Np1 = _I_Np1k[k]
            nlinks = nk[k]
            njunctions = nlinks + 1
            _i_end = _i_1 + nlinks
            _I_end = _I_1 + njunctions
            _H_d = _H_dk[k]
            _z_inv_1 = _z_inv_Ik[_I_1]
            _z_inv_Np1 = _z_inv_Ik[_I_Np1]
            pos_prev = _elem_pos[k]
            # Junction arrays for superlink k
            _x_I = _x_Ik[_I_1:_I_end]
            _z_inv_I = _z_inv_Ik[_I_1:_I_end]
            _h_I = _h_Ik[_I_1:_I_end]
            _dx_i = _dx_ik[_i_1:_i_end]
            # Move junction if downstream head is within range
            move_junction = (_H_d > _z_inv_Np1) & (_H_d < _z_inv_1)
            if move_junction:
                z_m = _H_d
                _x0 = _x_I[_I_1]
                x_m = (_H_d - _b0[k]) / _m[k] + _x0
            else:
                z_m = _zc[k]
                x_m = _xc[k]
            # Determine new x-position of junction
            c = np.searchsorted(_x_I, x_m)
            cm1 = c - 1
            # Compute fractional x-position along superlink k
            frac = (x_m - _x_I[cm1]) / (_x_I[c] - _x_I[cm1])
            # Interpolate depth at new position
            h_m = (1 - frac) * _h_I[cm1] + (frac) * _h_I[c]
            # Link length ratio
            r = _dx_i[pos_prev - 1] / (_dx_i[pos_prev - 1]
                                    + _dx_i[pos_prev])
            # Set new values
            _x_I[pos_prev] = x_m
            _z_inv_I[pos_prev] = z_m
            _h_I[pos_prev] = h_m
            Ix = np.argsort(_x_I)
            _dx_i = np.diff(_x_I[Ix])
            _x_Ik[_I_1:_I_end] = _x_I[Ix]
            _z_inv_Ik[_I_1:_I_end] = _z_inv_I[Ix]
            _h_Ik[_I_1:_I_end] = _h_I[Ix]
            _dx_ik[_i_1:_i_end] = _dx_i
            # Set position to new position
            pos_change = np.argsort(Ix)
            pos_next = pos_change[pos_prev]
            _elem_pos[k] = pos_next
            shifted = (pos_prev != pos_next)
            # If position has shifted interpolate flow
            if shifted:
                ix = np.arange(nlinks)
                ix[pos_prev] = pos_next
                ix.sort()
                _Q_i = _Q_ik[_i_1:_i_end]
                _Q_i[pos_prev - 1] = (1 - r) * _Q_i[pos_prev - 1] + r * _Q_i[pos_prev]
                _Q_ik[_i_1:_i_end] = _Q_i[ix]

