import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.collections import PolyCollection

# TODO: vmin and vmax for superlinks/superjunctions inconsistent

# def _plot_superlink_profile(self, k, ax, x_offset=0, im=[], superlink_kwargs={}):
#     _Ik = np.flatnonzero(self._kI == k)
#     _ik = np.flatnonzero(self._ki == k)
#     nk = _ik.size
#     x = self.x_Ik[_Ik] + x_offset
#     h = self.h_Ik[_Ik]
#     z = self._z_inv_Ik[_Ik]
#     pj = ax.plot(x, z, c='0.25', alpha=0.75)
#     # im.append(pj)
#     for j in range(nk):
#         xj = [x[j], x[j], x[j+1], x[j+1]]
#         yj = [z[j], z[j] + h[j], z[j+1] + h[j+1], z[j+1]]
#         pj = patches.Polygon(xy=list(zip(xj,yj)), **superlink_kwargs)
#         ax.add_patch(pj)
#         im.append(pj)

def _plot_superlink_profile(self, k, ax, x_offset=0, im=[], superlink_kwargs={}):
    _I = self._kI == k
    _Ik = self._Ik[(self._kI == k)[self._Ik]]
    _Ip1k = self._Ip1k[(self._kI == k)[self._Ip1k]]
    _x_Ik = self.x_Ik[_Ik] + x_offset
    _x_Ip1k = self.x_Ik[_Ip1k] + x_offset
    _h_Ik = self.h_Ik[_Ik]
    _h_Ip1k = self.h_Ik[_Ip1k]
    _z_Ik = self._z_inv_Ik[_Ik]
    _z_Ip1k = self._z_inv_Ik[_Ip1k]
    _z = self._z_inv_Ik[_I]
    _x = self._x_Ik[_I] + x_offset
    pj = ax.plot(_x, _z, c='0.25', alpha=0.75)
    # im.append(pj)
    poly = np.dstack([np.column_stack([_x_Ik, _x_Ik, _x_Ip1k, _x_Ip1k]),
                      np.column_stack([_z_Ik, _z_Ik + _h_Ik,
                                       _z_Ip1k + _h_Ip1k, _z_Ip1k])])
    pj = PolyCollection(poly, **superlink_kwargs)
    ax.add_collection(pj)
    im.append(pj)

def _plot_superjunction_profile(self, j, ax, x_offset=0, x_width=1, im=[],
                                superjunction_kwargs={}):
    x = [x_offset, x_offset + x_width]
    H = self.H_j[j]
    z = self._z_inv_j[j]
    xj = [x[0], x[0], x[1], x[1]]
    yj = [z, H, H, z]
    pj = ax.plot(x, [z, z], c='0.25', alpha=0.75)
    pj = patches.Polygon(xy=list(zip(xj,yj)), **superjunction_kwargs)
    ax.add_patch(pj)
    im.append(pj)

def plot_profile(self, js, ax=None, width=1, superlink_kwargs={}, superjunction_kwargs={}):
    if ax is None:
        fig, ax = plt.subplots()
    # Create mapping dict from superjunction pair to superlink
    jk = {}
    for k in range(self.NK):
        tup = (self.superlinks.loc[k, 'sj_0'], self.superlinks.loc[k, 'sj_1'])
        jk.update({tup : k})
    key_list = [(js[i], js[i+1]) for i in range(len(js) - 1)]
    x_offset = 0
    y_min = self.H_j.max()
    y_max = self.H_j.min()
    im = []
    for key in key_list:
        j0, j1 = key
        _plot_superjunction_profile(self, j0, ax=ax, x_offset=x_offset,
                                    x_width=width, im=im,
                                    superjunction_kwargs=superjunction_kwargs)
        x_offset += width
        if key in jk:
            k = jk[key]
            _plot_superlink_profile(self, k, ax=ax, x_offset=x_offset, im=im,
                                    superlink_kwargs=superlink_kwargs)
            x_offset += self._dx_k[k]
            zmin = self._z_inv_Ik[self._kI == k].min()
            hmax = (self._z_inv_Ik[self._kI == k]
                    + self._h_Ik[self._kI == k]).max()
            y_min = min(y_min, self._z_inv_j[j0], self._z_inv_j[j1], zmin)
            y_max = max(y_max, self.H_j[j0], self.H_j[j1], hmax)
    _plot_superjunction_profile(self, j1, ax=ax, x_offset=x_offset,
                                x_width=width, im=im,
                                superjunction_kwargs=superjunction_kwargs)
    x_offset += width
    ax.set_xlim(0 - 0.02 * x_offset,
                x_offset + 0.02 * x_offset)
    ax.set_ylim(y_min - 0.02 * (y_max - y_min),
                y_max + 0.02 * (y_max - y_min))
    ax.set_ylabel('Elevation (m)')
    ax.set_xlabel('Horizontal coordinate (m)')
    return im

def plot_network_2d(self, ax=None, superjunction_kwargs={}, junction_kwargs={},
                    link_kwargs={}, orifice_kwargs={}, weir_kwargs={}, pump_kwargs={}):
    if ax is None:
        fig, ax = plt.subplots()
    collections = []
    _map_x_j = self._map_x_j
    _map_y_j = self._map_y_j
    _x_Ik = self._x_Ik
    _dx_k = self._dx_k
    _kI = self._kI
    _J_uk = self._J_uk
    _J_dk = self._J_dk
    _Ik = self._Ik
    _Ip1k = self._Ip1k
    frac_pos = _x_Ik / _dx_k[_kI]
    _map_x_Ik = frac_pos * (_map_x_j[_J_dk] - _map_x_j[_J_uk])[_kI] + _map_x_j[_J_uk][_kI]
    _map_y_Ik = frac_pos * (_map_y_j[_J_dk] - _map_y_j[_J_uk])[_kI] + _map_y_j[_J_uk][_kI]
    lines = np.dstack([np.column_stack([_map_x_Ik[_Ik], _map_x_Ik[_Ip1k]]),
                       np.column_stack([_map_y_Ik[_Ik], _map_y_Ik[_Ip1k]])])
    sc_Ik = ax.scatter(_map_x_Ik, _map_y_Ik, **junction_kwargs)
    lc_ik = mc.LineCollection(lines, **link_kwargs)
    sc_j = ax.scatter(_map_x_j, _map_y_j, **superjunction_kwargs)
    collections.extend([sc_Ik, lc_ik, sc_j])
    ax.add_collection(lc_ik)
    if self.n_o:
        _J_uo = self._J_uo
        _J_do = self._J_do
        lines = np.dstack([np.column_stack([_map_x_j[_J_uo], _map_x_j[_J_do]]),
                           np.column_stack([_map_y_j[_J_uo], _map_y_j[_J_do]])])
        lc_o = mc.LineCollection(lines, **orifice_kwargs)
        ax.add_collection(lc_o)
        collections.append(lc_o)
    if self.n_w:
        _J_uw = self._J_uw
        _J_dw = self._J_dw
        lines = np.dstack([np.column_stack([_map_x_j[_J_uw], _map_x_j[_J_dw]]),
                           np.column_stack([_map_y_j[_J_uw], _map_y_j[_J_dw]])])
        lc_w = mc.LineCollection(lines, **weir_kwargs)
        ax.add_collection(lc_w)
        collections.append(lc_w)
    if self.n_p:
        _J_up = self._J_up
        _J_dp = self._J_dp
        lines = np.dstack([np.column_stack([_map_x_j[_J_up], _map_x_j[_J_dp]]),
                           np.column_stack([_map_y_j[_J_up], _map_y_j[_J_dp]])])
        lc_p = mc.LineCollection(lines, **pump_kwargs)
        ax.add_collection(lc_p)
        collections.append(lc_p)
    return collections

def plot_network_3d(self, ax=None, superjunction_signal=None, junction_signal=None,
                    superjunction_stems=True, junction_stems=True,
                    border=True, fill=True, base_line_kwargs={}, superjunction_stem_kwargs={},
                    junction_stem_kwargs={}, border_kwargs={}, fill_kwargs={},
                    orifice_kwargs={}, weir_kwargs={}, pump_kwargs={}):
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    _map_x_j = self._map_x_j
    _map_y_j = self._map_y_j
    _x_Ik = self._x_Ik
    _dx_k = self._dx_k
    _kI = self._kI
    _J_uk = self._J_uk
    _J_dk = self._J_dk
    _Ik = self._Ik
    _Ip1k = self._Ip1k
    _z_inv_Ik = self._z_inv_Ik
    frac_pos = _x_Ik / _dx_k[_kI]
    _z_inv_j = self._z_inv_j
    if superjunction_signal is None:
        superjunction_signal = self.H_j - _z_inv_j
    if junction_signal is None:
        junction_signal = self.h_Ik
    _map_x_Ik = frac_pos * (_map_x_j[_J_dk] - _map_x_j[_J_uk])[_kI] + _map_x_j[_J_uk][_kI]
    _map_y_Ik = frac_pos * (_map_y_j[_J_dk] - _map_y_j[_J_uk])[_kI] + _map_y_j[_J_uk][_kI]
    collections = []
    base = np.dstack([np.column_stack([_map_x_Ik[_Ik], _map_x_Ik[_Ip1k]]),
                    np.column_stack([_map_y_Ik[_Ik], _map_y_Ik[_Ip1k]]),
                    np.column_stack([_z_inv_Ik[_Ik], _z_inv_Ik[_Ip1k]])])
    lc_z = art3d.Line3DCollection(base, **base_line_kwargs)
    ax.add_collection3d(lc_z)
    collections.append(lc_z)
    if self.n_o:
        _J_uo = self._J_uo
        _J_do = self._J_do
        lines = np.dstack([np.column_stack([_map_x_j[_J_uo], _map_x_j[_J_do]]),
                           np.column_stack([_map_y_j[_J_uo], _map_y_j[_J_do]]),
                           np.column_stack([_z_inv_j[_J_uo], _z_inv_j[_J_do]])])
        lc_o = art3d.Line3DCollection(lines, **orifice_kwargs)
        ax.add_collection3d(lc_o)
        collections.append(lc_o)
    if self.n_w:
        _J_uw = self._J_uw
        _J_dw = self._J_dw
        lines = np.dstack([np.column_stack([_map_x_j[_J_uw], _map_x_j[_J_dw]]),
                           np.column_stack([_map_y_j[_J_uw], _map_y_j[_J_dw]]),
                           np.column_stack([_z_inv_j[_J_uw], _z_inv_j[_J_dw]])])
        lc_w = art3d.Line3DCollection(lines, **weir_kwargs)
        ax.add_collection3d(lc_w)
        collections.append(lc_w)
    if self.n_p:
        _J_up = self._J_up
        _J_dp = self._J_dp
        lines = np.dstack([np.column_stack([_map_x_j[_J_up], _map_x_j[_J_dp]]),
                           np.column_stack([_map_y_j[_J_up], _map_y_j[_J_dp]]),
                           np.column_stack([_z_inv_j[_J_up], _z_inv_j[_J_dp]])])
        lc_p = art3d.Line3DCollection(lines, **pump_kwargs)
        ax.add_collection3d(lc_p)
        collections.append(lc_p)
    if superjunction_stems:
        stems = np.dstack([np.column_stack([_map_x_j, _map_x_j]),
                           np.column_stack([_map_y_j, _map_y_j]),
                           np.column_stack([_z_inv_j, _z_inv_j + superjunction_signal])])
        st_j = art3d.Line3DCollection(stems, **superjunction_stem_kwargs)
        ax.add_collection3d(st_j)
        collections.append(st_j)
    if junction_stems:
        stems = np.dstack([np.column_stack([_map_x_Ik, _map_x_Ik]),
                        np.column_stack([_map_y_Ik, _map_y_Ik]),
                        np.column_stack([_z_inv_Ik, _z_inv_Ik + junction_signal])])
        st_h = art3d.Line3DCollection(stems, **junction_stem_kwargs)
        ax.add_collection3d(st_h)
        collections.append(st_h)
    if border:
        border_lines = np.dstack([np.column_stack([_map_x_Ik[_Ik], _map_x_Ik[_Ip1k]]),
                                np.column_stack([_map_y_Ik[_Ik], _map_y_Ik[_Ip1k]]),
                                np.column_stack([_z_inv_Ik[_Ik] + junction_signal[_Ik],
                                                _z_inv_Ik[_Ip1k] + junction_signal[_Ip1k]])])
        lc_h = art3d.Line3DCollection(border_lines, **border_kwargs)
        ax.add_collection3d(lc_h)
        collections.append(lc_h)
    if fill:
        poly = np.dstack([np.column_stack([_map_x_Ik[_Ik], _map_x_Ik[_Ik],
                                        _map_x_Ik[_Ip1k], _map_x_Ik[_Ip1k]]),
                        np.column_stack([_map_y_Ik[_Ik], _map_y_Ik[_Ik],
                                        _map_y_Ik[_Ip1k], _map_y_Ik[_Ip1k]]),
                        np.column_stack([_z_inv_Ik[_Ik], _z_inv_Ik[_Ik] + junction_signal[_Ik],
                                         _z_inv_Ik[_Ip1k] + junction_signal[_Ip1k],
                                         _z_inv_Ik[_Ip1k]])])
        poly_h = art3d.Poly3DCollection(poly, **fill_kwargs)
        ax.add_collection3d(poly_h)
        collections.append(poly_h)
    ax.set_xlim3d(_map_x_j.min(), _map_x_j.max())
    ax.set_ylim3d(_map_y_j.min(), _map_y_j.max())
    ax.set_zlim3d(min((_z_inv_Ik + junction_signal).min(),
                      (_z_inv_j + superjunction_signal).min()),
                  max((_z_inv_Ik + junction_signal).max(),
                      (_z_inv_j + superjunction_signal).max()))
    return collections
