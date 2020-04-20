import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib import collections as mc
import mpl_toolkits.mplot3d.art3d as art3d

# TODO: vmin and vmax for superlinks/superjunctions inconsistent

def _plot_superlink_profile(self, k, ax, x_offset=0, facecolor='c',
                            cmap=cm.Blues_r, zorder=0, vrange=None, im=[]):
    _Ik = np.flatnonzero(self._kI == k)
    _ik = np.flatnonzero(self._ki == k)
    nk = _ik.size
    x = self.x_Ik[_Ik] + x_offset
    h = self.h_Ik[_Ik]
    z = self._z_inv_Ik[_Ik]
    if isinstance(facecolor, np.ndarray):
        if vrange is None:
            vmin, vmax = facecolor.min(), facecolor.max()
        else:
            vmin, vmax = vrange
        norm = matplotlib.colors.Normalize(vmin=vmin,
                                           vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        c = facecolor[_ik]
    else:
        mapper = None
    pj = ax.plot(x, z, c='0.25', alpha=0.75)
    # im.append(pj)
    for j in range(nk):
        xj = [x[j], x[j], x[j+1], x[j+1]]
        yj = [z[j], z[j] + h[j], z[j+1] + h[j+1], z[j+1]]
        if mapper is None:
            pj = ax.add_patch(patches.Polygon(xy=list(zip(xj,yj)), linewidth=1,
                                            color=facecolor, zorder=zorder))
            im.append(pj)
        else:
            pj = ax.add_patch(patches.Polygon(xy=list(zip(xj,yj)), linewidth=1,
                                            color=mapper.to_rgba(c[j]),
                                              zorder=zorder))
            im.append(pj)


def _plot_superjunction_profile(self, j, ax, x_offset=0, x_width=1, facecolor='c',
                                cmap=cm.Blues_r, zorder=1, vrange=None, im=[]):
    x = [x_offset, x_offset + x_width]
    H = self.H_j[j]
    z = self._z_inv_j[j]
    xj = [x[0], x[0], x[1], x[1]]
    yj = [z, H, H, z]
    if isinstance(facecolor, np.ndarray):
        if vrange is None:
            vmin, vmax = facecolor.min(), facecolor.max()
        else:
            vmin, vmax = vrange
        norm = matplotlib.colors.Normalize(vmin=vmin,
                                           vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        c = facecolor[j]
    else:
        mapper = None
    if mapper is None:
        pj = ax.add_patch(patches.Polygon(xy=list(zip(xj,yj)), linewidth=1,
                                        facecolor=facecolor, edgecolor='0.25',
                                        zorder=zorder))
        im.append(pj)
    else:
        pj = ax.add_patch(patches.Polygon(xy=list(zip(xj,yj)), linewidth=1,
                                        facecolor=mapper.to_rgba(c), edgecolor='0.25',
                                        zorder=zorder))
        im.append(pj)

def plot_profile(self, js, ax, width=1, sl_facecolor='c', sj_facecolor='c',
                 sl_cmap=cm.Blues_r, sj_cmap=cm.Blues_r, sl_vrange=None,
                 sj_vrange=None, zorder=1, ylim=None, xlim=None):
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
        k = jk[key]
        _plot_superjunction_profile(self, j0, ax=ax, x_offset=x_offset,
                                    x_width=width, facecolor=sj_facecolor,
                                    cmap=sj_cmap, zorder=zorder+1, vrange=sj_vrange, im=im)
        x_offset += width
        _plot_superlink_profile(self, k, ax=ax, x_offset=x_offset, facecolor=sl_facecolor,
                                cmap=sl_cmap, zorder=zorder, vrange=sl_vrange, im=im)
        x_offset += self._dx_k[k]
        zmin = self._z_inv_Ik[self._kI == k].min()
        hmax = (self._z_inv_Ik[self._kI == k]
                + self._h_Ik[self._kI == k]).max()
        y_min = min(y_min, self._z_inv_j[j0], self._z_inv_j[j1], zmin)
        y_max = max(y_max, self.H_j[j0], self.H_j[j1], hmax)
    _plot_superjunction_profile(self, j1, ax=ax, x_offset=x_offset,
                                x_width=width, facecolor=sj_facecolor,
                                cmap=sj_cmap, zorder=zorder+1, vrange=sj_vrange, im=im)
    x_offset += width
    if xlim is None:
        ax.set_xlim(0 - 0.02 * x_offset,
                    x_offset + 0.02 * x_offset)
    else:
        ax.set_xlim(*xlim)
    if ylim is None:
        ax.set_ylim(y_min - 0.02 * (y_max - y_min),
                    y_max + 0.02 * (y_max - y_min))
    else:
        ax.set_ylim(*ylim)
    return im

def plot_network_2d(self, ax, superjunction_kwargs={}, junction_kwargs={},
                    link_kwargs={}):
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
    ax.add_collection(lc_ik)
    return sc_j, sc_Ik, lc_ik

def plot_network_3d(self, ax, superjunction_signal=None, junction_signal=None,
                    superjunction_stems=True, junction_stems=True,
                    border=True, fill=True, base_line_kwargs={}, superjunction_stem_kwargs={},
                    junction_stem_kwargs={}, border_kwargs={}, fill_kwargs={}):
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
