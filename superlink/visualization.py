import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as patches

def _plot_superlink_profile(self, k, ax, x_offset=0, zorder=0):
    _Ik = np.flatnonzero(self._kI == k)
    _ik = np.flatnonzero(self._ki == k)
    nk = _ik.size
    x = self.x_Ik[_Ik] + x_offset
    h = self.h_Ik[_Ik]
    z = self._z_inv_Ik[_Ik]
    c = self._u_ik[_ik]

    #norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max(), clip=True)
    #mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)
    #color = mapper.to_rgba(c[j])
    color = 'c'

    ax.plot(x, z, c='0.25', alpha=0.75)
    for j in range(nk):
        xj = [x[j], x[j], x[j+1], x[j+1]]
        yj = [z[j], z[j] + h[j], z[j+1] + h[j+1], z[j+1]]
        pj = ax.add_patch(patches.Polygon(xy=list(zip(xj,yj)), linewidth=1,
                                          color=color, zorder=zorder))

def _plot_superjunction_profile(self, j, ax, x_offset=0, x_width=1, zorder=1):
    x = [x_offset, x_offset + x_width]
    H = self.H_j[j]
    z = self._z_inv_j[j]
    color = 'c'
    xj = [x[0], x[0], x[1], x[1]]
    yj = [z, H, H, z]
    pj = ax.add_patch(patches.Polygon(xy=list(zip(xj,yj)), linewidth=1,
                                      facecolor=color, edgecolor='0.25',
                                      zorder=zorder))

def plot_profile(self, js, ax, width=1, zorder=1):
    # Create mapping dict from superjunction pair to superlink
    jk = {}
    for k in range(self.NK):
        tup = (self.superlinks.loc[k, 'sj_0'], self.superlinks.loc[k, 'sj_1'])
        jk.update({tup : k})
    key_list = [(js[i], js[i+1]) for i in range(len(js) - 1)]
    x_offset = 0
    y_min = self.H_j.max()
    y_max = self.H_j.min()
    for key in key_list:
        j0, j1 = key
        k = jk[key]
        _plot_superjunction_profile(self, j0, ax=ax, x_offset=x_offset,
                                    x_width=width, zorder=zorder+1)
        x_offset += width
        _plot_superlink_profile(self, k, ax=ax, x_offset=x_offset, zorder=zorder)
        x_offset += self._dx_k[k]
        zmin = self._z_inv_Ik[self._kI == k].min()
        hmax = (self._z_inv_Ik[self._kI == k]
                + self._h_Ik[self._kI == k]).max()
        y_min = min(y_min, self._z_inv_j[j0], self._z_inv_j[j1], zmin)
        y_max = max(y_max, self.H_j[j0], self.H_j[j1], hmax)
    _plot_superjunction_profile(self, j1, ax=ax, x_offset=x_offset,
                                x_width=width, zorder=zorder+1)
    x_offset += width
    ax.set_xlim(0 - 0.02 * x_offset,
                x_offset + 0.02 * x_offset)
    ax.set_ylim(y_min - 0.02 * (y_max - y_min),
                y_max + 0.02 * (y_max - y_min))
