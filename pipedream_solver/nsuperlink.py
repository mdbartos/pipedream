import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg
from numba import njit, prange
from numba.types import float64, int64, uint32, uint16, uint8, boolean, UniTuple, Tuple, List, DictType, void
import pipedream_solver.geometry
import pipedream_solver.ngeometry
import pipedream_solver.storage
from pipedream_solver.superlink import SuperLink
from pipedream_solver._nsuperlink import *

class nSuperLink(SuperLink):
    """
    Numba implementation of SUPERLINK hydraulic solver, as described in:

    Ji, Z. (1998). General Hydrodynamic Model for Sewer/Channel Network Systems.
    Journal of Hydraulic Engineering, 124(3), 307–315.
    doi: 10.1061/(asce)0733-9429(1998)124:3(307)

    Inputs:
    -----------
    superlinks: pd.DataFrame
        Table containing all superlinks in the network along with their attributes.
        The following fields are required:

        |------------+-------+------+-----------------------------------------------------------|
        | Field      | Type  | Unit | Description                                               |
        |------------+-------+------+-----------------------------------------------------------|
        | id         | int   |      | Integer id for the superlink                              |
        | name       | str   |      | Name of the superlink                                     |
        | sj_0       | int   |      | Index of the upstream superjunction                       |
        | sj_1       | int   |      | Index of the downstream superjunction                     |
        | in_offset  | float | m    | Offset of superlink invert above upstream superjunction   |
        | out_offset | float | m    | Offset of superlink invert above downstream superjunction |
        | C_uk       | float | -    | Upstream discharge coefficient                            |
        | C_dk       | float | -    | Downstream discharge coefficient                          |
        |------------+-------+------+-----------------------------------------------------------|

        If internal links and junctions are provided (arguments 3 and 4), the following
        fields are required:

        |-------+------+------+------------------------------------------|
        | Field | Type | Unit | Description                              |
        |-------+------+------+------------------------------------------|
        | j_0   | int  |      | Index of first junction inside superlink |
        | j_1   | int  |      | Index of last junction inside superlink  |
        |-------+------+------+------------------------------------------|

        If internal links and junctions are not provided (arguments 3 and 4), the following
        fields are required:

        |-------+-------+-------+------------------------------------------------------|
        | Field | Type  | Unit  | Description                                          |
        |-------+-------+-------+------------------------------------------------------|
        | dx    | float | m     | Length of superlink                                  |
        | n     | float | -     | Manning's roughness coefficient for superlink        |
        | shape | str   |       | Cross-sectional geometry type (see geometry module)  |
        | g1    | float | m     | First dimension of cross-sectional geometry          |
        | g2    | float | m     | Second dimension of cross-sectional geometry         |
        | g3    | float | m     | Third dimension of cross-sectional geometry          |
        | g4    | float | m     | Fourth dimension of cross-sectional geometry         |
        | Q_0   | float | m^3/s | Initial flow in internal links                       |
        | h_0   | float | m     | Initial depth in internal junctions                  |
        | A_s   | float | m     | Surface area of internal junctions                   |
        | ctrl  | bool  |       | Indicates presence of control structure in superlink |
        | A_c   | float | m^2   | Cross-sectional area of internal control structure   |
        | C     | float | -     | Discharge coefficient of internal control structure  |
        |-------+-------+-------+------------------------------------------------------|

    superjunctions: pd.DataFrame
        Table containing all superjunctions in the network along with their attributes.
        The following fields are required:

        |-----------+-------+------+-------------------------------------------------------|
        | Field     | Type  | Unit | Description                                           |
        |-----------+-------+------+-------------------------------------------------------|
        | id        | int   |      | Integer id for superjunction                          |
        | name      | str   |      | Name of superjunction                                 |
        | z_inv     | float | m    | Elevation of bottom of superjunction                  |
        | h_0       | float | m    | Initial depth in superjunction                        |
        | bc        | bool  |      | Indicates boundary condition at superjunction         |
        | storage   | str   |      | Storage type: `functional` or `tabular`               |
        | a         | float | m    | `a` value in function relating surface area and depth |
        | b         | float | -    | `b` value in function relating surface area and depth |
        | c         | float | m^2  | `c` value in function relating surface area and depth |
        | max_depth | float | m    | Maximum depth allowed at superjunction                |
        |-----------+-------+------+-------------------------------------------------------|

    links: pd.DataFrame (optional)
        Table containing all links in the network along with their attributes.
        Note that if links and junction are not supplied, they will automatically be
        generated at even intervals within each superlink.
        The following fields are required:

        |-------+-------+-------+-----------------------------------------------------|
        | Field | Type  | Unit  | Description                                         |
        |-------+-------+-------+-----------------------------------------------------|
        | j_0   | int   |       | Index of upstream junction                          |
        | j_1   | int   |       | Index of downstream junction                        |
        | k     | int   |       | Index of containing superlink                       |
        | dx    | float | m     | Length of link                                      |
        | n     | float | -     | Manning's roughness coefficient for link            |
        | shape | str   |       | Cross-sectional geometry type (see geometry module) |
        | g1    | float | m     | First dimension of cross-sectional geometry         |
        | g2    | float | m     | Second dimension of cross-sectional geometry        |
        | g3    | float | m     | Third dimension of cross-sectional geometry         |
        | g4    | float | m     | Fourth dimension of cross-sectional geometry        |
        | Q_0   | float | m^3/s | Initial flow in internal links                      |
        | h_0   | float | m     | Initial depth in internal junctions                 |
        | A_s   | float | m     | Surface area of internal junctions                  |
        | ctrl  | bool  |       | Indicates presence of control structure in link     |
        | A_c   | float | m^2   | Cross-sectional area of internal control structure  |
        | C     | float | -     | Discharge coefficient of internal control structure |
        |-------+-------+-------+-----------------------------------------------------|

    junctions: pd.DataFrame (optional)
        Table containing all junctions in the network along with their attributes.
        Note that if links and junction are not supplied, they will automatically be
        generated at even intervals within each superlink.
        The following fields are required:

        |-------+-------+------+-------------------------------|
        | Field | Type  | Unit | Description                   |
        |-------+-------+------+-------------------------------|
        | id    | int   |      | Integer id for junction       |
        | k     | int   |      | Index of containing superlink |
        | h_0   | float | m    | Initial depth at junction     |
        | A_s   | float | m^2  | Surface area of junction      |
        | z_inv | float | m    | Invert elevation of junction  |
        |-------+-------+------+-------------------------------|

    transects: dict (optional)
        Dictionary describing nonfunctional channel cross-sectional geometries.
        Takes the following structure:

        {
            <transect_name> :
                {
                    'x' : <x-coordinates of cross-section (list of floats)>,
                    'y' : <y-coordinates of cross-section (list of floats)>,
                    'horiz_points' : <Number of horizontal sampling points (int)>
                    'vert_points' : <Number of vertical sampling points (int)>
                }
            ...
        }

    storages: dict (optional)
        Dictionary describing tabular storages for superjunctions.
        Takes the following structure:

        {
            <storage_name> :
                {
                    'h' : <Depths (list of floats)>,
                    'A' : <Surface areas associated with depths (list of floats)>,
                }
            ...
        }

    orifices: pd.DataFrame (optional)
        Table containing orifice control structures, and their attributes.
        The following fields are required:

        |-------------+-------+------+------------------------------------------------------|
        | Field       | Type  | Unit | Description                                          |
        |-------------+-------+------+------------------------------------------------------|
        | id          | int   |      | Integer id for the orifice                           |
        | name        | str   |      | Name of the orifice                                  |
        | sj_0        | int   |      | Index of the upstream superjunction                  |
        | sj_1        | int   |      | Index of the downstream superjunction                |
        | orientation | str   |      | Orifice orientation: `bottom` or `side`              |
        | C           | float | -    | Discharge coefficient for orifice                    |
        | A           | float | m^2  | Full area of orifice                                 |
        | y_max       | float | m    | Full height of orifice                               |
        | z_o         | float | m    | Offset of bottom above upstream superjunction invert |
        |-------------+-------+------+------------------------------------------------------|

    weirs: pd.DataFrame (optional)
        Table containing weir control structures, and their attributes.
        The following fields are required:

        |-------+-------+------+------------------------------------------------------|
        | Field | Type  | Unit | Description                                          |
        |-------+-------+------+------------------------------------------------------|
        | id    | int   |      | Integer id for the weir                              |
        | name  | str   |      | Name of the weir                                     |
        | sj_0  | int   |      | Index of the upstream superjunction                  |
        | sj_1  | int   |      | Index of the downstream superjunction                |
        | z_w   | float | m    | Offset of bottom above upstream superjunction invert |
        | y_max | float | m    | Full height of weir                                  |
        | C_r   | float | -    | Discharge coefficient for rectangular portion        |
        | C_t   | float | -    | Discharge coefficient for triangular portions        |
        | L     | float | m    | Length of rectangular portion of weir                |
        | s     | float | -    | Inverse slope of triangular portion of weir          |
        |-------+-------+------+------------------------------------------------------|

    pumps: pd.DataFrame (optional)
        Table containing pump control structures and their attributes.
        The following fields are required:

        |--------+-------+------+------------------------------------------------------|
        | Field  | Type  | Unit | Description                                          |
        |--------+-------+------+------------------------------------------------------|
        | id     | int   |      | Integer id for the pump                              |
        | name   | str   |      | Name of the pump                                     |
        | sj_0   | int   |      | Index of the upstream superjunction                  |
        | sj_1   | int   |      | Index of the downstream superjunction                |
        | z_p    | float | m    | Offset of bottom above upstream superjunction invert |
        | a_q    | float |      | Vertical coefficient of pump ellipse                 |
        | a_h    | float |      | Horizontal coefficient of pump ellipse               |
        | dH_min | float | m    | Minimum pump head                                    |
        | dH_max | float | m    | Maximum pump head                                    |
        |--------+-------+------+------------------------------------------------------|

    dt: float
        Default timestep of model (in seconds).

    min_depth: float
        Minimum depth allowed at junctions and superjunctions (in meters).

    method: str
        Method for computing internal states in superlinks. Must be one of the following:
        - `b`   : Backwards (default)
        - `f`   : Forwards
        - `lsq` : Least-squares

    auto_permute: bool
        If True, permute the superjunctions to enable use of a banded matrix solver and
        increase solver speed. Superjunctions are permuted using the Reverse
        Cuthill-McKee algorithm.

    internal_links: int
        If junctions/links are not provided, this gives the number of internal
        links that will be generated inside each superlink.

    sparse: bool
        (Deprecated)

    bc_method: str
        (Deprecated)

    exit_hydraulics: bool
        (Deprecated)

    end_length: float
        (Deprecated)

    bc_method: str
        (Deprecated)

    end_method: str
        (Deprecated)

    Methods:
    -----------
    step : Advance model to next time step, computing hydraulic states
    save_state : Save current model state
    load_state : Load model state

    Attributes:
    -----------
    t        : Current time (s)
    H_j      : Superjunction heads (m)
    h_Ik     : Junction depths (m)
    Q_ik     : Link flows (m^3/s)
    Q_uk     : Flows into upstream ends of superlinks (m^3/s)
    Q_dk     : Flows into downstream ends of superlinks (m^3/s)
    Q_o      : Orifice flows (m^3/s)
    Q_w      : Weir flows (m^3/s)
    Q_p      : Pump flows (m^3/s)
    A_ik     : Cross-sectional area of flow in links (m^2)
    Pe_ik    : Wetted perimeter in links (m)
    R_ik     : Hydraulic radius in links (m)
    B_ik     : Top width of flow in links (m)
    A_sj     : Superjunction surface areas (m^2)
    V_sj     : Superjunction stored volumes (m^3)
    z_inv_j  : Superjunction invert elevation (m)
    z_inv_uk : Offset of superlink upstream invert above superjunction (m)
    z_inv_dk : Offset of superlink downstream invert above superjunction (m)
    """
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

    def configure_storages(self):
        """
        Prepare data structures for computation of superjunction storage.
        """
        # Import instance variables
        storages = self.storages                # Table of storages
        _storage_type = self._storage_type      # Type of storage (functional/tabular)
        _storage_table = self._storage_table    # Tabular storages
        _storage_factory = {}
        _storage_indices = None
        _storage_hs = np.array([])
        _storage_As = np.array([])
        _storage_Vs = np.array([])
        _storage_inds = np.array([])
        _storage_lens = np.array([])
        _storage_js = np.array([])
        _storage_codes = np.array([])
        # Separate storages into functional and tabular
        _functional = (_storage_type.str.lower() == 'functional').values
        _tabular = (_storage_type.str.lower() == 'tabular').values
        # All entries must either be function or tabular
        assert (_tabular.sum() + _functional.sum()) == _storage_type.shape[0]
        # Configure tabular storages
        if storages:
            _tabular_storages = _storage_table[_tabular]
            _storage_indices = pd.Series(_tabular_storages.index, _tabular_storages.values)
            unique_storage_names = np.unique(_storage_indices.index.values)
            storage_name_to_ind = pd.Series(np.arange(unique_storage_names.size),
                                            index=unique_storage_names)
            sj_to_storage_ind = _storage_table.dropna().map(storage_name_to_ind)
            _storage_inds = []
            _storage_lens = []
            _storage_As = []
            _storage_Vs = []
            _storage_hs = []
            order = []
            ix = 0
            for name, storage in storages.items():
                A = storage['A']
                h = storage['h']
                V = scipy.integrate.cumtrapz(h, A, initial=0.)
                _storage_As.append(A)
                _storage_Vs.append(V)
                _storage_hs.append(h)
                _storage_inds.append(ix)
                order.append(storage_name_to_ind[name])
                ix += len(h)
                _storage_lens.append(len(h))
            order = np.argsort(order)
            _storage_hs = np.concatenate([_storage_hs[i] for i in order])
            _storage_As = np.concatenate([_storage_As[i] for i in order])
            _storage_Vs = np.concatenate([_storage_Vs[i] for i in order])
            _storage_inds = np.asarray(_storage_inds)[order]
            _storage_lens = np.asarray(_storage_lens)[order]
            _storage_js = sj_to_storage_ind.index.values
            _storage_codes = sj_to_storage_ind.values
        # Export instance variables
        self._storage_indices = _storage_indices
        self._storage_factory = _storage_factory
        self._storage_hs = _storage_hs
        self._storage_As = _storage_As
        self._storage_Vs = _storage_Vs
        self._storage_inds = _storage_inds
        self._storage_lens = _storage_lens
        self._storage_js = _storage_js
        self._storage_codes = _storage_codes
        self._functional = _functional
        self._tabular = _tabular

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
        _g4_ik = self._g4_ik           # Geometry 4 of link ik (other)
        _g5_ik = self._g5_ik           # Geometry 5 of link ik (other)
        _g6_ik = self._g6_ik           # Geometry 6 of link ik (other)
        _g7_ik = self._g7_ik           # Geometry 7 of link ik (other)
        _geom_codes = self._geom_codes
        _ellipse_ix = self._ellipse_ix
        _transect_factory = self._transect_factory
        _transect_indices = self._transect_indices
        _has_irregular = self._has_irregular
        # Compute hydraulic geometry for regular geometries
        # NOTE: Handle case for elliptical perimeter first
        handle_elliptical_perimeter(_Pe_ik, _ellipse_ix, _Ik, _Ip1k, _h_Ik,
                                   _g1_ik, _g2_ik)
        # Compute hydraulic geometries for all other regular geometries
        numba_hydraulic_geometry(_A_ik, _Pe_ik, _R_ik, _B_ik, _h_Ik,
                                 _g1_ik, _g2_ik, _g3_ik, _g4_ik, _g5_ik, _g6_ik, _g7_ik,
                                 _geom_codes, _Ik, _ik)
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
        _Pe_uk = self._Pe_uk
        _R_uk = self._R_uk
        _dx_ik = self._dx_ik           # Length of link ik
        _g1_ik = self._g1_ik           # Geometry 1 of link ik (vertical)
        _g2_ik = self._g2_ik           # Geometry 2 of link ik (horizontal)
        _g3_ik = self._g3_ik           # Geometry 3 of link ik (other)
        _g4_ik = self._g4_ik           # Geometry 4 of link ik (other)
        _g5_ik = self._g5_ik           # Geometry 5 of link ik (other)
        _g6_ik = self._g6_ik           # Geometry 6 of link ik (other)
        _g7_ik = self._g7_ik           # Geometry 7 of link ik (other)        
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
        numba_boundary_geometry(_A_uk, _Pe_uk, _R_uk, _B_uk, _h_Ik, H_j, _z_inv_uk,
                                _g1_ik, _g2_ik, _g3_ik, _g4_ik, _g5_ik, _g6_ik, _g7_ik,
                                _geom_codes, _i_1k, _I_1k, _J_uk)
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
        _Pe_dk = self._Pe_dk
        _R_dk = self._R_dk
        _dx_ik = self._dx_ik           # Length of link ik
        _g1_ik = self._g1_ik           # Geometry 1 of link ik (vertical)
        _g2_ik = self._g2_ik           # Geometry 2 of link ik (horizontal)
        _g3_ik = self._g3_ik           # Geometry 3 of link ik (other)
        _g4_ik = self._g4_ik           # Geometry 4 of link ik (other)
        _g5_ik = self._g5_ik           # Geometry 5 of link ik (other)
        _g6_ik = self._g6_ik           # Geometry 6 of link ik (other)
        _g7_ik = self._g7_ik           # Geometry 7 of link ik (other)        
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
        numba_boundary_geometry(_A_dk, _Pe_dk, _R_dk, _B_dk, _h_Ik, H_j, _z_inv_dk,
                                _g1_ik, _g2_ik, _g3_ik, _g4_ik, _g5_ik, _g6_ik, _g7_ik,
                                _geom_codes, _i_nk, _I_Np1k, _J_dk)
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

    def orifice_hydraulic_geometry(self, u=None):
        """
        Compute hydraulic geometry for each link.
        """
        # Import instance variables
        _Ao = self._Ao             # Flow area at link ik
        _g1_o = self._g1_o           # Geometry 1 of link ik (vertical)
        _g2_o = self._g2_o           # Geometry 2 of link ik (horizontal)
        _g3_o = self._g3_o           # Geometry 3 of link ik (other)
        _geom_factory_o = self._geom_factory_o
        _geom_codes_o = self._geom_codes_o
        n_o = self.n_o
        _z_o = self._z_o
        _J_uo = self._J_uo
        _J_do = self._J_do
        H_j = self.H_j
        _z_inv_j = self._z_inv_j
        # Compute effective head
        H_uo = H_j[_J_uo]
        H_do = H_j[_J_do]
        _z_inv_uo = _z_inv_j[_J_uo]
        h_e = np.maximum(H_uo - _z_inv_uo - _z_o, H_do - _z_inv_uo - _z_o)
        if u is None:
            u = np.zeros(n_o, dtype=np.float64)
        # Compute orifice geometries
        numba_orifice_geometry(_Ao, h_e, u, _g1_o, _g2_o, _g3_o, _geom_codes_o, n_o)
        # Export to instance variables
        self._Ao = _Ao

    def compute_storage_areas(self):
        """
        Compute surface area of superjunctions at current time step.
        """
        # Import instance variables
        _functional = self._functional              # Superlinks with functional area curves
        _tabular = self._tabular                    # Superlinks with tabular area curves
        _storage_factory = self._storage_factory    # Dictionary of storage curves
        _storage_indices = self._storage_indices    # Indices of storage curves
        _storage_a = self._storage_a                # Coefficient of functional storage curve
        _storage_b = self._storage_b                # Exponent of functional storage curve
        _storage_c = self._storage_c                # Constant of functional storage curve
        H_j = self.H_j                              # Head at superjunction j
        _z_inv_j = self._z_inv_j                    # Invert elevation at superjunction j
        min_depth = self.min_depth                  # Minimum depth allowed at superjunctions/nodes
        _A_sj = self._A_sj                          # Surface area at superjunction j
        _storage_hs = self._storage_hs
        _storage_As = self._storage_As
        _storage_inds = self._storage_inds
        _storage_lens = self._storage_lens
        _storage_js = self._storage_js
        _storage_codes = self._storage_codes
        # Compute storage areas
        _h_j = np.maximum(H_j - _z_inv_j, min_depth)
        numba_compute_functional_storage_areas(_h_j, _A_sj, _storage_a, _storage_b,
                                                _storage_c, _functional)
        if _tabular.any():
            numba_compute_tabular_storage_areas(_h_j, _A_sj, _storage_hs, _storage_As,
                                                _storage_js, _storage_codes,
                                                _storage_inds, _storage_lens)
        # Export instance variables
        self._A_sj = _A_sj

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

    def link_coeffs(self, _dt=None, first_iter=True):
        """
        Compute link momentum coefficients: a_ik, b_ik, c_ik and P_ik.
        """
        # Import instance variables
        _u_Ik = self._u_Ik         # Flow velocity at junction Ik
        _u_Ip1k = self._u_Ip1k     # Flow velocity at junction I + 1k
        _dx_ik = self._dx_ik       # Length of link ik
        _Sf_method_ik = self._Sf_method_ik
        _n_ik = self._n_ik         # Manning's roughness of link ik
        _Q_ik_prev = np.copy(self.states['Q_ik'])
        _Q_ik_next = self._Q_ik         # Flow rate at link ik
        _A_ik = self._A_ik         # Flow area at link ik
        _R_ik = self._R_ik         # Hydraulic radius at link ik
        _S_o_ik = self._S_o_ik     # Channel bottom slope at link ik
        _A_c_ik = self._A_c_ik     # Area of control structure at link ik
        _C_ik = self._C_ik         # Discharge coefficient of control structure at link ik
        _ctrl = self._ctrl         # Control structure exists at link ik (y/n)
        inertial_damping = self.inertial_damping    # Use inertial damping (y/n)
        _sigma_ik = self._sigma_ik  # Inertial damping coefficient
        g = 9.81
        # If time step not specified, use instance time
        if _dt is None:
            _dt = self._dt
        # Compute link coefficients
        _a_ik = numba_a_ik(_u_Ik, _sigma_ik)
        _c_ik = numba_c_ik(_u_Ip1k, _sigma_ik)
        _b_ik = numba_b_ik(_dx_ik, _dt, _n_ik, _Q_ik_next, _A_ik, _R_ik, _A_c_ik,
                           _C_ik, _a_ik, _c_ik, _ctrl, _sigma_ik, _Sf_method_ik, g)
        _P_ik = numba_P_ik(_Q_ik_prev, _dx_ik, _dt, _A_ik, _S_o_ik,
                           _sigma_ik, g)
        # Export to instance variables
        self._a_ik = _a_ik
        self._b_ik = _b_ik
        self._c_ik = _c_ik
        self._P_ik = _P_ik

    def node_coeffs(self, _Q_0Ik=None, _dt=None, first_iter=True):
        """
        Compute nodal continuity coefficients: D_Ik and E_Ik.
        """
        # Import instance variables
        forward_I_i = self.forward_I_i       # Index of link after junction Ik
        backward_I_i = self.backward_I_i     # Index of link before junction Ik
        _is_start = self._is_start
        _is_end = self._is_end
        _B_ik = self._B_ik                   # Top width of link ik
        _dx_ik = self._dx_ik                 # Length of link ik
        _A_SIk = self._A_SIk                 # Surface area of junction Ik
        _h_Ik_prev = np.copy(self.states['h_Ik'])         # Depth at junction Ik
        _E_Ik = self._E_Ik                   # Continuity coefficient E_Ik
        _D_Ik = self._D_Ik                   # Continuity coefficient D_Ik
        _B_uk = self._B_uk
        _B_dk = self._B_dk
        _dx_uk = self._dx_uk
        _dx_dk = self._dx_dk
        _kI = self._kI
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no nodal input specified, use zero input
        if _Q_0Ik is None:
            _Q_0Ik = np.zeros(_h_Ik_prev.size)
        # Compute E_Ik and D_Ik
        numba_node_coeffs(_D_Ik, _E_Ik, _Q_0Ik, _B_ik, _h_Ik_prev, _dx_ik, _A_SIk,
                          _B_uk, _B_dk, _dx_uk, _dx_dk, _kI,
                          _dt, forward_I_i, backward_I_i, _is_start, _is_end)
        # Export instance variables
        self._E_Ik = _E_Ik
        self._D_Ik = _D_Ik

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

    def superlink_upstream_head_coefficients(self, _dt=None):
        """
        Compute upstream head coefficients for superlinks: kappa_uk, lambda_uk, and mu_uk.
        """
        # Import instance variables
        _I_1k = self._I_1k             # Index of first junction in superlink k
        _i_1k = self._i_1k             # Index of first link in superlink k
        _h_Ik = self._h_Ik             # Depth at junction Ik
        _J_uk = self._J_uk             # Superjunction upstream of superlink k
        _z_inv_uk = self._z_inv_uk     # Invert offset of upstream end of superlink k
        _A_ik = self._A_ik             # Flow area of link ik
        _B_ik = self._B_ik             # Top width of link ik
        _Q_ik = self._Q_ik             # Flow rate of link ik
        _bc_method = self._bc_method   # Method for computing superlink boundary condition (j/z)
        H_j = self.H_j                 # Head at superjunction j
        _A_uk = self._A_uk             # Flow area at upstream end of superlink k
        _B_uk = self._B_uk             # Top width at upstream end of superlink k
        _R_uk = self._R_uk
        _dx_uk = self._dx_uk
        _S_o_uk = self._S_o_uk
        _theta_uk = self._theta_uk
        # Placeholder discharge coefficient
        _C_uk = self._C_uk
        # Current upstream flows
        _Q_uk_next = self._Q_uk
        _Q_uk_prev = np.copy(self.states['Q_uk'])
        # Friction parameters
        _n_uk = self._n_uk
        _Sf_method_uk = self._Sf_method_uk
        g = 9.81
        # If time step not specified, use instance time
        if _dt is None:
            _dt = self._dt
        # Compute theta indicator variables
        _H_juk = H_j[_J_uk]
        upstream_depth_above_invert = _H_juk >= _z_inv_uk
        _theta_uk.fill(0.)
        _theta_uk[upstream_depth_above_invert] = 1.
        if _bc_method == 'z':
            # Compute superlink upstream coefficients (Zahner)
            _gamma_uk = gamma_uk(_Q_uk_next, _C_uk, _A_uk, g)
            self._kappa_uk = _gamma_uk
            self._lambda_uk = _theta_uk
            self._mu_uk = - _theta_uk * _z_inv_uk
        elif _bc_method == 'b':
            # Compute superlink upstream coefficients (momentum)
            self._kappa_uk = kappa_uk(_Q_uk_next, _dx_uk, _A_uk, _C_uk,
                                      _R_uk, _n_uk, _Sf_method_uk, _dt, g)
            self._lambda_uk = _theta_uk
            self._mu_uk = mu_uk(_Q_uk_prev, _dx_uk, _A_uk, _theta_uk, _z_inv_uk,
                                _S_o_uk, _dt, g)
        else:
            raise ValueError('Invalid BC method {}.'.format(_bc_method))
        self._theta_uk = _theta_uk

    def superlink_downstream_head_coefficients(self, _dt=None):
        """
        Compute downstream head coefficients for superlinks: kappa_dk, lambda_dk, and mu_dk.
        """
        # Import instance variables
        _I_Np1k = self._I_Np1k         # Index of last junction in superlink k
        _i_nk = self._i_nk             # Index of last link in superlink k
        _h_Ik = self._h_Ik             # Depth at junction Ik
        _J_dk = self._J_dk             # Superjunction downstream of superlink k
        _z_inv_dk = self._z_inv_dk     # Invert offset of downstream end of superlink k
        _A_ik = self._A_ik             # Flow area of link ik
        _B_ik = self._B_ik             # Top width of link ik
        _Q_ik = self._Q_ik             # Flow rate of link ik
        _bc_method = self._bc_method   # Method for computing superlink boundary condition (j/z)
        H_j = self.H_j                 # Head at superjunction j
        _A_dk = self._A_dk             # Flow area at downstream end of superlink k
        _B_dk = self._B_dk             # Top width at downstream end of superlink k
        _R_dk = self._R_dk
        _dx_dk = self._dx_dk
        _S_o_dk = self._S_o_dk
        _theta_dk = self._theta_dk
        # Placeholder discharge coefficient
        _C_dk = self._C_dk
        # Current downstream flows
        _Q_dk_next = self._Q_dk
        _Q_dk_prev = np.copy(self.states['Q_dk'])
        # Friction parameters
        _n_dk = self._n_dk
        _Sf_method_dk = self._Sf_method_dk
        g = 9.81
        if _dt is None:
            _dt = self._dt
        # Compute theta indicator variables
        _H_jdk = H_j[_J_dk]
        downstream_depth_above_invert = _H_jdk >= _z_inv_dk
        _theta_dk.fill(0.)
        _theta_dk[downstream_depth_above_invert] = 1.
        if _bc_method == 'z':
            # Compute superlink downstream coefficients (Zahner)
            _gamma_dk = gamma_dk(_Q_dk_next, _C_dk, _A_dk, g)
            self._kappa_dk = _gamma_dk
            self._lambda_dk = _theta_dk
            self._mu_dk = - _theta_dk * _z_inv_dk
        elif _bc_method == 'b':
            # Compute superlink upstream coefficients (momentum)
            self._kappa_dk = kappa_dk(_Q_dk_next, _dx_dk, _A_dk, _C_dk,
                                      _R_dk, _n_dk, _Sf_method_dk, _dt, g)
            self._lambda_dk = _theta_dk
            self._mu_dk = mu_dk(_Q_dk_prev, _dx_dk, _A_dk, _theta_dk, _z_inv_dk, _S_o_dk, _dt, g)
        else:
            raise ValueError('Invalid BC method {}.'.format(_bc_method))
        self._theta_dk = _theta_dk

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
        # Compute D_k_star
        _D_k_star = numba_D_k_star(_X_1k, _kappa_uk, _U_Nk,
                                   _kappa_dk, _Z_1k, _W_Nk)
        # Compute upstream superlink flow coefficients
        _alpha_uk = numba_alpha_uk(_U_Nk, _kappa_dk, _X_1k,
                                   _Z_1k, _W_Nk, _D_k_star,
                                   _lambda_uk)
        _beta_uk = numba_beta_uk(_U_Nk, _kappa_dk, _Z_1k,
                                 _W_Nk, _D_k_star, _lambda_dk)
        _chi_uk = numba_chi_uk(_U_Nk, _kappa_dk, _Y_1k,
                               _X_1k, _mu_uk, _Z_1k,
                               _mu_dk, _V_Nk, _W_Nk,
                               _D_k_star)
        # Compute downstream superlink flow coefficients
        _alpha_dk = numba_alpha_dk(_X_1k, _kappa_uk, _W_Nk,
                                   _D_k_star, _lambda_uk)
        _beta_dk = numba_beta_dk(_X_1k, _kappa_uk, _U_Nk,
                                 _W_Nk, _Z_1k, _D_k_star,
                                 _lambda_dk)
        _chi_dk = numba_chi_dk(_X_1k, _kappa_uk, _V_Nk,
                               _W_Nk, _mu_uk, _U_Nk,
                               _mu_dk, _Y_1k, _Z_1k,
                               _D_k_star)
        # Export instance variables
        self._D_k_star = _D_k_star
        self._alpha_uk = _alpha_uk
        self._beta_uk = _beta_uk
        self._chi_uk = _chi_uk
        self._alpha_dk = _alpha_dk
        self._beta_dk = _beta_dk
        self._chi_dk = _chi_dk

    def orifice_flow_coefficients(self, u=None):
        """
        Compute orifice flow coefficients: alpha_uo, beta_uo, chi_uo,
        alpha_do, beta_do, chi_do.
        """
        # Import instance variables
        H_j = self.H_j               # Head at superjunction j
        _z_inv_j = self._z_inv_j     # Invert elevation at superjunction j
        _J_uo = self._J_uo           # Index of superjunction upstream of orifice o
        _J_do = self._J_do           # Index of superjunction downstream of orifice o
        _z_o = self._z_o             # Elevation offset of bottom of orifice o
        _tau_o = self._tau_o         # Orientation of orifice o (side/bottom)
        _y_max_o = self._y_max_o     # Maximum height of orifice o
        _Qo = self._Qo               # Current flow rate of orifice o
        _Co = self._Co               # Discharge coefficient of orifice o
        _Ao = self._Ao               # Maximum flow area of orifice o
        _alpha_o = self._alpha_o     # Orifice flow coefficient alpha_o
        _beta_o = self._beta_o       # Orifice flow coefficient beta_o
        _chi_o = self._chi_o         # Orifice flow coefficient chi_o
        _unidir_o = self._unidir_o
        # If no input signal, assume orifice is closed
        if u is None:
            u = np.zeros(self.n_o, dtype=np.float64)
        # Specify orifice heads at previous timestep
        numba_orifice_flow_coefficients(_alpha_o, _beta_o, _chi_o, H_j, _Qo, u, _z_inv_j,
                                        _z_o, _tau_o, _Co, _Ao, _y_max_o, _unidir_o, _J_uo, _J_do)
        # Export instance variables
        self._alpha_o = _alpha_o
        self._beta_o = _beta_o
        self._chi_o = _chi_o

    def weir_flow_coefficients(self, u=None):
        """
        Compute weir flow coefficients: alpha_uw, beta_uw, chi_uw,
        alpha_dw, beta_dw, chi_dw.
        """
        # Import instance variables
        H_j = self.H_j             # Head at superjunction j
        _z_inv_j = self._z_inv_j   # Invert elevation of superjunction j
        _J_uw = self._J_uw         # Index of superjunction upstream of weir w
        _J_dw = self._J_dw         # Index of superjunction downstream of weir w
        _z_w = self._z_w           # Elevation offset of bottom of weir w
        _y_max_w = self._y_max_w   # Maximum height of weir w
        _Qw = self._Qw             # Current flow rate through weir w
        _Cwr = self._Cwr           # Discharge coefficient for rectangular section
        _Cwt = self._Cwt           # Discharge coefficient for triangular section
        _s_w = self._s_w           # Side slope of triangular section
        _L_w = self._L_w           # Transverse length of rectangular section
        _alpha_w = self._alpha_w   # Weir flow coefficient alpha_w
        _beta_w = self._beta_w     # Weir flow coefficient beta_w
        _chi_w = self._chi_w       # Weir flow coefficient chi_w
        _Hw = self._Hw             # Current effective head above weir w
        # If no input signal, assume weir is closed
        if u is None:
            u = np.zeros(self.n_w, dtype=np.float64)
        # Compute weir flow coefficients
        numba_weir_flow_coefficients(_Hw, _Qw, _alpha_w, _beta_w, _chi_w, H_j, _z_inv_j, _z_w,
                                     _y_max_w, u, _L_w, _s_w, _Cwr, _Cwt, _J_uw, _J_dw)
        # Export instance variables
        self._alpha_w = _alpha_w
        self._beta_w = _beta_w
        self._chi_w = _chi_w

    def pump_flow_coefficients(self, u=None):
        """
        Compute pump flow coefficients: alpha_up, beta_up, chi_up,
        alpha_dp, beta_dp, chi_dp.
        """
        # Import instance variables
        H_j = self.H_j              # Head at superjunction j
        _z_inv_j = self._z_inv_j    # Invert elevation at superjunction j
        _J_up = self._J_up          # Index of superjunction upstream of pump p
        _J_dp = self._J_dp          # Index of superjunction downstream of pump p
        _z_p = self._z_p            # Offset of pump inlet above upstream invert elevation
        _dHp_max = self._dHp_max    # Maximum pump head difference
        _dHp_min = self._dHp_min    # Minimum pump head difference
        _a_p = self._a_p            # Pump curve parameter `a`
        _b_p = self._b_p            # Pump curve parameter `b`
        _c_p = self._c_p            # Pump curve parameter `c`
        _Qp = self._Qp              # Current flow rate through pump p
        _alpha_p = self._alpha_p    # Pump flow coefficient alpha_p
        _beta_p = self._beta_p      # Pump flow coefficient beta_p
        _chi_p = self._chi_p        # Pump flow coefficient chi_p
        # If no input signal, assume pump is closed
        if u is None:
            u = np.zeros(self.n_p, dtype=np.float64)
        # Check max/min head differences
        assert (_dHp_min <= _dHp_max).all()
        # Compute pump flow coefficients
        numba_pump_flow_coefficients(_alpha_p, _beta_p, _chi_p, H_j, _z_inv_j, _Qp, u,
                                     _z_p, _dHp_max, _dHp_min, _a_p, _b_p, _c_p, _J_up, _J_dp)
        # Export instance variables
        self._alpha_p = _alpha_p
        self._beta_p = _beta_p
        self._chi_p = _chi_p

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
        _dx_uk = self._dx_uk
        _dx_dk = self._dx_dk
        _B_uk = self._B_uk
        _B_dk = self._B_dk
        _theta_uk = self._theta_uk
        _theta_dk = self._theta_dk
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
        H_j_next = self.H_j                   # Head at superjunction j
        H_j_prev = self.states['H_j']
        bc = self.bc                     # Superjunction j has a fixed boundary condition (y/n)
        D = self.D                       # Vector for storing chi coefficients
        b = self.b                       # Right-hand side vector
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no boundary head specified, use current superjunction head
        if H_bc is None:
            H_bc = H_j_next
        # If no flow input specified, assume zero external inflow
        if _Q_0j is None:
            _Q_0j = 0
        # If no control input signal specified assume zero input
        if u is None:
            u = 0
        # Compute upstream/downstream link volume parameters
        _xi_uk = xi_uk(_dx_uk, _B_uk, _theta_uk, _dt)
        _xi_dk = xi_dk(_dx_dk, _B_dk, _theta_dk, _dt)
        # Clear old data
        _F_jj.fill(0)
        D.fill(0)
        numba_clear_off_diagonals(A, bc, _J_uk, _J_dk, NK)
        # Create A matrix
        numba_create_A_matrix(A, _F_jj, bc, _J_uk, _J_dk, _alpha_uk,
                              _alpha_dk, _beta_uk, _beta_dk, _xi_uk, _xi_dk,
                              _A_sj, _dt, M, NK)
        # Create D vector
        numba_add_at(D, _J_uk, -_chi_uk)
        numba_add_at(D, _J_dk, _chi_dk)
        numba_add_at(D, _J_uk, _xi_uk * H_j_prev[_J_uk])
        numba_add_at(D, _J_dk, _xi_dk * H_j_prev[_J_dk])
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
            numba_add_at(D, _J_uo, -_chi_uo)
            numba_add_at(D, _J_do, _chi_do)
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
        # TODO: Which A_sj? Might need to apply product rule here.
        b = (_A_sj * H_j_prev / _dt) + _Q_0j + D
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

    def solve_sparse_matrix(self, u=None, implicit=True):
        """
        Solve sparse system Ax = b for superjunction heads at time t + dt.
        """
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
            H_j_next = scipy.sparse.linalg.spsolve(l, r)
        else:
            H_j_next = scipy.linalg.solve(l, r)
        assert np.isfinite(H_j_next).all()
        # Constrain heads based on allowed maximum/minimum depths
        # TODO: Not sure what's happening here
        # H_j_next = np.maximum(H_j_next, _z_inv_j + min_depth)
        H_j_next = np.maximum(H_j_next, _z_inv_j)
        H_j_next = np.minimum(H_j_next, _z_inv_j + max_depth)
        # Export instance variables
        self.H_j = H_j_next

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
        assert np.isfinite(H_j_next).all()
        # Constrain heads based on allowed maximum/minimum depths
        # TODO: Not sure what's happening here
        # H_j_next = np.maximum(H_j_next, _z_inv_j + min_depth)
        # H_j_next = np.minimum(H_j_next, _z_inv_j + max_depth)
        H_j_next = np.maximum(H_j_next, _z_inv_j)
        H_j_next = np.minimum(H_j_next, _z_inv_j + max_depth)
        # Export instance variables
        self.H_j = H_j_next

    def solve_internals_backwards(self, subcritical_only=False):
        """
        Solve for internal states of each superlink in the backward direction.
        """
        # Import instance variables
        _I_1k = self._I_1k                  # Index of first junction in superlink k
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
        numba_solve_internals(_h_Ik, _Q_ik, _h_uk, _h_dk, _U_Ik, _V_Ik, _W_Ik,
                              _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, nk, NK,
                              min_depth, max_depth_k, first_link_backwards=True)
        # TODO: Temporary
        assert np.isfinite(_h_Ik).all()
        # Ensure non-negative depths?
        _h_Ik[_h_Ik < min_depth] = min_depth
        # _h_Ik[_h_Ik > junction_max_depth] = junction_max_depth
        # _h_Ik[_h_Ik > max_depth] = max_depth
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
        numba_solve_internals(_h_Ik, _Q_ik, _h_uk, _h_dk, _U_Ik, _V_Ik, _W_Ik,
                              _X_Ik, _Y_Ik, _Z_Ik, _i_1k, _I_1k, nk, NK,
                              min_depth, max_depth_k, first_link_backwards=False)
        # Ensure non-negative depths?
        _h_Ik[_h_Ik < min_depth] = min_depth
        # _h_Ik[_h_Ik > max_depth] = max_depth
        # Export instance variables
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    def solve_internals_lsq(self):
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
        Q_ik_b, Q_ik_f = self.superlink_flow_from_recurrence()
        _Q_ik = (Q_ik_b + Q_ik_f) / 2
        # Export instance variables
        self._Q_ik = _Q_ik
        self._h_Ik = _h_Ik

    def superlink_flow_from_recurrence(self):
        # Import instance variables
        _h_Ik = self._h_Ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        _Ik = self._Ik
        _ki = self._ki
        # TODO: Need to store nlinks instead of this
        _ik = self._ik
        n = _ik.size
        # Compute internal flow estimates in both directions
        Q_ik_b = numba_Q_i_next_b(_X_Ik, _h_Ik, _Y_Ik, _Z_Ik,
                                  _h_dk, _Ik, _ki, n)
        Q_ik_f = numba_Q_im1k_next_f(_U_Ik, _h_Ik, _V_Ik, _W_Ik,
                                     _h_uk, _Ik, _ki, n)
        return Q_ik_b, Q_ik_f

    def solve_orifice_flows(self, dt, u=None):
        """
        Solve for orifice discharges given superjunction heads at time t + dt.
        """
        # Import instance variables
        H_j = self.H_j                # Head at superjunction j
        _z_inv_j = self._z_inv_j      # Invert elevation at superjunction j
        _J_uo = self._J_uo            # Index of superjunction upstream of orifice o
        _J_do = self._J_do            # Index of superjunction downstream of orifice o
        _z_o = self._z_o              # Offset of orifice above upstream invert elevation
        _tau_o = self._tau_o          # Orientation of orifice o (bottom/side)
        _y_max_o = self._y_max_o      # Maximum height of orifice o
        _Co = self._Co                # Discharge coefficient of orifice o
        _Ao = self._Ao                # Maximum flow area of orifice o
        _V_sj = self._V_sj
        _unidir_o = self._unidir_o
        g = 9.81
        # If no input signal, assume orifice is closed
        if u is None:
            u = np.zeros(self.n_o, dtype=np.float64)
        # Compute orifice flows
        _Qo_next = numba_solve_orifice_flows(H_j, u, _z_inv_j, _z_o, _tau_o, _y_max_o, _Co, _Ao,
                                             _unidir_o, _J_uo, _J_do, g)
        # TODO: Move this inside numba function
        upstream_ctrl = (H_j[_J_uo] > H_j[_J_do])
        _Qo_max = np.where(upstream_ctrl, _V_sj[_J_uo], _V_sj[_J_do]) / dt
        _Qo_next = np.sign(_Qo_next) * np.minimum(np.abs(_Qo_next), _Qo_max)
        # Export instance variables
        self._Qo = _Qo_next

    def solve_weir_flows(self, u=None):
        """
        Solve for weir discharges given superjunction heads at time t + dt.
        """
        # Import instance variables
        H_j = self.H_j              # Head at superjunction j
        _z_inv_j = self._z_inv_j    # Invert elevation of superjunction j
        _J_uw = self._J_uw          # Index of superjunction upstream of weir w
        _J_dw = self._J_dw          # Index of superjunction downstream of weir w
        _z_w = self._z_w            # Offset of weir w above invert of upstream superjunction
        _y_max_w = self._y_max_w    # Maximum height of weir w
        _Qw = self._Qw              # Current flow rate through weir w
        _Cwr = self._Cwr            # Discharge coefficient of rectangular portion of weir w
        _Cwt = self._Cwt            # Discharge coefficient of triangular portion of weir w
        _s_w = self._s_w            # Inverse side slope of triangular portion of weir w
        _L_w = self._L_w            # Transverse length of rectangular portion of weir w
        _Hw = self._Hw              # Current effective head on weir w
        # If no input signal, assume weir is closed
        if u is None:
            u = np.zeros(self.n_w, dtype=np.float64)
        # Solve for weir flows
        _Qw_next = numba_solve_weir_flows(_Hw, _Qw, H_j, _z_inv_j, _z_w,
                                          _y_max_w, u, _L_w, _s_w, _Cwr,
                                          _Cwt, _J_uw, _J_dw)
        # Export instance variables
        self._Qw = _Qw_next

    def solve_pump_flows(self, u=None):
        """
        Solve for pump discharges given superjunction heads at time t + dt.
        """
        # Import instance variables
        H_j = self.H_j              # Head at superjunction j
        _z_inv_j = self._z_inv_j    # Invert elevation of superjunction j
        _J_up = self._J_up          # Index of superjunction upstream of pump p
        _J_dp = self._J_dp          # Index of superjunction downstream of pump p
        _z_p = self._z_p            # Offset of pump inlet above upstream invert
        _dHp_max = self._dHp_max    # Maximum pump head difference
        _dHp_min = self._dHp_min    # Minimum pump head difference
        _a_p = self._a_p            # Pump curve parameter `a`
        _b_p = self._b_p            # Pump curve parameter `b`
        _c_p = self._c_p            # Pump curve parameter `c`
        _Qp = self._Qp              # Current flow rate through pump p
        # If no input signal, assume pump is closed
        if u is None:
            u = np.zeros(self.n_p, dtype=np.float64)
        # Compute pump flows
        _Qp_next = numba_solve_pump_flows(H_j, u, _z_inv_j, _z_p, _dHp_max,
                                          _dHp_min, _a_p, _b_p, _c_p, _J_up, _J_dp)
        self._Qp = _Qp_next

    def compute_storage_volumes(self):
        """
        Compute storage volume of superjunctions at current time step.
        """
        # Import instance variables
        _functional = self._functional              # Superlinks with functional area curves
        _tabular = self._tabular                    # Superlinks with tabular area curves
        _storage_factory = self._storage_factory    # Dictionary of storage curves
        _storage_indices = self._storage_indices    # Indices of storage curves
        _storage_a = self._storage_a                # Coefficient of functional storage curve
        _storage_b = self._storage_b                # Exponent of functional storage curve
        _storage_c = self._storage_c                # Constant of functional storage curve
        H_j = self.H_j                              # Head at superjunction j
        _z_inv_j = self._z_inv_j                    # Invert elevation at superjunction j
        min_depth = self.min_depth                  # Minimum depth allowed at superjunctions/nodes
        _V_sj = self._V_sj                          # Surface area at superjunction j
        _storage_hs = self._storage_hs
        _storage_As = self._storage_As
        _storage_Vs = self._storage_Vs
        _storage_inds = self._storage_inds
        _storage_lens = self._storage_lens
        _storage_js = self._storage_js
        _storage_codes = self._storage_codes
        # Compute storage areas
        _h_j = np.maximum(H_j - _z_inv_j, min_depth)
        numba_compute_functional_storage_volumes(_h_j, _V_sj, _storage_a, _storage_b,
                                                 _storage_c, _functional)
        if _tabular.any():
            numba_compute_tabular_storage_volumes(_h_j, _V_sj, _storage_hs, _storage_As,
                                                  _storage_Vs, _storage_js, _storage_codes,
                                                  _storage_inds, _storage_lens)
        # Export instance variables
        self._V_sj = _V_sj
        # TODO: Temporary to maintain compatibility
        return _V_sj

    def reposition_junctions(self, reposition=None):
        """
        Reposition junctions inside superlinks to enable capture of backwater effects.
        """
        # Import instance variables
        _b0 = self._b0                # Vertical coordinate of upstream end of superlink k
        _b1 = self._b1                # Vertical coordinate of downstream end of superlink k
        _m = self._m                  # Slope of superlink k
        _xc = self._xc                # Horizontal coordinate of center of superlink k
        _zc = self._zc                # Invert elevation of center of superlink k
        _h_Ik = self._h_Ik            # Depth at junction Ik
        _Q_ik = self._Q_ik            # Flow rate at link ik
        _J_dk = self._J_dk            # Index of superjunction downstream of superlink k
        _x_Ik = self._x_Ik            # Horizontal coordinate of junction Ik
        _dx_ik = self._dx_ik          # Length of link ik
        _z_inv_Ik = self._z_inv_Ik    # Invert elevation of junction Ik
        _S_o_ik = self._S_o_ik        # Channel bottom slope of link ik
        _I_1k = self._I_1k            # Index of first junction in superlink k
        _I_Np1k = self._I_Np1k        # Index of last junction in superlink k
        _i_1k = self._i_1k            # Index of first link in superlink k
        H_j = self.H_j                # Head at superjunction j
        _elem_pos = self._elem_pos
        nk = self.nk
        NK = self.NK
        # Check if possible to move elements
        if not self.mobile_elements:
            raise ValueError('Model must be instantiated with `mobile_elements=True` to reposition junctions.')
        # Get downstream head
        _H_dk = H_j[_J_dk]
        # Handle which superlinks to reposition
        if reposition is None:
            reposition = np.ones(NK, dtype=np.bool_)
        # Reposition junctions
        numba_reposition_junctions(_x_Ik, _z_inv_Ik, _h_Ik, _dx_ik, _Q_ik, _H_dk,
                                    _b0, _zc, _xc, _m, _elem_pos, _i_1k, _I_1k,
                                    _I_Np1k, nk, NK, reposition)

def handle_elliptical_perimeter(_Pe_ik, _ellipse_ix, _Ik, _Ip1k, _h_Ik, _g1_ik, _g2_ik):
    if (_ellipse_ix.size > 0):
        _ik_g = _ellipse_ix
        _Ik_g = _Ik[_ik_g]
        _Ip1k_g = _Ip1k[_ik_g]
        _Pe_ik[_ik_g] = pipedream_solver.geometry.Elliptical.Pe_ik(_h_Ik[_Ik_g],
                                                            _h_Ik[_Ip1k_g],
                                                            _g1_ik[_ik_g],
                                                            _g2_ik[_ik_g])

