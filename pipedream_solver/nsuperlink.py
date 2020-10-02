import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg
from numba import njit, prange
import pipedream_solver.geometry
import pipedream_solver.ngeometry
import pipedream_solver.storage
from pipedream_solver.superlink import SuperLink

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
                 end_length=None, end_method='b', internal_links=4):
        super().__init__(superlinks, superjunctions,
                         links, junctions, transects, storages,
                         orifices, weirs, pumps, dt, sparse,
                         min_depth, method, inertial_damping,
                         bc_method, exit_hydraulics, auto_permute,
                         end_length, end_method, internal_links)

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
            u = np.zeros(n_o, dtype=float)
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
        _n_ik = self._n_ik         # Manning's roughness of link ik
        _Q_ik = self._Q_ik         # Flow rate at link ik
        _A_ik = self._A_ik         # Flow area at link ik
        _R_ik = self._R_ik         # Hydraulic radius at link ik
        _S_o_ik = self._S_o_ik     # Channel bottom slope at link ik
        _A_c_ik = self._A_c_ik     # Area of control structure at link ik
        _C_ik = self._C_ik         # Discharge coefficient of control structure at link ik
        _ctrl = self._ctrl         # Control structure exists at link ik (y/n)
        inertial_damping = self.inertial_damping    # Use inertial damping (y/n)
        g = 9.81
        # If using inertial damping, import coefficient
        if inertial_damping:
            _sigma_ik = self._sigma_ik
        # Otherwise, set coefficient to unity
        else:
            _sigma_ik = 1
        # If time step not specified, use instance time
        if _dt is None:
            _dt = self._dt
        # Compute link coefficients
        _a_ik = numba_a_ik(_u_Ik, _sigma_ik)
        _c_ik = numba_c_ik(_u_Ip1k, _sigma_ik)
        _b_ik = numba_b_ik(_dx_ik, _dt, _n_ik, _Q_ik, _A_ik, _R_ik,
                           _A_c_ik, _C_ik, _a_ik, _c_ik, _ctrl, _sigma_ik)
        if first_iter:
            _P_ik = numba_P_ik(_Q_ik, _dx_ik, _dt, _A_ik, _S_o_ik,
                            _sigma_ik)
        # Export to instance variables
        self._a_ik = _a_ik
        self._b_ik = _b_ik
        self._c_ik = _c_ik
        if first_iter:
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
        _h_Ik = self._h_Ik                   # Depth at junction Ik
        _E_Ik = self._E_Ik                   # Continuity coefficient E_Ik
        _D_Ik = self._D_Ik                   # Continuity coefficient D_Ik
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no nodal input specified, use zero input
        if _Q_0Ik is None:
            _Q_0Ik = np.zeros(_h_Ik.size)
        # Compute E_Ik and D_Ik
        numba_node_coeffs(_D_Ik, _E_Ik, _Q_0Ik, _B_ik, _h_Ik, _dx_ik, _A_SIk,
                          _dt, forward_I_i, backward_I_i, _is_start, _is_end,
                          first_iter)
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

    def superlink_upstream_head_coefficients(self):
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
        # Placeholder discharge coefficient
        _C_uk = self._C_uk
        # Current upstream flows
        _Q_uk_t = self._Q_uk
        if _bc_method == 'z':
            # Compute superlink upstream coefficients (Zahner)
            _gamma_uk = gamma_uk(_Q_uk_t, _C_uk, _A_uk)
            self._kappa_uk = _gamma_uk
            self._lambda_uk = 1
            self._mu_uk = - _z_inv_uk
        elif _bc_method == 'j':
            # Current upstream depth
            _h_uk = _h_Ik[_I_1k]
            # Junction head
            _H_juk = H_j[_J_uk]
            # Head difference
            _dH_uk = _H_juk - _h_uk - _z_inv_uk
            # Compute superlink upstream coefficients (Ji)
            _kappa_uk = self.kappa_uk(_A_uk, _dH_uk, _Q_uk_t, _B_uk)
            _lambda_uk = self.lambda_uk(_A_uk, _dH_uk, _B_uk)
            _mu_uk = self.mu_uk(_A_uk, _dH_uk, _B_uk, _z_inv_uk)
            self._kappa_uk = _kappa_uk
            self._lambda_uk = _lambda_uk
            self._mu_uk = _mu_uk
        else:
            raise ValueError('Invalid BC method {}.'.format(_bc_method))

    def superlink_downstream_head_coefficients(self):
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
        # Placeholder discharge coefficient
        _C_dk = self._C_dk
        # Current downstream flows
        _Q_dk_t = self._Q_dk
        if _bc_method == 'z':
            # Compute superlink downstream coefficients (Zahner)
            _gamma_dk = gamma_dk(_Q_dk_t, _C_dk, _A_dk)
            self._kappa_dk = _gamma_dk
            self._lambda_dk = 1
            self._mu_dk = - _z_inv_dk
        elif _bc_method == 'j':
            # Downstream top width
            # Current downstream depth
            _h_dk = _h_Ik[_I_Np1k]
            # Junction head
            _H_jdk = H_j[_J_dk]
            # Head difference
            _dH_dk = _h_dk + _z_inv_dk - _H_jdk
            # Compute superlink upstream coefficients (Ji)
            _kappa_dk = self.kappa_dk(_A_dk, _dH_dk, _Q_dk_t, _B_dk)
            _lambda_dk = self.lambda_dk(_A_dk, _dH_dk, _B_dk)
            _mu_dk = self.mu_dk(_A_dk, _dH_dk, _B_dk, _z_inv_dk)
            self._kappa_dk = _kappa_dk
            self._lambda_dk = _lambda_dk
            self._mu_dk = _mu_dk
        else:
            raise ValueError('Invalid BC method {}.'.format(_bc_method))

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
        # If no input signal, assume orifice is closed
        if u is None:
            u = np.zeros(self.n_o, dtype=float)
        # Specify orifice heads at previous timestep
        numba_orifice_flow_coefficients(_alpha_o, _beta_o, _chi_o, H_j, _Qo, u, _z_inv_j,
                                        _z_o, _tau_o, _Co, _Ao, _y_max_o, _J_uo, _J_do)
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
            u = np.zeros(self.n_w, dtype=float)
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
        _ap_h = self._ap_h          # Horizontal axis length of elliptical pump curve
        _ap_q = self._ap_q          # Vertical axis length of elliptical pump curve
        _Qp = self._Qp              # Current flow rate through pump p
        _alpha_p = self._alpha_p    # Pump flow coefficient alpha_p
        _beta_p = self._beta_p      # Pump flow coefficient beta_p
        _chi_p = self._chi_p        # Pump flow coefficient chi_p
        # If no input signal, assume pump is closed
        if u is None:
            u = np.zeros(self.n_p, dtype=float)
        # Check max/min head differences
        assert (_dHp_min <= _dHp_max).all()
        # Compute pump flow coefficients
        numba_pump_flow_coefficients(_alpha_p, _beta_p, _chi_p, H_j, _z_inv_j, _Qp, u,
                                     _z_p, _dHp_max, _dHp_min, _ap_q, _ap_h, _J_up, _J_dp)
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
        # If no input signal, assume orifice is closed
        if u is None:
            u = np.zeros(self.n_o, dtype=float)
        # Compute orifice flows
        _Qo_next = numba_solve_orifice_flows(H_j, u, _z_inv_j, _z_o, _tau_o, _y_max_o, _Co, _Ao,
                                             _J_uo, _J_do)
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
            u = np.zeros(self.n_w, dtype=float)
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
        _ap_h = self._ap_h          # Horizontal axis length of elliptical pump curve
        _ap_q = self._ap_q          # Vertical axis length of elliptical pump curve
        _Qp = self._Qp              # Current flow rate through pump p
        # If no input signal, assume pump is closed
        if u is None:
            u = np.zeros(self.n_p, dtype=float)
        # Compute pump flows
        _Qp_next = numba_solve_pump_flows(H_j, u, _z_inv_j, _z_p, _dHp_max,
                                          _dHp_min, _ap_q, _ap_h, _J_up, _J_dp)
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
        _x0 = self._x0                # Horizontal coordinate of center of superlink k
        _z0 = self._z0                # Invert elevation of center of superlink k
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
        # Get downstream head
        _H_dk = H_j[_J_dk]
        # Handle which superlinks to reposition
        if reposition is None:
            reposition = np.ones(NK, dtype=bool)
        # Reposition junctions
        numba_reposition_junctions(_x_Ik, _z_inv_Ik, _h_Ik, _dx_ik, _Q_ik, _H_dk,
                                    _b0, _z0, _x0, _m, _elem_pos, _i_1k, _I_1k,
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
                _A_ik[i] = pipedream_solver.ngeometry.Circular_A_ik(h_I, h_Ip1, g1_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Circular_Pe_ik(h_I, h_Ip1, g1_i)
                _R_ik[i] = pipedream_solver.ngeometry.Circular_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Circular_B_ik(h_I, h_Ip1, g1_i)
            elif geom_code == 2:
                _A_ik[i] = pipedream_solver.ngeometry.Rect_Closed_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Rect_Closed_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Rect_Closed_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Rect_Closed_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 3:
                _A_ik[i] = pipedream_solver.ngeometry.Rect_Open_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Rect_Open_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Rect_Open_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Rect_Open_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 4:
                _A_ik[i] = pipedream_solver.ngeometry.Triangular_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Triangular_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Triangular_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Triangular_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 5:
                _A_ik[i] = pipedream_solver.ngeometry.Trapezoidal_A_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Trapezoidal_Pe_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
                _R_ik[i] = pipedream_solver.ngeometry.Trapezoidal_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Trapezoidal_B_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
            elif geom_code == 6:
                _A_ik[i] = pipedream_solver.ngeometry.Parabolic_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Parabolic_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Parabolic_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Parabolic_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 7:
                # NOTE: Assumes that perimeter has already been calculated
                _A_ik[i] = pipedream_solver.ngeometry.Elliptical_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Elliptical_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Elliptical_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 8:
                _A_ik[i] = pipedream_solver.ngeometry.Wide_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _Pe_ik[i] = pipedream_solver.ngeometry.Wide_Pe_ik(h_I, h_Ip1, g1_i, g2_i)
                _R_ik[i] = pipedream_solver.ngeometry.Wide_R_ik(_A_ik[i], _Pe_ik[i])
                _B_ik[i] = pipedream_solver.ngeometry.Wide_B_ik(h_I, h_Ip1, g1_i, g2_i)
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
                _A_bk[k] = pipedream_solver.ngeometry.Circular_A_ik(h_I, h_Ip1, g1_i)
                _B_bk[k] = pipedream_solver.ngeometry.Circular_B_ik(h_I, h_Ip1, g1_i)
            elif geom_code == 2:
                _A_bk[k] = pipedream_solver.ngeometry.Rect_Closed_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = pipedream_solver.ngeometry.Rect_Closed_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 3:
                _A_bk[k] = pipedream_solver.ngeometry.Rect_Open_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = pipedream_solver.ngeometry.Rect_Open_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 4:
                _A_bk[k] = pipedream_solver.ngeometry.Triangular_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = pipedream_solver.ngeometry.Triangular_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 5:
                _A_bk[k] = pipedream_solver.ngeometry.Trapezoidal_A_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
                _B_bk[k] = pipedream_solver.ngeometry.Trapezoidal_B_ik(h_I, h_Ip1, g1_i, g2_i, g3_i)
            elif geom_code == 6:
                _A_bk[k] = pipedream_solver.ngeometry.Parabolic_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = pipedream_solver.ngeometry.Parabolic_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 7:
                _A_bk[k] = pipedream_solver.ngeometry.Elliptical_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = pipedream_solver.ngeometry.Elliptical_B_ik(h_I, h_Ip1, g1_i, g2_i)
            elif geom_code == 8:
                _A_bk[k] = pipedream_solver.ngeometry.Wide_A_ik(h_I, h_Ip1, g1_i, g2_i)
                _B_bk[k] = pipedream_solver.ngeometry.Wide_B_ik(h_I, h_Ip1, g1_i, g2_i)
    return 1

@njit
def numba_orifice_geometry(_Ao, h_eo, u_o, _g1_o, _g2_o, _g3_o, _geom_codes_o, n_o):
    for i in range(n_o):
        geom_code = _geom_codes_o[i]
        g1 = _g1_o[i]
        g2 = _g2_o[i]
        g3 = _g3_o[i]
        u = u_o[i]
        h_e = h_eo[i]
        if geom_code:
            if geom_code == 1:
                _Ao[i] = pipedream_solver.ngeometry.Circular_A_ik(h_e, h_e, g1 * u)
            elif geom_code == 2:
                _Ao[i] = pipedream_solver.ngeometry.Rect_Closed_A_ik(h_e, h_e, g1 * u, g2)
            elif geom_code == 3:
                _Ao[i] = pipedream_solver.ngeometry.Rect_Open_A_ik(h_e, h_e, g1 * u, g2)
            elif geom_code == 4:
                _Ao[i] = pipedream_solver.ngeometry.Triangular_A_ik(h_e, h_e, g1 * u, g2)
            elif geom_code == 5:
                _Ao[i] = pipedream_solver.ngeometry.Trapezoidal_A_ik(h_e, h_e, g1 * u, g2, g3)
            elif geom_code == 6:
                _Ao[i] = pipedream_solver.ngeometry.Parabolic_A_ik(h_e, h_e, g1 * u, g2)
            elif geom_code == 7:
                _Ao[i] = pipedream_solver.ngeometry.Elliptical_A_ik(h_e, h_e, g1 * u, g2)
            elif geom_code == 8:
                _Ao[i] = pipedream_solver.ngeometry.Wide_A_ik(h_e, h_e, g1 * u, g2)
    return 1

@njit
def numba_compute_functional_storage_areas(h, A, a, b, c, _functional):
    M = h.size
    for j in range(M):
        if _functional[j]:
            if h[j] < 0:
                A[j] = 0
            else:
                A[j] = a[j] * (h[j]**b[j]) + c[j]
    return A

@njit
def numba_compute_functional_storage_volumes(h, V, a, b, c, _functional):
    M = h.size
    for j in range(M):
        if _functional[j]:
            if h[j] < 0:
                V[j] = 0
            else:
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

@njit
def numba_a_ik(u_Ik, sigma_ik):
    """
    Compute link coefficient 'a' for link i, superlink k.
    """
    return -np.maximum(u_Ik, 0) * sigma_ik

@njit
def numba_c_ik(u_Ip1k, sigma_ik):
    """
    Compute link coefficient 'c' for link i, superlink k.
    """
    return -np.maximum(-u_Ip1k, 0) * sigma_ik

@njit
def numba_b_ik(dx_ik, dt, n_ik, Q_ik_t, A_ik, R_ik,
               A_c_ik, C_ik, a_ik, c_ik, ctrl, sigma_ik, g=9.81):
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

@njit
def numba_P_ik(Q_ik_t, dx_ik, dt, A_ik, S_o_ik, sigma_ik, g=9.81):
    """
    Compute link coefficient 'P' for link i, superlink k.
    """
    t_0 = (Q_ik_t * dx_ik / dt) * sigma_ik
    t_1 = g * A_ik * S_o_ik * dx_ik
    return t_0 + t_1

@njit
def E_Ik(B_ik, dx_ik, B_im1k, dx_im1k, A_SIk, dt):
    """
    Compute node coefficient 'E' for node I, superlink k.
    """
    t_0 = B_ik * dx_ik / 2
    t_1 = B_im1k * dx_im1k / 2
    t_2 = A_SIk
    t_3 = dt
    return (t_0 + t_1 + t_2) / t_3

@njit
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

@njit
def numba_node_coeffs(_D_Ik, _E_Ik, _Q_0Ik, _B_ik, _h_Ik, _dx_ik, _A_SIk, _dt,
                      _forward_I_i, _backward_I_i, _is_start, _is_end, first_iter):
    N = _h_Ik.size
    for I in range(N):
        if _is_start[I]:
            i = _forward_I_i[I]
            _E_Ik[I] = E_Ik(_B_ik[i], _dx_ik[i], 0.0, 0.0, _A_SIk[I], _dt)
            if first_iter:
                _D_Ik[I] = D_Ik(_Q_0Ik[I], _B_ik[i], _dx_ik[i], 0.0, 0.0, _A_SIk[I],
                                _h_Ik[I], _dt)
        elif _is_end[I]:
            im1 = _backward_I_i[I]
            _E_Ik[I] = E_Ik(0.0, 0.0, _B_ik[im1], _dx_ik[im1],
                            _A_SIk[I], _dt)
            if first_iter:
                _D_Ik[I] = D_Ik(_Q_0Ik[I], 0.0, 0.0, _B_ik[im1],
                                _dx_ik[im1], _A_SIk[I], _h_Ik[I], _dt)
        else:
            i = _forward_I_i[I]
            im1 = i - 1
            _E_Ik[I] = E_Ik(_B_ik[i], _dx_ik[i], _B_ik[im1], _dx_ik[im1],
                            _A_SIk[I], _dt)
            if first_iter:
                _D_Ik[I] = D_Ik(_Q_0Ik[I], _B_ik[i], _dx_ik[i], _B_ik[im1],
                                _dx_ik[im1], _A_SIk[I], _h_Ik[I], _dt)
    return 1

@njit
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
            if _h_Ik[I] < min_depth:
                _h_Ik[I] = min_depth
            if _h_Ik[I] > max_depth:
                _h_Ik[I] = max_depth
        if first_link_backwards:
            _Q_ik[i_1] = Q_i_b(_h_Ik[I_1], _h_Np1k, _X_Ik[I_1], _Y_Ik[I_1],
                            _Z_Ik[I_1])
        else:
            # Not theoretically correct, but seems to be more stable sometimes
            _Q_ik[i_1] = Q_i_f(_h_Ik[I_1 + 1], _h_1k, _U_Ik[I_1], _V_Ik[I_1],
                            _W_Ik[I_1])
    return 1

@njit
def Q_i_f(h_Ip1k, h_1k, U_Ik, V_Ik, W_Ik):
    t_0 = U_Ik * h_Ip1k
    t_1 = V_Ik
    t_2 = W_Ik * h_1k
    return t_0 + t_1 + t_2

@njit
def Q_i_b(h_Ik, h_Np1k, X_Ik, Y_Ik, Z_Ik):
    t_0 = X_Ik * h_Ik
    t_1 = Y_Ik
    t_2 = Z_Ik * h_Np1k
    return t_0 + t_1 + t_2

@njit
def h_i_b(Q_ik, h_Np1k, X_Ik, Y_Ik, Z_Ik):
    num = Q_ik - Y_Ik - Z_Ik * h_Np1k
    den = X_Ik
    result = safe_divide(num, den)
    return result

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
def gamma_o(Q_o_t, Ao, Co=0.67, g=9.81):
    """
    Compute flow coefficient 'gamma' for orifice o.
    """
    num = 2 * g * Co**2 * Ao**2
    den = np.abs(Q_o_t)
    result = safe_divide_vec(num, den)
    return result

@njit
def gamma_w(Q_w_t, H_w_t, L_w, s_w, Cwr=1.838, Cwt=1.380):
    """
    Compute flow coefficient 'gamma' for weir w.
    """
    num = (Cwr * L_w * H_w_t + Cwt * s_w * H_w_t**2)**2
    den = np.abs(Q_w_t)
    result = safe_divide_vec(num, den)
    return result

@njit
def gamma_p(Q_p_t, dH_p_t, a_q=1.0, a_h=1.0):
    """
    Compute flow coefficient 'gamma' for pump p.
    """
    num = a_q**2 * np.abs(dH_p_t)
    den = a_h**2 * np.abs(Q_p_t)
    result = safe_divide_vec(num, den)
    return result

@njit
def gamma_uk(Q_uk_t, C_uk, A_uk, g=9.81):
    """
    Compute flow coefficient 'gamma' for upstream end of superlink k
    """
    num = -np.abs(Q_uk_t)
    den = 2 * (C_uk**2) * (A_uk**2) * g
    result = safe_divide_vec(num, den)
    return result

@njit
def gamma_dk(Q_dk_t, C_dk, A_dk, g=9.81):
    """
    Compute flow coefficient 'gamma' for downstream end of superlink k
    """
    num = np.abs(Q_dk_t)
    den = 2 * (C_dk**2) * (A_dk**2) * g
    result = safe_divide_vec(num, den)
    return result

@njit
def numba_orifice_flow_coefficients(_alpha_o, _beta_o, _chi_o, H_j, _Qo, u, _z_inv_j,
                                    _z_o, _tau_o, _Co, _Ao, _y_max_o, _J_uo, _J_do):
    _H_uo = H_j[_J_uo]
    _H_do = H_j[_J_do]
    _z_inv_uo = _z_inv_j[_J_uo]
    # Create indicator functions
    _omega_o = np.zeros_like(_H_uo)
    _omega_o[_H_uo >= _H_do] = 1.0
    # Compute universal coefficients
    _gamma_o = gamma_o(_Qo, _Ao, _Co)
    # Create conditionals
    cond_0 = (_omega_o * _H_uo + (1 - _omega_o) * _H_do >
                _z_o + _z_inv_uo + (_tau_o * _y_max_o * u))
    cond_1 = ((1 - _omega_o) * _H_uo + _omega_o * _H_do >
                _z_o + _z_inv_uo + (_tau_o * _y_max_o * u / 2))
    cond_2 = (_omega_o * _H_uo + (1 - _omega_o) * _H_do >
                _z_o + _z_inv_uo)
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
    d = (~cond_0 & ~cond_2)
    _alpha_o[d] = 0.0
    _beta_o[d] = 0.0
    _chi_o[d] = 0.0
    return 1

@njit
def numba_solve_orifice_flows(H_j, u, _z_inv_j, _z_o,
                              _tau_o, _y_max_o, _Co, _Ao, _J_uo, _J_do, g=9.81):
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
    d = (~cond_0 & ~cond_2)
    _alpha_o[d] = 0.0
    _beta_o[d] = 0.0
    _chi_o[d] = 0.0
    # Compute flow
    _Qo_next = (-1)**(1 - _omega_o) * np.sqrt(np.abs(
               _alpha_o * _H_uo + _beta_o * _H_do + _chi_o))
    # Export instance variables
    return _Qo_next

@njit
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

@njit
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

@njit
def numba_pump_flow_coefficients(_alpha_p, _beta_p, _chi_p, H_j, _z_inv_j, _Qp, u,
                                 _z_p, _dHp_max, _dHp_min, _ap_q, _ap_h, _J_up, _J_dp):
    # Get upstream and downstream heads and invert elevation
    _H_up = H_j[_J_up]
    _H_dp = H_j[_J_dp]
    _z_inv_up = _z_inv_j[_J_up]
    # Compute effective head
    _dHp = _H_dp - _H_up
    cond_0 = _H_up > _z_inv_up + _z_p
    cond_1 = (_dHp > _dHp_min) & (_dHp < _dHp_max)
    _dHp[_dHp > _dHp_max] = _dHp_max
    _dHp[_dHp < _dHp_min] = _dHp_min
    # Compute universal coefficients
    _gamma_p = gamma_p(_Qp, _dHp, _ap_q, _ap_h)
    # Fill coefficient arrays
    # Head in pump curve range
    a = (cond_0 & cond_1)
    _alpha_p[a] = _gamma_p[a] * u[a]**2
    _beta_p[a] = -_gamma_p[a] * u[a]**2
    _chi_p[a] = (_gamma_p[a] * _ap_h[a]**2 / np.abs(_dHp[a])) * u[a]**2
    # Head outside of pump curve range
    b = (cond_0 & ~cond_1)
    _alpha_p[b] = 0.0
    _beta_p[b] = 0.0
    _chi_p[b] = np.sqrt(np.maximum(_ap_q[b]**2 * (1 - _dHp[b]**2 / _ap_h[b]**2), 0.0)) * u[b]
    # Depth below inlet
    c = (~cond_0)
    _alpha_p[c] = 0.0
    _beta_p[c] = 0.0
    _chi_p[c] = 0.0
    return 1

@njit
def numba_solve_pump_flows(H_j, u, _z_inv_j, _z_p, _dHp_max, _dHp_min, _ap_q, _ap_h,
                           _J_up, _J_dp):
    _H_up = H_j[_J_up]
    _H_dp = H_j[_J_dp]
    _z_inv_up = _z_inv_j[_J_up]
    # Create conditionals
    _dHp = _H_dp - _H_up
    _dHp[_dHp > _dHp_max] = _dHp_max
    _dHp[_dHp < _dHp_min] = _dHp_min
    cond_0 = _H_up > _z_inv_up + _z_p
    # Compute universal coefficients
    _Qp_next = u * np.sqrt(np.maximum(_ap_q**2 * (1 - (_dHp)**2 / _ap_h**2), 0.0))
    _Qp_next[~cond_0] = 0.0
    return _Qp_next

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

@njit
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

@njit
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

@njit
def numba_reposition_junctions(_x_Ik, _z_inv_Ik, _h_Ik, _dx_ik, _Q_ik, _H_dk,
                               _b0, _z0, _x0, _m, _elem_pos, _i_1k, _I_1k,
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
                x_m = (_H_d - _b0[k]) / _m[k]
            else:
                z_m = _z0[k]
                x_m = _x0[k]
                # NOTE: Changing this to not move instead
                # z_m = _z_inv_I[pos_prev]
                # x_m = _x_I[pos_prev]
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
            # TODO: For testing only, remove this later
            # r = 0.5
            if shifted:
                ix = np.arange(nlinks)
                ix[pos_prev] = pos_next
                ix.sort()
                _Q_i = _Q_ik[_i_1:_i_end]
                _Q_i[pos_prev - 1] = (1 - r) * _Q_i[pos_prev - 1] + r * _Q_i[pos_prev]
                _Q_ik[_i_1:_i_end] = _Q_i[ix]

