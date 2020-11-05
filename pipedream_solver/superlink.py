import copy
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import pipedream_solver.geometry
import pipedream_solver.storage
import pipedream_solver.visualization

class SuperLink():
    """
    SUPERLINK hydraulic solver, as described in:

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
        # Copy input tables to prevent modification
        superjunctions = copy.deepcopy(superjunctions)
        superlinks = copy.deepcopy(superlinks)
        orifices = copy.deepcopy(orifices)
        weirs = copy.deepcopy(weirs)
        pumps = copy.deepcopy(pumps)
        # TODO: This needs to be done for orifices/weirs/pumps as well
        # Ensure nominal direction of superlinks is correct
        if (superlinks is not None) and (superjunctions is not None):
            for k in superlinks.index.values:
                sj_0 = superlinks.loc[k, 'sj_0']
                sj_1 = superlinks.loc[k, 'sj_1']
                z_inv_0 = superjunctions.loc[sj_0, 'z_inv']
                z_inv_1 = superjunctions.loc[sj_1, 'z_inv']
                z_inv_uk = superlinks.loc[k, 'in_offset']
                z_inv_dk = superlinks.loc[k, 'out_offset']
                cond = z_inv_0 + z_inv_uk < z_inv_1 + z_inv_dk
                if cond:
                    superlinks.loc[k, 'sj_0'] = sj_1
                    superlinks.loc[k, 'sj_1'] = sj_0
        # Save copied input tables to class instance
        self.superjunctions = superjunctions
        self.superlinks = superlinks
        self.orifices = orifices
        self.weirs = weirs
        self.pumps = pumps
        # Dimensions of supersystem
        self.M = len(superjunctions)
        if superlinks is not None:
            self.NK = len(superlinks)
        else:
            self.NK = 0
        # Permute superjunctions if specified
        if auto_permute:
            self._initialize_with_permuted_columns()
            self.banded = True
            # TODO: Redundant, but safer
            superjunctions = self.superjunctions
            superlinks = self.superlinks
            orifices = self.orifices
            weirs = self.weirs
            pumps = self.pumps
        else:
            self.permutations = np.arange(len(superjunctions))
            self.banded = False
        # TODO: What about id?
        if not 'map_x' in self.superjunctions.columns:
            self.superjunctions['map_x'] = 0.
        if not 'map_y' in self.superjunctions.columns:
            self.superjunctions['map_y'] = 0.
        self._map_x_j = self.superjunctions['map_x'].values.astype(float)
        self._map_y_j = self.superjunctions['map_y'].values.astype(float)
        if 'name' in self.superjunctions.columns:
            self.superjunction_names = self.superjunctions['name'].values
        else:
            self.superjunction_names = self.superjunctions.index.values
        if 'name' in self.superlinks.columns:
            self.superlink_names = self.superlinks['name'].values
        else:
            self.superlink_names = self.superlinks.index.values
        # If internal links and junctions are not provided, create them
        if (links is None) or (junctions is None):
            generate_elems = True
            self._configure_internals_variable(njunctions_fixed=internal_links)
            links = self.links
            junctions = self.junctions
        else:
            generate_elems = False
            self.links = links
            self.junctions = junctions
        self.NIk = self.junctions.shape[0]
        self.Nik = self.links.shape[0]
        self.transects = transects
        self.storages = storages
        self._dt = dt
        self._sparse = sparse
        self._method = method
        self._bc_method = bc_method
        self._end_method = end_method
        self._exit_hydraulics = exit_hydraulics
        self.inertial_damping = inertial_damping
        self.min_depth = min_depth
        self._I = junctions.index.values
        self._ik = links.index.values
        self._i = self._ik
        self._Ik = links['j_0'].values.astype(int)
        self._Ip1k = links['j_1'].values.astype(int)
        self._kI = junctions['k'].values.astype(int)
        self._ki = links['k'].values.astype(int)
        self.start_nodes = self.superlinks['j_0'].values.astype(int)
        self.end_nodes = self.superlinks['j_1'].values.astype(int)
        self._is_start = np.zeros(self._I.size, dtype=bool)
        self._is_end = np.zeros(self._I.size, dtype=bool)
        self._is_start[self.start_nodes] = True
        self._is_end[self.end_nodes] = True
        self.middle_nodes = self._I[(~self._is_start) & (~self._is_end)]
        self._is_penult = np.roll(self._is_end, -1)
        # Dimensions
        self.nk = np.bincount(self._ki)
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
        # Set maximum superjunction depth
        if 'max_depth' in superjunctions:
            self.max_depth = superjunctions['max_depth'].values.astype(float)
        else:
            self.max_depth = np.full(len(superjunctions), np.inf, dtype=float)
        # Set maximum superlink depth for stability
        if 'max_depth' in superlinks:
            self.max_depth_k = superlinks['max_depth'].values.astype(float)
        else:
            self.max_depth_k = np.full(len(superlinks), np.inf, dtype=float)
        if 'C_uk' in superlinks:
            self._C_uk = superlinks['C_uk'].values.astype(float)
        else:
            self._C_uk = 0.67 * np.ones(self.NK)
        if 'C_dk' in superlinks:
            self._C_dk = superlinks['C_dk'].values.astype(float)
        else:
            self._C_dk = 0.67 * np.ones(self.NK)
        self._h_Ik = junctions.loc[self._I, 'h_0'].values.astype(float)
        self._A_SIk = junctions.loc[self._I, 'A_s'].values.astype(float)
        self._z_inv_Ik = junctions.loc[self._I, 'z_inv'].values.astype(float)
        self._S_o_ik = ((self._z_inv_Ik[self._Ik] - self._z_inv_Ik[self._Ip1k])
                        / self._dx_ik)
        self._x_Ik = np.zeros(self._I.size)
        self._x_Ik[~self._is_start] = links.groupby('k')['dx'].cumsum().values
        # TODO: Allow specifying initial flows
        self._Q_0Ik = np.zeros(self._I.size, dtype=float)
        # Handle orifices
        if orifices is not None:
            self.n_o = self.orifices.shape[0]
            self._J_uo = self.orifices['sj_0'].values.astype(int)
            self._J_do = self.orifices['sj_1'].values.astype(int)
            self._Ao_max = self.orifices['A'].values.astype(float)
            self._Ao = np.copy(self._Ao_max)
            self._Co = self.orifices['C'].values.astype(float)
            self._z_o = self.orifices['z_o'].values.astype(float)
            self._y_max_o = self.orifices['y_max'].values.astype(float)
            self._orient_o = self.orifices['orientation'].values
            self._tau_o = (self._orient_o == 'side').astype(float)
            if 'shape' in self.orifices.columns:
                self._shape_o = self.orifices['shape']
                self._g1_o = self.orifices['g1'].values.astype(float)
                self._g2_o = self.orifices['g2'].values.astype(float)
                self._g3_o = self.orifices['g3'].values.astype(float)
            else:
                self._shape_o = pd.Series(['rect_open'] * self.n_o)
                self._g1_o = np.sqrt(self._Ao_max)
                self._g2_o = np.sqrt(self._Ao_max)
                self._g3_o = np.zeros(self.n_o)
            self._Qo = np.zeros(self.n_o, dtype=float)
            self._alpha_o = np.zeros(self.n_o, dtype=float)
            self._beta_o = np.zeros(self.n_o, dtype=float)
            self._chi_o = np.zeros(self.n_o, dtype=float)
            self._alpha_uom = np.zeros(self.M, dtype=float)
            self._beta_dol = np.zeros(self.M, dtype=float)
            self._chi_uol = np.zeros(self.M, dtype=float)
            self._chi_dom = np.zeros(self.M, dtype=float)
        else:
            self._J_uo = np.array([], dtype=int)
            self._J_do = np.array([], dtype=int)
            self._Ao_max = np.array([], dtype=float)
            self._Ao = np.array([], dtype=float)
            self._Co = np.array([], dtype=float)
            self._z_o = np.array([], dtype=float)
            self._y_max_o = np.array([], dtype=float)
            self._orient_o = np.array([])
            self._shape_o = pd.Series(np.array([]))
            self._g1_o = np.array([], dtype=float)
            self._g2_o = np.array([], dtype=float)
            self._g3_o = np.array([], dtype=float)
            self.n_o = 0
            self._Qo = np.array([], dtype=float)
            self._alpha_o = np.array([], dtype=float)
            self._beta_o = np.array([], dtype=float)
            self._chi_o = np.array([], dtype=float)
            self._alpha_uom = np.array([], dtype=float)
            self._beta_dol = np.array([], dtype=float)
            self._chi_uol = np.array([], dtype=float)
            self._chi_dom = np.array([], dtype=float)
        # Handle weirs
        if weirs is not None:
            self._J_uw = self.weirs['sj_0'].values.astype(int)
            self._J_dw = self.weirs['sj_1'].values.astype(int)
            self._Cwr = self.weirs['Cr'].values.astype(float)
            self._Cwt = self.weirs['Ct'].values.astype(float)
            self._L_w = self.weirs['L'].values.astype(float)
            self._s_w = self.weirs['s'].values.astype(float)
            self._z_w = self.weirs['z_w'].values.astype(float)
            self._y_max_w = self.weirs['y_max'].values.astype(float)
            self.n_w = self.weirs.shape[0]
            self._Hw = np.zeros(self.n_w, dtype=float)
            self._Qw = np.zeros(self.n_w, dtype=float)
            self._alpha_w = np.zeros(self.n_w, dtype=float)
            self._beta_w = np.zeros(self.n_w, dtype=float)
            self._chi_w = np.zeros(self.n_w, dtype=float)
            self._alpha_uwm = np.zeros(self.M, dtype=float)
            self._beta_dwl = np.zeros(self.M, dtype=float)
            self._chi_uwl = np.zeros(self.M, dtype=float)
            self._chi_dwm = np.zeros(self.M, dtype=float)
        else:
            self._J_uw = np.array([], dtype=int)
            self._J_dw = np.array([], dtype=int)
            self._Cwr = np.array([], dtype=float)
            self._Cwt = np.array([], dtype=float)
            self._L_w = np.array([], dtype=float)
            self._s_w = np.array([], dtype=float)
            self._z_w = np.array([], dtype=float)
            self._y_max_w = np.array([], dtype=float)
            self.n_w = 0
            self._Hw = np.array([], dtype=float)
            self._Qw = np.array([], dtype=float)
            self._alpha_w = np.array([], dtype=float)
            self._beta_w = np.array([], dtype=float)
            self._chi_w = np.array([], dtype=float)
            self._alpha_uwm = np.array([], dtype=float)
            self._beta_dwl = np.array([], dtype=float)
            self._chi_uwl = np.array([], dtype=float)
            self._chi_dwm = np.array([], dtype=float)
        # Handle pumps
        if pumps is not None:
            self._J_up = self.pumps['sj_0'].values.astype(int)
            self._J_dp = self.pumps['sj_1'].values.astype(int)
            self._z_p = self.pumps['z_p'].values.astype(float)
            self._ap_h = self.pumps['a_h'].values.astype(float)
            self._ap_q = self.pumps['a_q'].values.astype(float)
            self._dHp_max = self.pumps['dH_max'].values.astype(float)
            self._dHp_min = self.pumps['dH_min'].values.astype(float)
            self.n_p = self.pumps.shape[0]
            self._Qp = np.zeros(self.n_p, dtype=float)
            self._alpha_p = np.zeros(self.n_p, dtype=float)
            self._beta_p = np.zeros(self.n_p, dtype=float)
            self._chi_p = np.zeros(self.n_p, dtype=float)
            self._alpha_upm = np.zeros(self.M, dtype=float)
            self._beta_dpl = np.zeros(self.M, dtype=float)
            self._chi_upl = np.zeros(self.M, dtype=float)
            self._chi_dpm = np.zeros(self.M, dtype=float)
        else:
            self._J_up = np.array([], dtype=int)
            self._J_dp = np.array([], dtype=int)
            self._z_p = np.array([], dtype=float)
            self._ap_h = np.array([], dtype=float)
            self._ap_q = np.array([], dtype=float)
            self._dHp_max = np.array([], dtype=float)
            self._dHp_min = np.array([], dtype=float)
            self.n_p = 0
            self._Qp = np.array([], dtype=float)
            self._alpha_p = np.array([], dtype=float)
            self._beta_p = np.array([], dtype=float)
            self._chi_p = np.array([], dtype=float)
            self._alpha_upm = np.array([], dtype=float)
            self._beta_dpl = np.array([], dtype=float)
            self._chi_upl = np.array([], dtype=float)
            self._chi_dpm = np.array([], dtype=float)
        # Enforce minimum depth
        self._h_Ik = np.maximum(self._h_Ik, self.min_depth)
        # Computational arrays
        self._A_ik = np.zeros(self._ik.size)
        self._Pe_ik = np.zeros(self._ik.size)
        self._R_ik = np.zeros(self._ik.size)
        self._B_ik = np.zeros(self._ik.size)
        # Node velocities
        self._u_ik = np.zeros(self._ik.size, dtype=float)
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
        self._k_1k = np.cumsum(self.nk - 1) - (self.nk[0] - 1)
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
        self._z_inv_uk = np.copy(self._z_inv_Ik[self._I_1k])
        # Coefficients for head at downstream ends of superlink k
        self._J_dk = self.superlinks['sj_1'].values.astype(int)
        self._z_inv_dk = np.copy(self._z_inv_Ik[self._I_Np1k])
        # Sparse matrix coefficients
        if sparse:
            self.A = scipy.sparse.lil_matrix((self.M, self.M))
        else:
            self.A = np.zeros((self.M, self.M))
        self.b = np.zeros(self.M)
        self.D = np.zeros(self.M)
        self.bc = self.superjunctions['bc'].values.astype(bool)
        if sparse:
            self.B = scipy.sparse.lil_matrix((self.M, self.n_o))
            self.O = scipy.sparse.lil_matrix((self.M, self.M))
            self.W = scipy.sparse.lil_matrix((self.M, self.M))
            self.P = scipy.sparse.lil_matrix((self.M, self.M))
        else:
            self.B = np.zeros((self.M, self.n_o))
            self.O = np.zeros((self.M, self.M))
            self.W = np.zeros((self.M, self.M))
            self.P = np.zeros((self.M, self.M))
        # TODO: Should these be size NK?
        self._theta_uk = np.ones(self.NK)
        self._theta_dk = np.ones(self.NK)
        self._alpha_ukm = np.zeros(self.M, dtype=float)
        self._beta_dkl = np.zeros(self.M, dtype=float)
        self._chi_ukl = np.zeros(self.M, dtype=float)
        self._chi_dkm = np.zeros(self.M, dtype=float)
        self._k = self.superlinks.index.values
        self._A_sj = np.zeros(self.M, dtype=float)
        self._V_sj = np.zeros(self.M, dtype=float)
        self._F_jj = np.zeros(self.M, dtype=float)
        # TODO: Allow initial input to be specified
        self._Q_0j = 0
        # Set upstream and downstream superlink variables
        self._Q_uk = self._Q_ik[self._i_1k]
        self._Q_dk = self._Q_ik[self._i_nk]
        self._h_uk = self._h_Ik[self._I_1k]
        self._h_dk = self._h_Ik[self._I_Np1k]
        # Other parameters
        self._O_diag = np.zeros(self.M)
        self._W_diag = np.zeros(self.M)
        self._P_diag = np.zeros(self.M)
        # Superlink end hydraulic geometries
        self._A_uk = np.copy(self._A_ik[self._i_1k])
        self._A_dk = np.copy(self._A_ik[self._i_nk])
        self._B_uk = np.copy(self._B_ik[self._i_1k])
        self._B_dk = np.copy(self._B_ik[self._i_nk])
        self._Pe_uk = np.copy(self._Pe_ik[self._i_1k])
        self._Pe_dk = np.copy(self._Pe_ik[self._i_nk])
        self._R_uk = np.copy(self._R_ik[self._i_1k])
        self._R_dk = np.copy(self._R_ik[self._i_nk])
        self._link_start = np.zeros(self._ik.size, dtype=bool)
        self._link_end = np.zeros(self._ik.size, dtype=bool)
        self._link_start[self._i_1k] = True
        self._link_end[self._i_nk] = True
        self._h_c = np.zeros(self.NK)
        self._h_n = np.zeros(self.NK)
        # Set up hydraulic geometry computations
        self.configure_storages()
        self.configure_hydraulic_geometry()
        # Get superlink lengths
        self._dx_k = np.zeros(self.NK, dtype=float)
        np.add.at(self._dx_k, self._ki, self._dx_ik)
        self._Q_k = np.zeros(self.NK, dtype=float)
        self._A_k = np.zeros(self.NK, dtype=float)
        self._dt_ck = np.ones(self.NK, dtype=float)
        self._Q_in = np.zeros(self.M, dtype=float)
        # Initialize state dictionary
        self.states = {}
        # Iteration counter
        self.iter_count = 0
        self.t = 0
        # Compute bandwidth
        self._compute_bandwidth()
        # Initialize to stable state
        self.step(dt=1e-6, first_time=True)
        # Reset iteration counter
        self.iter_count = 0
        self.t = 0

    @property
    def h_Ik(self):
        return self._h_Ik

    @h_Ik.setter
    def h_Ik(self, value):
        self._h_Ik = np.asarray(value)

    @property
    def Q_ik(self):
        return self._Q_ik

    @Q_ik.setter
    def Q_ik(self, value):
        self._Q_ik = np.asarray(value)

    @property
    def Q_uk(self):
        return self._Q_uk

    @Q_uk.setter
    def Q_uk(self, value):
        self._Q_uk = np.asarray(value)

    @property
    def Q_dk(self):
        return self._Q_dk

    @Q_dk.setter
    def Q_dk(self, value):
        self._Q_dk = np.asarray(value)

    @property
    def Q_o(self):
        return self._Qo

    @Q_o.setter
    def Q_o(self, value):
        self._Qo = np.asarray(value)

    @property
    def Q_w(self):
        return self._Qw

    @Q_w.setter
    def Q_w(self, value):
        self._Qw = np.asarray(value)

    @property
    def Q_p(self):
        return self._Qp

    @Q_p.setter
    def Q_p(self, value):
        self._Qp = np.asarray(value)

    @property
    def A_ik(self):
        return self._A_ik

    @A_ik.setter
    def A_ik(self, value):
        self._A_ik = np.asarray(value)

    @property
    def Pe_ik(self):
        return self._Pe_ik

    @Pe_ik.setter
    def Pe_ik(self, value):
        self._Pe_ik = np.asarray(value)

    @property
    def R_ik(self):
        return self._R_ik

    @R_ik.setter
    def R_ik(self, value):
        self._R_ik = np.asarray(value)

    @property
    def B_ik(self):
        return self._B_ik

    @B_ik.setter
    def B_ik(self, value):
        self._B_ik = np.asarray(value)

    @property
    def A_sj(self):
        return self._A_sj

    @A_sj.setter
    def A_sj(self, value):
        self._A_sj = np.asarray(value)

    @property
    def V_sj(self):
        return self._V_sj

    @V_sj.setter
    def V_sj(self, value):
        self._V_sj = np.asarray(value)

    @property
    def z_inv_j(self):
        return self._z_inv_j

    @z_inv_j.setter
    def z_inv_j(self, value):
        self._z_inv_j = np.asarray(value)

    @property
    def z_inv_uk(self):
        return self._z_inv_uk

    @z_inv_uk.setter
    def z_inv_uk(self, value):
        self._z_inv_uk = np.asarray(value)

    @property
    def z_inv_dk(self):
        return self._z_inv_dk

    @z_inv_dk.setter
    def z_inv_dk(self, value):
        self._z_inv_dk = np.asarray(value)

    @property
    def x_Ik(self):
        return self._x_Ik

    @x_Ik.setter
    def x_Ik(self, value):
        self._x_Ik = np.asarray(value)

    @property
    def adjacency_matrix(self, J_u=None, J_d=None, symmetric=True):
        M = self.M
        # TODO: Maybe a cleaner way of doing this
        if (J_u is None) or (J_d is None):
            superlinks = self.superlinks
            orifices = self.orifices
            weirs = self.weirs
            pumps = self.pumps
            # Create array of upstream and downstream indices
            J_u = np.concatenate([elem['sj_0'].values for elem in
                                (superlinks, orifices, weirs, pumps)
                                if elem is not None])
            J_d = np.concatenate([elem['sj_1'].values for elem in
                                (superlinks, orifices, weirs, pumps)
                                if elem is not None])
        At = np.zeros((M, M))
        At[J_u, J_d] = 1
        if symmetric:
            At[J_d, J_u] = 1
        return At

    def _compute_bandwidth(self):
        At = self.adjacency_matrix
        bandwidth = 0
        for k in range(1, len(At)):
            if np.diag(At, k=k).any():
                bandwidth = k
            else:
                break
        self.bandwidth = bandwidth

    def _to_banded(self):
        At = self.adjacency_matrix
        At = scipy.sparse.csgraph.csgraph_from_dense(At)
        permutations = scipy.sparse.csgraph.reverse_cuthill_mckee(At)
        return permutations

    def _initialize_with_permuted_columns(self):
        # NOTE: This must be called during initialization only
        # Import instance variables
        superjunctions = self.superjunctions
        superlinks = self.superlinks
        orifices = self.orifices
        weirs = self.weirs
        pumps = self.pumps
        # Find permutation array
        permutations = self._to_banded()
        perm_inv = np.argsort(permutations)
        # Permute superjunctions
        # TODO: Should id be permuted, or should it stay constant?
        superjunctions['id'] = perm_inv[superjunctions['id'].values]
        superjunctions.index = superjunctions['id'].values
        superjunctions = superjunctions.sort_index()
        if superlinks is not None:
            superlinks['sj_0'] = perm_inv[superlinks['sj_0'].values]
            superlinks['sj_1'] = perm_inv[superlinks['sj_1'].values]
        if orifices is not None:
            orifices['sj_0'] = perm_inv[orifices['sj_0'].values]
            orifices['sj_1'] = perm_inv[orifices['sj_1'].values]
        if weirs is not None:
            weirs['sj_0'] = perm_inv[weirs['sj_0'].values]
            weirs['sj_1'] = perm_inv[weirs['sj_1'].values]
        if pumps is not None:
            pumps['sj_0'] = perm_inv[pumps['sj_0'].values]
            pumps['sj_1'] = perm_inv[pumps['sj_1'].values]
        # TODO: Remember to permute weirs/orifices/pumps too
        # Export instance variables
        self.superjunctions = superjunctions
        self.superlinks = superlinks
        self.orifices = orifices
        self.weirs = weirs
        self.pumps = pumps
        self.permutations = permutations

    def _configure_internals_variable(self, njunctions_fixed=4):
        # Import instance variables
        superlinks = self.superlinks            # Table of superlinks
        superjunctions = self.superjunctions    # Table of superjunctions
        # Set parameters
        njunctions = njunctions_fixed + 1
        nlinks = njunctions - 1
        link_ncols = 15
        junction_ncols = 5
        n_superlinks = len(superlinks)
        NJ = njunctions * n_superlinks
        NL = nlinks * n_superlinks
        elems = np.repeat(njunctions_fixed, n_superlinks)
        total_elems = elems + 1
        upstream_nodes = np.cumsum(total_elems) - total_elems
        downstream_nodes = np.cumsum(total_elems) - 2
        # Configure links
        links = pd.DataFrame(np.zeros((NL, link_ncols)))
        links.columns = ['A_c', 'C', 'Q_0', 'ctrl', 'dx', 'g1', 'g2', 'g3', 'g4',
                        'id', 'j_0', 'j_1', 'k', 'n', 'shape']
        links['A_c'] = np.repeat(superlinks['A_c'].values, nlinks)
        links['C'] = np.repeat(superlinks['C'].values, nlinks)
        links['Q_0'] = np.repeat(superlinks['Q_0'].values, nlinks)
        links['ctrl'] = np.repeat(superlinks['ctrl'].values, nlinks)
        links['k'] = np.repeat(superlinks.index.values, nlinks)
        links['shape'] = np.repeat(superlinks['shape'].values, nlinks)
        links['n'] = np.repeat(superlinks['n'].values, nlinks)
        links['g1'] = np.repeat(superlinks['g1'].values, nlinks)
        links['g2'] = np.repeat(superlinks['g2'].values, nlinks)
        links['g3'] = np.repeat(superlinks['g3'].values, nlinks)
        links['g4'] = np.repeat(superlinks['g4'].values, nlinks)
        links['id'] = links.index.values
        j = np.arange(NJ)
        links['j_0'] = np.delete(j, downstream_nodes + 1)
        links['j_1'] = np.delete(j, upstream_nodes)
        # Configure junctions
        junctions = pd.DataFrame(np.zeros((NJ, junction_ncols)))
        junctions.columns = ['A_s', 'h_0', 'id', 'k', 'z_inv']
        junctions['A_s'] = np.repeat(superlinks['A_s'].values, njunctions)
        junctions['h_0'] = np.repeat(superlinks['h_0'].values, njunctions)
        junctions['k'] = np.repeat(superlinks.index.values, njunctions)
        junctions['id'] = junctions.index.values
        # Configure internal variables
        x = np.zeros(NJ)
        z = np.zeros(NJ)
        dx = np.zeros(NL)
        h = junctions['h_0'].values
        Q = links['Q_0'].values
        xx = x.reshape(-1, njunctions)
        zz = z.reshape(-1, njunctions)
        dxdx = dx.reshape(-1, nlinks)
        hh = h.reshape(-1, njunctions)
        QQ = Q.reshape(-1, nlinks)
        dx_j = superlinks['dx'].values
        _z_inv_j = superjunctions['z_inv'].values.astype(float)
        inoffset = superlinks['in_offset'].values.astype(float)
        outoffset = superlinks['out_offset'].values.astype(float)
        _J_uk = superlinks['sj_0'].values.astype(int)
        _J_dk = superlinks['sj_1'].values.astype(int)
        if (njunctions % 2):
            _x0 = (dx_j / 2)
        else:
            _x0 = (dx_j / 2) + (dx_j / nlinks / 2)
        xx[:, :-1] = np.vstack([np.linspace(0, i, njunctions - 1)
                                for i in dx_j])
        xx[:, -1] = _x0
        _b0 = _z_inv_j[_J_uk] + inoffset
        _b1 = _z_inv_j[_J_dk] + outoffset
        _m = (_b1 - _b0) / dx_j
        _z0 = _m * _x0 + _b0
        zz[:] = xx * _m.reshape(-1, 1) + _b0.reshape(-1, 1)
        ix = np.argsort(xx)
        _fixed = (ix < nlinks)
        r, c = np.where(~_fixed)
        cm1 = ix[r, c - 1]
        cp1 = ix[r, c + 1]
        frac = (xx[r, xx.shape[1] - 1] - xx[r, cm1]) / (xx[r, cp1] - xx[r, cm1])
        # Set depths and flows
        hh[:, -1] = (1 - frac) * hh[r, cm1] + (frac) * hh[r, cp1]
        QQ[:, -1] = QQ[r, cm1]
        # Write new variables
        xx = np.take_along_axis(xx, ix, axis=-1)
        zz = np.take_along_axis(zz, ix, axis=-1)
        hh = np.take_along_axis(hh, ix, axis=-1)
        dxdx = np.diff(xx)
        junctions['z_inv'] = zz.ravel()
        junctions['h_0'] = hh.ravel()
        links['dx'] = dxdx.ravel()
        links['Q_0'] = QQ.ravel()
        # Set start and end junctions on superlinks
        superlinks['j_0'] = links.groupby('k')['j_0'].min()
        superlinks['j_1'] = links.groupby('k')['j_1'].max()
        # Export instance variables
        self.junctions = junctions
        self.links = links
        self.superlinks = superlinks
        self._fixed = _fixed
        self._elem_pos = c
        self._b0 = _b0
        self._b1 = _b1
        self._m = _m
        self._x0 = _x0
        self._z0 = _z0

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
    def V_1k(self, P_1k, D_2k, c_1k, T_1k, a_1k=0.0, D_1k=0.0):
        """
        Compute forward recurrence coefficient 'V' for node 1, superlink k.
        """
        num = P_1k - D_2k * c_1k + D_1k * a_1k
        den = T_1k
        return num, den

    @safe_divide
    def W_1k(self, A_1k, T_1k, a_1k=0.0, E_1k=0.0, g=9.81):
        """
        Compute forward recurrence coefficient 'W' for node 1, superlink k.
        """
        num = g * A_1k - E_1k * a_1k
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
    def Y_Nk(self, P_nk, D_Nk, a_nk, O_nk, c_nk=0.0, D_Np1k=0.0):
        """
        Compute backward recurrence coefficient 'Y' for node N, superlink k.
        """
        num = P_nk + D_Nk * a_nk - D_Np1k * c_nk
        den = O_nk
        return num, den

    @safe_divide
    def Z_Nk(self, A_nk, O_nk, c_nk=0.0, E_Np1k=0.0, g=9.81):
        """
        Compute backward recurrence coefficient 'Z' for node N, superlink k.
        """
        num = E_Np1k * c_nk - g * A_nk
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

    @safe_divide
    def kappa_uk(self, A_uk, dH_uk, Q_uk_t, B_uk):
        """
        Compute head coefficient 'kappa' for upstream end of superlink k
        """
        num = 2 * A_uk * dH_uk
        den = Q_uk_t * (2 * dH_uk * B_uk - A_uk)
        return num, den

    @safe_divide
    def lambda_uk(self, A_uk, dH_uk, B_uk):
        """
        Compute head coefficient 'lambda' for upstream end of superlink k
        """
        num = - A_uk
        den = 2 * dH_uk * B_uk - A_uk
        return num, den

    @safe_divide
    def mu_uk(self, A_uk, dH_uk, B_uk, z_inv_uk):
        """
        Compute head coefficient 'mu' for upstream end of superlink k
        """
        num = A_uk * (dH_uk + z_inv_uk)
        den = 2 * dH_uk * B_uk - A_uk
        return num, den

    @safe_divide
    def kappa_dk(self, A_dk, dH_dk, Q_dk_t, B_dk):
        """
        Compute head coefficient 'kappa' for downstream end of superlink k
        """
        num = 2 * A_dk * dH_dk
        den = Q_dk_t * (2 * dH_dk * B_dk + A_dk)
        return num, den

    @safe_divide
    def lambda_dk(self, A_dk, dH_dk, B_dk):
        """
        Compute head coefficient 'lambda' for downstream end of superlink k
        """
        num = A_dk
        den = 2 * dH_dk * B_dk + A_dk
        return num, den

    @safe_divide
    def mu_dk(self, A_dk, dH_dk, B_dk, z_inv_dk):
        """
        Compute head coefficient 'mu' for downstream end of superlink k
        """
        num = A_dk * (dH_dk - z_inv_dk)
        den = 2 * dH_dk * B_dk + A_dk
        return num, den

    def D_k_star(self, X_1k, kappa_uk, U_Nk, kappa_dk, Z_1k, W_Nk):
        """
        Compute superlink boundary condition coefficient 'D_k_star'.
        """
        t_0 = (X_1k * kappa_uk - 1) * (U_Nk * kappa_dk - 1)
        t_1 = (Z_1k * kappa_dk) * (W_Nk * kappa_uk)
        return t_0 - t_1

    @safe_divide
    def alpha_uk(self, U_Nk, kappa_dk, X_1k, Z_1k, W_Nk, D_k_star, lambda_uk, theta_uk=1):
        """
        Compute superlink boundary condition coefficient 'alpha' for upstream end
        of superlink k.
        """
        num = theta_uk * ((1 - U_Nk * kappa_dk) * X_1k * lambda_uk
                          + (Z_1k * kappa_dk * W_Nk * lambda_uk))
        den = D_k_star
        return num, den

    @safe_divide
    def beta_uk(self, U_Nk, kappa_dk, Z_1k, W_Nk, D_k_star, lambda_dk, theta_dk=1):
        """
        Compute superlink boundary condition coefficient 'beta' for upstream end
        of superlink k.
        """
        num = theta_dk * ((1 - U_Nk * kappa_dk) * Z_1k * lambda_dk
               + (Z_1k * kappa_dk * U_Nk * lambda_dk))
        den = D_k_star
        return num, den

    @safe_divide
    def chi_uk(self, U_Nk, kappa_dk, Y_1k, X_1k, mu_uk, Z_1k,
               mu_dk, V_Nk, W_Nk, D_k_star, theta_uk=1, theta_dk=1):
        """
        Compute superlink boundary condition coefficient 'chi' for upstream end
        of superlink k.
        """
        t_0 = (1 - U_Nk * kappa_dk) * (Y_1k + theta_uk * X_1k * mu_uk + theta_dk * Z_1k * mu_dk)
        t_1 = (Z_1k * kappa_dk) * (V_Nk + theta_uk * W_Nk * mu_uk + theta_dk * U_Nk * mu_dk)
        num = t_0 + t_1
        den = D_k_star
        return num, den

    @safe_divide
    def alpha_dk(self, X_1k, kappa_uk, W_Nk, D_k_star, lambda_uk, theta_uk=1):
        """
        Compute superlink boundary condition coefficient 'alpha' for downstream end
        of superlink k.
        """
        num = theta_uk * ((1 - X_1k * kappa_uk) * W_Nk * lambda_uk
               + (W_Nk * kappa_uk * X_1k * lambda_uk))
        den = D_k_star
        return num, den

    @safe_divide
    def beta_dk(self, X_1k, kappa_uk, U_Nk, W_Nk, Z_1k, D_k_star, lambda_dk, theta_dk=1):
        """
        Compute superlink boundary condition coefficient 'beta' for downstream end
        of superlink k.
        """
        num = theta_dk * ((1 - X_1k * kappa_uk) * U_Nk * lambda_dk
               + (W_Nk * kappa_uk * Z_1k * lambda_dk))
        den = D_k_star
        return num, den

    @safe_divide
    def chi_dk(self, X_1k, kappa_uk, V_Nk, W_Nk, mu_uk, U_Nk,
               mu_dk, Y_1k, Z_1k, D_k_star, theta_uk=1, theta_dk=1):
        """
        Compute superlink boundary condition coefficient 'chi' for downstream end
        of superlink k.
        """
        t_0 = (1 - X_1k * kappa_uk) * (V_Nk + theta_uk * W_Nk * mu_uk + theta_dk * U_Nk * mu_dk)
        t_1 = (W_Nk * kappa_uk) * (Y_1k + theta_uk * X_1k * mu_uk + theta_dk * Z_1k * mu_dk)
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

    def B_j(self, J_uo, J_do, Ao, H_j, Co=0.67, g=9.81):
        dH_u = H_j[J_uo] - H_j[J_do]
        dH_d = H_j[J_do] - H_j[J_uo]
        # TODO: Why are these reversed?
        # TODO: It's because everything gets multiplied by -1 in solution matrix eqn
        Qo_d = Co * Ao * np.sign(dH_u) * np.sqrt(2 * g * np.abs(dH_u))
        Qo_u = Co * Ao * np.sign(dH_d) * np.sqrt(2 * g * np.abs(dH_d))
        return Qo_u, Qo_d

    @safe_divide
    def gamma_o(self, Q_o_t, Ao, Co=0.67, g=9.81):
        """
        Compute flow coefficient 'gamma' for orifice o.
        """
        num = 2 * g * Co**2 * Ao**2
        den = np.abs(Q_o_t)
        return num, den

    @safe_divide
    def gamma_w(self, Q_w_t, H_w_t, L_w, s_w, Cwr=1.838, Cwt=1.380):
        """
        Compute flow coefficient 'gamma' for weir w.
        """
        num = (Cwr * L_w * H_w_t + Cwt * s_w * H_w_t**2)**2
        den = np.abs(Q_w_t)
        return num, den

    @safe_divide
    def gamma_p(self, Q_p_t, dH_p_t, a_q=1.0, a_h=1.0):
        """
        Compute flow coefficient 'gamma' for pump p.
        """
        num = a_q**2 * np.abs(dH_p_t)
        den = a_h**2 * np.abs(Q_p_t)
        return num, den

    def configure_hydraulic_geometry(self):
        """
        Prepare data structures for hydraulic geometry computations.
        """
        # Import instance variables
        transects = self.transects          # Table of transects
        _shape_ik = self._shape_ik          # Shape of link ik
        _shape_o = self._shape_o
        _transect_ik = self._transect_ik    # Transect associated with link ik
        _link_start = self._link_start      # Link is first link in superlink k
        _link_end = self._link_end          # Link is last link in superlink k
        _geom_numbers = pipedream_solver.geometry.geom_code
        nk = self.nk
        n_o = self.n_o
        # Set attributes
        _geom_factory = {}
        _geom_factory_o = {}
        _transect_factory = {}
        _transect_indices = None
        _uk_geom_factory = {}
        _uk_transect_factory = {}
        _uk_transect_indices = None
        _dk_geom_factory = {}
        _dk_transect_factory = {}
        _dk_transect_indices = None
        # Handle regular geometries
        _is_irregular = _shape_ik.str.lower() == 'irregular'
        _has_irregular = _is_irregular.any()
        _uk_has_irregular = (_link_start & _is_irregular).any()
        _dk_has_irregular = (_link_end & _is_irregular).any()
        _unique_geom = set(_shape_ik.str.lower().unique())
        _unique_geom.discard('irregular')
        _regular_shapes = _shape_ik[~_is_irregular]
        _geom_indices = pd.Series(_regular_shapes.index,
                                  index=_regular_shapes.str.lower().values)
        for geom in _unique_geom:
            _ik_g = _geom_indices.loc[[geom]].values
            _geom_factory[geom] = _ik_g
            _uk_geom_factory[geom] = _ik_g[_link_start[_ik_g]]
            _dk_geom_factory[geom] = _ik_g[_link_end[_ik_g]]
        # Handle irregular geometries
        if _has_irregular:
            _irregular_transects = _transect_ik[_is_irregular]
            _transect_indices = pd.Series(_irregular_transects.index,
                                          index=_irregular_transects.values)
            _uk_transect_indices = _transect_indices[_link_start[_transect_indices]]
            _dk_transect_indices = _transect_indices[_link_end[_transect_indices]]
            for transect_name, transect in transects.items():
                _transect_factory[transect_name] = pipedream_solver.geometry.Irregular(**transect)
        # Create array of geom codes
        # TODO: Should have a variable that gives total number of links instead of summing
        _geom_codes = np.zeros(nk.sum(), dtype=int)
        for geom, indices in _geom_factory.items():
            _geom_codes[indices] = _geom_numbers[geom]
        # NOTE: Handle case for elliptical geometry
        _ellipse_ix = np.flatnonzero(_geom_codes ==
                                     pipedream_solver.geometry.geom_code['elliptical'])
        # Handle orifices
        if n_o:
            _unique_geom_o = set(_shape_o.str.lower().unique())
            _geom_indices_o = pd.Series(_shape_o.index, index=_shape_o.str.lower().values)
            for geom in _unique_geom_o:
                _o_g = _geom_indices_o.loc[[geom]].values
                _geom_factory_o[geom] = _o_g
            _geom_codes_o = np.zeros(n_o, dtype=int)
            for geom, indices in _geom_factory_o.items():
                _geom_codes_o[indices] = _geom_numbers[geom]
            # Export instance variables
            self._geom_factory_o = _geom_factory_o
            self._geom_codes_o = _geom_codes_o
        # Export instance variables
        self._has_irregular = _has_irregular
        self._geom_factory = _geom_factory
        self._transect_factory = _transect_factory
        self._transect_indices = _transect_indices
        self._uk_has_irregular = _uk_has_irregular
        self._dk_has_irregular = _dk_has_irregular
        self._uk_geom_factory = _uk_geom_factory
        self._dk_geom_factory = _dk_geom_factory
        self._uk_transect_indices = _uk_transect_indices
        self._dk_transect_indices = _dk_transect_indices
        self._geom_codes = _geom_codes
        self._ellipse_ix = _ellipse_ix

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
                    _storage_factory[table_name] = pipedream_solver.storage.Tabular(**table)
        # Export instance variables
        self._storage_indices = _storage_indices
        self._storage_factory = _storage_factory
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
            generator = getattr(pipedream_solver.geometry, Geom)
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
        _uk_geom_factory = self._uk_geom_factory
        _transect_factory = self._transect_factory
        _uk_transect_indices = self._uk_transect_indices
        _uk_has_irregular = self._uk_has_irregular
        # Compute hydraulic geometry for regular geometries
        for geom, indices in _uk_geom_factory.items():
            Geom = geom.title()
            _ik_g = indices
            _ki_g = _ki[_ik_g]
            _Ik_g = _Ik[_ik_g]
            generator = getattr(pipedream_solver.geometry, Geom)
            _g1_g = _g1_ik[_ik_g]
            _g2_g = _g2_ik[_ik_g]
            _g3_g = _g3_ik[_ik_g]
            _h_Ik_g = _h_Ik[_Ik_g]
            _H_j_g = H_j[_J_uk[_ki_g]] - _z_inv_uk[_ki_g]
            if area == 'max':
                _h_u_g = np.maximum(_h_Ik_g, _H_j_g)
                _A_uk[_ki_g] = generator.A_ik(_h_u_g, _h_u_g,
                                              g1=_g1_g, g2=_g2_g, g3=_g3_g)
                _B_uk[_ki_g] = generator.B_ik(_h_u_g, _h_u_g,
                                              g1=_g1_g, g2=_g2_g, g3=_g3_g)
            elif area == 'avg':
                _A_uk[_ki_g] = generator.A_ik(_h_Ik_g, _H_j_g,
                                            g1=_g1_g, g2=_g2_g, g3=_g3_g)
                _B_uk[_ki_g] = generator.B_ik(_h_Ik_g, _H_j_g,
                                              g1=_g1_g, g2=_g2_g, g3=_g3_g)
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
        _dk_geom_factory = self._dk_geom_factory
        _transect_factory = self._transect_factory
        _dk_transect_indices = self._dk_transect_indices
        _dk_has_irregular = self._dk_has_irregular
        # Compute hydraulic geometry for regular geometries
        for geom, indices in _dk_geom_factory.items():
            Geom = geom.title()
            _ik_g = indices
            _ki_g = _ki[_ik_g]
            _Ip1k_g = _Ip1k[_ik_g]
            generator = getattr(pipedream_solver.geometry, Geom)
            _g1_g = _g1_ik[_ik_g]
            _g2_g = _g2_ik[_ik_g]
            _g3_g = _g3_ik[_ik_g]
            _h_Ip1k_g = _h_Ik[_Ip1k_g]
            _H_j_g = H_j[_J_dk[_ki_g]] - _z_inv_dk[_ki_g]
            if area == 'max':
                _h_d_g = np.maximum(_h_Ip1k_g, _H_j_g)
                _A_dk[_ki_g] = generator.A_ik(_h_d_g, _h_d_g,
                                              g1=_g1_g, g2=_g2_g, g3=_g3_g)
                _B_dk[_ki_g] = generator.B_ik(_h_d_g, _h_d_g,
                                              g1=_g1_g, g2=_g2_g, g3=_g3_g)
            elif area == 'avg':
                _A_dk[_ki_g] = generator.A_ik(_h_Ip1k_g, _H_j_g,
                                              g1=_g1_g, g2=_g2_g, g3=_g3_g)
                _B_dk[_ki_g] = generator.B_ik(_h_Ip1k_g, _H_j_g,
                                              g1=_g1_g, g2=_g2_g, g3=_g3_g)
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
        Compute hydraulic geometry for each orifice.
        """
        # Import instance variables
        _Ao = self._Ao             # Flow area at link ik
        _g1_o = self._g1_o           # Geometry 1 of link ik (vertical)
        _g2_o = self._g2_o           # Geometry 2 of link ik (horizontal)
        _g3_o = self._g3_o           # Geometry 3 of link ik (other)
        _geom_factory_o = self._geom_factory_o
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
        # Compute hydraulic geometry for regular geometries
        for geom, indices in _geom_factory_o.items():
            Geom = geom.title()
            _o_g = indices
            generator = getattr(pipedream_solver.geometry, Geom)
            _g1_g = _g1_o[_o_g] * u[_o_g]
            _g2_g = _g2_o[_o_g]
            _g3_g = _g3_o[_o_g]
            _Ao[_o_g] = generator.A_ik(h_e, h_e,
                                       g1=_g1_g, g2=_g2_g, g3=_g3_g)
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
        # Compute storage areas
        _h_j = np.maximum(H_j - _z_inv_j, min_depth)
        if _functional.any():
            generator = getattr(pipedream_solver.storage, 'Functional')
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

    def compute_storage_volumes(self):
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
        _V_sj = self._V_sj                          # Surface area at superjunction j
        # Compute storage areas
        _h_j = np.maximum(H_j - _z_inv_j, min_depth)
        if _functional.any():
            generator = getattr(pipedream_solver.storage, 'Functional')
            _V_sj[_functional] = generator.V_sj(_h_j[_functional],
                                                _storage_a[_functional],
                                                _storage_b[_functional],
                                                _storage_c[_functional])
        if _tabular.any():
            for storage_name, generator in _storage_factory.items():
                _j_g = _storage_indices.loc[[storage_name]].values
                _V_sj[_j_g] = generator.V_sj(_h_j[_j_g])
        # Export instance variables
        self._V_sj = _V_sj
        # TODO: Temporary to maintain compatibility
        return _V_sj

    def node_velocities(self):
        """
        Compute velocity of flow at each link and junction.
        """
        # Import instance variables
        _ik = self._ik                       # Link index
        _Ik = self._Ik                       # Junction index
        _Ip1k = self._Ip1k                   # Next junction index
        _h_Ik = self._h_Ik                   # Depth at junction Ik
        _A_ik = self._A_ik                   # Flow area at link ik
        _Pe_ik = self._Pe_ik                 # Hydraulic perimeter at link ik
        _R_ik = self._R_ik                   # Hydraulic radius at link ik
        _B_ik = self._B_ik                   # Top width at link ik
        _Q_ik = self._Q_ik                   # Flow rate at link ik
        _u_Ik = self._u_Ik                   # Flow velocity at junction Ik
        _u_Ip1k = self._u_Ip1k               # Flow velocity at junction I + 1k
        backward_I_i = self.backward_I_i     # Index of link before junction Ik
        forward_I_i = self.forward_I_i       # Index of link after junction Ik
        _dx_ik = self._dx_ik                 # Length of link ik
        # Determine start and end nodes
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
        """
        Compute Froude number for each link, and the average Froude number for each link.
        Compute the inertial damping coefficient, sigma_ik.
        """
        # Import instance variables
        _u_ik = self._u_ik    # Flow velocity at link ik
        _A_ik = self._A_ik    # Flow area at link ik
        _B_ik = self._B_ik    # Top width at link ik
        _ki = self._ki        # Index of superlink containing link ik
        _kI = self._kI        # Index of superlink containing junction Ik
        NK = self.NK          # Number of superlinks
        nk = self.nk          # Number of links in each superlink
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
        _a_ik = self.a_ik(_u_Ik, _sigma_ik)
        _c_ik = self.c_ik(_u_Ip1k, _sigma_ik)
        _b_ik = self.b_ik(_dx_ik, _dt, _n_ik, _Q_ik, _A_ik, _R_ik,
                          _A_c_ik, _C_ik, _a_ik, _c_ik, _ctrl)
        if first_iter:
            _P_ik = self.P_ik(_Q_ik, _dx_ik, _dt, _A_ik, _S_o_ik,
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
        _I = self._I                         # Indices of all junctions
        start_nodes = self.start_nodes       # Upstream nodes of superlinks
        end_nodes = self.end_nodes           # Downstream nodes of superlinks
        middle_nodes = self.middle_nodes     # Non-boundary nodes of superlinks
        forward_I_i = self.forward_I_i       # Index of link after junction Ik
        backward_I_i = self.backward_I_i     # Index of link before junction Ik
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
            _Q_0Ik = np.zeros(_I.size)
        # Compute E_Ik and D_Ik
        start_links = self.forward_I_i[start_nodes]
        end_links = self.backward_I_i[end_nodes]
        backward = self.backward_I_i[middle_nodes]
        forward = self.forward_I_i[middle_nodes]
        # TODO: Check control volumes
        _E_Ik[start_nodes] = self.E_Ik(_B_ik[start_links], _dx_ik[start_links],
                                        0.0, 0.0, _A_SIk[start_nodes], _dt)
        _E_Ik[end_nodes] = self.E_Ik(0.0, 0.0, _B_ik[end_links], _dx_ik[end_links],
                                        _A_SIk[end_nodes], _dt)
        _E_Ik[middle_nodes] = self.E_Ik(_B_ik[forward], _dx_ik[forward],
                                        _B_ik[backward], _dx_ik[backward],
                                        _A_SIk[middle_nodes], _dt)
        if first_iter:
            _D_Ik[start_nodes] = self.D_Ik(_Q_0Ik[start_nodes], _B_ik[start_links],
                                            _dx_ik[start_links], 0.0,
                                            0.0, _A_SIk[start_nodes],
                                            _h_Ik[start_nodes], _dt)
            _D_Ik[end_nodes] = self.D_Ik(_Q_0Ik[end_nodes], 0.0,
                                            0.0, _B_ik[end_links],
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
        """
        Compute forward recurrence coefficients: T_ik, U_Ik, V_Ik, and W_Ik.
        """
        # Import instance variables
        backward_I_I = self.backward_I_I  # Index of junction before junction Ik
        forward_I_I = self.forward_I_I    # Index of junction after junction Ik
        forward_I_i = self.forward_I_i    # Index of link after junction Ik
        _I_end = self._I_end              # Junction at downstream end of superlink (y/n)
        _I_1k = self._I_1k                # Index of first junction in each superlink
        _I_2k = self._I_2k                # Index of second junction in each superlink
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
        _end_method = self._end_method    # Method for computing flow at pipe ends
        # Compute coefficients for starting nodes
        _E_2k = _E_Ik[_I_2k]
        _D_2k = _D_Ik[_I_2k]
        if _end_method == 'o':
            _T_1k = self.T_1k(_a_ik[_i_1k], _b_ik[_i_1k], _c_ik[_i_1k])
            _U_1k = self.U_1k(_E_2k, _c_ik[_i_1k], _A_ik[_i_1k], _T_1k)
            _V_1k = self.V_1k(_P_ik[_i_1k], _D_2k, _c_ik[_i_1k], _T_1k)
            _W_1k = self.W_1k(_A_ik[_i_1k], _T_1k)
        else:
            _T_1k = self.T_1k(_a_ik[_i_1k], _b_ik[_i_1k], _c_ik[_i_1k])
            _U_1k = self.U_1k(_E_2k, _c_ik[_i_1k], _A_ik[_i_1k], _T_1k)
            _V_1k = self.V_1k(_P_ik[_i_1k], _D_2k, _c_ik[_i_1k], _T_1k,
                              _a_ik[_i_1k], _D_Ik[_I_1k])
            _W_1k = self.W_1k(_A_ik[_i_1k], _T_1k, _a_ik[_i_1k], _E_Ik[_I_1k])
        # I = 1, i = 1
        _T_ik[_i_1k] = _T_1k
        _U_Ik[_I_1k] = _U_1k
        _V_Ik[_I_1k] = _V_1k
        _W_Ik[_I_1k] = _W_1k
        # I = 2, i = 2
        _I_next = _I_2k[~_I_end[_I_2k]]
        # Loop from 2 -> Nk
        while _I_next.size:
            _Im1_next = backward_I_I[_I_next]
            _Ip1_next = forward_I_I[_I_next]
            _i_next = forward_I_i[_I_next]
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
        """
        Compute backward recurrence coefficients: O_ik, X_Ik, Y_Ik, and Z_Ik.
        """
        # Import instance variables
        backward_I_I = self.backward_I_I  # Index of junction before junction Ik
        forward_I_I = self.forward_I_I    # Index of junction after junction Ik
        forward_I_i = self.forward_I_i    # Index of link after junction Ik
        _I_start = self._I_start          # Junction at upstream end of superlink (y/n)
        _I_Np1k = self._I_Np1k            # Index of last junction in each superlink
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
        _end_method = self._end_method    # Method for computing flow at pipe ends
        _Z_Ik = self._Z_Ik                # Recurrence coefficient Z_Ik
        # Compute coefficients for starting nodes
        _E_Nk = _E_Ik[_I_Nk]
        _D_Nk = _D_Ik[_I_Nk]
        if _end_method == 'o':
            _O_nk = self.O_nk(_a_ik[_i_nk], _b_ik[_i_nk], _c_ik[_i_nk])
            _X_Nk = self.X_Nk(_A_ik[_i_nk], _E_Nk, _a_ik[_i_nk], _O_nk)
            _Y_Nk = self.Y_Nk(_P_ik[_i_nk], _D_Nk, _a_ik[_i_nk], _O_nk)
            _Z_Nk = self.Z_Nk(_A_ik[_i_nk], _O_nk)
        else:
            _O_nk = self.O_nk(_a_ik[_i_nk], _b_ik[_i_nk], _c_ik[_i_nk])
            _X_Nk = self.X_Nk(_A_ik[_i_nk], _E_Nk, _a_ik[_i_nk], _O_nk)
            _Y_Nk = self.Y_Nk(_P_ik[_i_nk], _D_Nk, _a_ik[_i_nk], _O_nk,
                              _c_ik[_i_nk], _D_Ik[_I_Np1k])
            _Z_Nk = self.Z_Nk(_A_ik[_i_nk], _O_nk, _c_ik[_i_nk], _E_Ik[_I_Np1k])
        # I = Nk, i = nk
        _O_ik[_i_nk] = _O_nk
        _X_Ik[_I_Nk] = _X_Nk
        _Y_Ik[_I_Nk] = _Y_Nk
        _Z_Ik[_I_Nk] = _Z_Nk
        # I = Nk-1, i = nk-1
        _I_next = backward_I_I[_I_Nk[~_I_start[_I_Nk]]]
        # Loop from Nk - 1 -> 1
        while _I_next.size:
            _Ip1_next = forward_I_I[_I_next]
            _i_next = forward_I_i[_I_next]
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
            _gamma_uk = self.gamma_uk(_Q_uk_t, _C_uk, _A_uk)
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
            _gamma_dk = self.gamma_dk(_Q_dk_t, _C_dk, _A_dk)
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
        # TODO: Remove?
        # _theta_uk = 1.
        # _theta_dk = 1.
        # Compute D_k_star
        _D_k_star = self.D_k_star(_X_1k, _kappa_uk, _U_Nk,
                                  _kappa_dk, _Z_1k, _W_Nk)
        # Compute upstream superlink flow coefficients
        _alpha_uk = self.alpha_uk(_U_Nk, _kappa_dk, _X_1k,
                                  _Z_1k, _W_Nk, _D_k_star,
                                  _lambda_uk, _theta_uk)
        _beta_uk = self.beta_uk(_U_Nk, _kappa_dk, _Z_1k,
                                _W_Nk, _D_k_star, _lambda_dk, _theta_dk)
        _chi_uk = self.chi_uk(_U_Nk, _kappa_dk, _Y_1k,
                              _X_1k, _mu_uk, _Z_1k,
                              _mu_dk, _V_Nk, _W_Nk,
                              _D_k_star, _theta_uk, _theta_dk)
        # Compute downstream superlink flow coefficients
        _alpha_dk = self.alpha_dk(_X_1k, _kappa_uk, _W_Nk,
                                  _D_k_star, _lambda_uk, _theta_uk)
        _beta_dk = self.beta_dk(_X_1k, _kappa_uk, _U_Nk,
                                _W_Nk, _Z_1k, _D_k_star,
                                _lambda_dk, _theta_dk)
        _chi_dk = self.chi_dk(_X_1k, _kappa_uk, _V_Nk,
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
        _orient_o = self._orient_o   # Orientation of orifice o (side/bottom)
        _y_max_o = self._y_max_o     # Maximum height of orifice o
        _Qo = self._Qo               # Current flow rate of orifice o
        _Co = self._Co               # Discharge coefficient of orifice o
        _Ao = self._Ao               # NOTE: Flow area of orifice now
        _tau_o = self._tau_o
        _alpha_o = self._alpha_o     # Orifice flow coefficient alpha_o
        _beta_o = self._beta_o       # Orifice flow coefficient beta_o
        _chi_o = self._chi_o         # Orifice flow coefficient chi_o
        # If no input signal, assume orifice is closed
        if u is None:
            u = np.zeros(self.n_o, dtype=float)
        # Specify orifice heads at previous timestep
        _H_uo = H_j[_J_uo]
        _H_do = H_j[_J_do]
        _z_inv_uo = _z_inv_j[_J_uo]
        # Create indicator functions
        _omega_o = (_H_uo >= _H_do).astype(float)
        # Compute universal coefficients
        _gamma_o = self.gamma_o(_Qo, _Ao, _Co)
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
        _beta_o[a] = - _gamma_o[a]
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
        # Specify weir heads at previous timestep
        _H_uw = H_j[_J_uw]
        _H_dw = H_j[_J_dw]
        _z_inv_uw = _z_inv_j[_J_uw]
        # Create indicator functions
        _omega_w = (_H_uw >= _H_dw).astype(float)
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
        _gamma_w = self.gamma_w(_Qw, _Hw, _L_w, _s_w, _Cwr, _Cwt)
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
        # Specify pump heads at previous timestep
        _H_up = H_j[_J_up]
        _H_dp = H_j[_J_dp]
        _z_inv_up = _z_inv_j[_J_up]
        # Create conditionals
        assert (_dHp_min <= _dHp_max).all()
        _dHp = _H_dp - _H_up
        cond_0 = _H_up > _z_inv_up + _z_p
        cond_1 = (_dHp > _dHp_min) & (_dHp < _dHp_max)
        _dHp[_dHp > _dHp_max] = _dHp_max
        _dHp[_dHp < _dHp_min] = _dHp_min
        # Compute universal coefficients
        _gamma_p = self.gamma_p(_Qp, _dHp, _ap_q, _ap_h)
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
        _A_sj = self._A_sj               # Surface area of superjunction j
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
        n_o = self.n_o                   # Number of orifices in system
        n_w = self.n_w                   # Number of weirs in system
        n_p = self.n_p                   # Number of pumps in system
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
        # Compute F_jj
        _alpha_ukm.fill(0)
        _beta_dkl.fill(0)
        np.add.at(_alpha_ukm, _J_uk, _alpha_uk)
        np.add.at(_beta_dkl, _J_dk, _beta_dk)
        _F_jj = self.F_jj(_A_sj, _dt, _beta_dkl, _alpha_ukm)
        # Set diagonal of A matrix
        i = np.arange(M)
        self.A[i[~bc], i[~bc]] = _F_jj[i[~bc]]
        self.A[i[bc], i[bc]] = 1
        # Compute off-diagonals of A Matrix
        bc_uk = bc[_J_uk]
        bc_dk = bc[_J_dk]
        self.A[_J_uk[~bc_uk], _J_dk[~bc_uk]] = 0.0
        self.A[_J_dk[~bc_dk], _J_uk[~bc_dk]] = 0.0
        np.add.at(self.A, (_J_uk[~bc_uk], _J_dk[~bc_uk]), _beta_uk[~bc_uk])
        np.add.at(self.A, (_J_dk[~bc_dk], _J_uk[~bc_dk]), -_alpha_dk[~bc_dk])
        # Compute G_j
        _chi_ukl.fill(0)
        _chi_dkm.fill(0)
        D.fill(0)
        np.add.at(_chi_ukl, _J_uk, _chi_uk)
        np.add.at(_chi_dkm, _J_dk, _chi_dk)
        D = -_chi_ukl + _chi_dkm
        # b = self.G_j(_A_sj, _dt, H_j, _Q_0j, _chi_ukl, _chi_dkm)
        # Compute control matrix
        if n_o:
            bc_uo = bc[_J_uo]
            bc_do = bc[_J_do]
            if implicit:
                _alpha_uo = _alpha_o
                _alpha_do = _alpha_o
                _beta_uo = _beta_o
                _beta_do = _beta_o
                _chi_uo = _chi_o
                _chi_do = _chi_o
                _alpha_uom.fill(0)
                _beta_dol.fill(0)
                _chi_uol.fill(0)
                _chi_dom.fill(0)
                _O_diag.fill(0)
                # Set diagonal
                np.add.at(_alpha_uom, _J_uo, _alpha_uo)
                np.add.at(_beta_dol, _J_do, _beta_do)
                _O_diag = -_beta_dol + _alpha_uom
                # Set off-diagonal
                self.O[i[~bc], i[~bc]] = _O_diag[i[~bc]]
                self.O[_J_uo[~bc_uo], _J_do[~bc_uo]] = 0.0
                self.O[_J_do[~bc_do], _J_uo[~bc_do]] = 0.0
                np.add.at(self.O, (_J_uo[~bc_uo], _J_do[~bc_uo]), _beta_uo[~bc_uo])
                np.add.at(self.O, (_J_do[~bc_do], _J_uo[~bc_do]), -_alpha_do[~bc_do])
                # Set right-hand side
                np.add.at(_chi_uol, _J_uo, _chi_uo)
                np.add.at(_chi_dom, _J_do, _chi_do)
                np.add.at(D, i[~bc], -_chi_uol[~bc] + _chi_dom[~bc])
            else:
                # TODO: Broken
                # _Qo_u, _Qo_d = self.B_j(_J_uo, _J_do, _Ao, H_j)
                # self.B[_J_uo[~bc_uo]] = _Qo_u[~bc_uo]
                # self.B[_J_do[~bc_do]] = _Qo_d[~bc_do]
                pass
        if n_w:
            bc_uw = bc[_J_uw]
            bc_dw = bc[_J_dw]
            if implicit:
                _alpha_uw = _alpha_w
                _alpha_dw = _alpha_w
                _beta_uw = _beta_w
                _beta_dw = _beta_w
                _chi_uw = _chi_w
                _chi_dw = _chi_w
                _alpha_uwm.fill(0)
                _beta_dwl.fill(0)
                _chi_uwl.fill(0)
                _chi_dwm.fill(0)
                _W_diag.fill(0)
                # Set diagonal
                np.add.at(_alpha_uwm, _J_uw, _alpha_uw)
                np.add.at(_beta_dwl, _J_dw, _beta_dw)
                _W_diag = -_beta_dwl + _alpha_uwm
                # Set off-diagonal
                self.W[i[~bc], i[~bc]] = _W_diag[i[~bc]]
                self.W[_J_uw[~bc_uw], _J_dw[~bc_uw]] = 0.0
                self.W[_J_dw[~bc_dw], _J_uw[~bc_dw]] = 0.0
                np.add.at(self.W, (_J_uw[~bc_uw], _J_dw[~bc_uw]), _beta_uw[~bc_uw])
                np.add.at(self.W, (_J_dw[~bc_dw], _J_uw[~bc_dw]), -_alpha_dw[~bc_dw])
                # Set right-hand side
                np.add.at(_chi_uwl, _J_uw, _chi_uw)
                np.add.at(_chi_dwm, _J_dw, _chi_dw)
                np.add.at(D, i[~bc], -_chi_uwl[~bc] + _chi_dwm[~bc])
            else:
                pass
        if n_p:
            bc_up = bc[_J_up]
            bc_dp = bc[_J_dp]
            if implicit:
                _alpha_up = _alpha_p
                _alpha_dp = _alpha_p
                _beta_up = _beta_p
                _beta_dp = _beta_p
                _chi_up = _chi_p
                _chi_dp = _chi_p
                _alpha_upm.fill(0)
                _beta_dpl.fill(0)
                _chi_upl.fill(0)
                _chi_dpm.fill(0)
                _P_diag.fill(0)
                # Set diagonal
                np.add.at(_alpha_upm, _J_up, _alpha_up)
                np.add.at(_beta_dpl, _J_dp, _beta_dp)
                _P_diag = -_beta_dpl + _alpha_upm
                # Set off-diagonal
                self.P[i[~bc], i[~bc]] = _P_diag[i[~bc]]
                self.P[_J_up[~bc_up], _J_dp[~bc_up]] = 0.0
                self.P[_J_dp[~bc_dp], _J_up[~bc_dp]] = 0.0
                np.add.at(self.P, (_J_up[~bc_up], _J_dp[~bc_up]), _beta_up[~bc_up])
                np.add.at(self.P, (_J_dp[~bc_dp], _J_up[~bc_dp]), -_alpha_dp[~bc_dp])
                # Set right-hand side
                np.add.at(_chi_upl, _J_up, _chi_up)
                np.add.at(_chi_dpm, _J_dp, _chi_dp)
                np.add.at(D, i[~bc], -_chi_upl[~bc] + _chi_dpm[~bc])
            else:
                pass
        b.fill(0)
        b = (_A_sj * H_j / _dt) + _Q_0j + D
        # Ensure boundary condition is specified
        b[bc] = H_bc[bc]
        # Export instance variables
        self.D = D
        self.b = b
        self._beta_dkl = _beta_dkl
        self._alpha_ukm = _alpha_ukm
        self._chi_ukl = _chi_ukl
        self._chi_dkm = _chi_dkm
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
                raise NotImplementedError
                # TODO: Broken
                # l = A
                # r = b + np.squeeze(B @ u)
        else:
            l = A
            r = b
        if _sparse:
            H_j_next = scipy.sparse.linalg.spsolve(l, r)
        else:
            H_j_next = scipy.linalg.solve(l, r)
        # Constrain heads based on allowed maximum/minimum depths
        H_j_next = np.maximum(H_j_next, _z_inv_j + min_depth)
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
        AB = np.zeros((2*bandwidth + 1, M))
        AB[bandwidth] = np.diag(l)
        for n in range(bandwidth):
            AB[n, (bandwidth - n):] = np.diag(l, k=bandwidth - n)
            AB[-n-1, :(-bandwidth + n)] = np.diag(l, k=-bandwidth + n)
        H_j_next = scipy.linalg.solve_banded((bandwidth, bandwidth), AB, r,
                                             check_finite=False, overwrite_ab=True)
        # Constrain heads based on allowed maximum/minimum depths
        H_j_next = np.maximum(H_j_next, _z_inv_j + min_depth)
        H_j_next = np.minimum(H_j_next, _z_inv_j + max_depth)
        # Export instance variables
        self.H_j = H_j_next

    def solve_superlink_flows(self):
        """
        Solve for superlink boundary discharges given superjunction
        heads at time t + dt.
        """
        # Import instance variables
        _J_uk = self._J_uk            # Index of superjunction upstream of superlink k
        _J_dk = self._J_dk            # Index of superjunction downstream of superlink k
        _alpha_uk = self._alpha_uk    # Superlink flow coefficient
        _alpha_dk = self._alpha_dk    # Superlink flow coefficient
        _beta_uk = self._beta_uk      # Superlink flow coefficient
        _beta_dk = self._beta_dk      # Superlink flow coefficient
        _chi_uk = self._chi_uk        # Superlink flow coefficient
        _chi_dk = self._chi_dk        # Superlink flow coefficient
        H_j = self.H_j                # Head at superjunction j
        # Compute flow at next time step
        _Q_uk_next = _alpha_uk * H_j[_J_uk] + _beta_uk * H_j[_J_dk] + _chi_uk
        _Q_dk_next = _alpha_dk * H_j[_J_uk] + _beta_dk * H_j[_J_dk] + _chi_dk
        # Export instance variables
        self._Q_uk = _Q_uk_next
        self._Q_dk = _Q_dk_next

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
        _orient_o = self._orient_o    # Orientation of orifice o (bottom/side)
        _y_max_o = self._y_max_o      # Maximum height of orifice o
        _Co = self._Co                # Discharge coefficient of orifice o
        _Ao = self._Ao                # Maximum flow area of orifice o
        _tau_o = self._tau_o
        _V_sj = self._V_sj
        # If no input signal, assume orifice is closed
        if u is None:
            u = np.zeros(self.n_o, dtype=float)
        g = 9.81
        # Create arrays to store flow coefficients for current time step
        _alpha_oo = np.zeros(self.n_o, dtype=float)
        _beta_oo = np.zeros(self.n_o, dtype=float)
        _chi_oo = np.zeros(self.n_o, dtype=float)
        # Specify orifice heads at previous timestep
        _H_uo = H_j[_J_uo]
        _H_do = H_j[_J_do]
        _z_inv_uo = _z_inv_j[_J_uo]
        # Create indicator functions
        upstream_ctrl = (_H_uo >= _H_do)
        _omega_o = (upstream_ctrl).astype(float)
        # Compute universal coefficients
        _gamma_oo = 2 * g * _Co**2 * _Ao**2
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
        _alpha_oo[a] = _gamma_oo[a]
        _beta_oo[a] = -_gamma_oo[a]
        _chi_oo[a] = 0.0
        # Submerged on one side
        b = (cond_0 & ~cond_1)
        _alpha_oo[b] = _gamma_oo[b] * _omega_o[b] * (-1)**(1 - _omega_o[b])
        _beta_oo[b] = _gamma_oo[b] * (1 - _omega_o[b]) * (-1)**(1 - _omega_o[b])
        _chi_oo[b] = (_gamma_oo[b] * (-1)**(1 - _omega_o[b])
                                      * (- _z_inv_uo[b] - _z_o[b]
                                         - _tau_o[b] * _y_max_o[b] * u[b] / 2))
        # Weir flow on one side
        c = (~cond_0 & cond_2)
        _alpha_oo[c] = _gamma_oo[c] * _omega_o[c] * (-1)**(1 - _omega_o[c])
        _beta_oo[c] = _gamma_oo[c] * (1 - _omega_o[c]) * (-1)**(1 - _omega_o[c])
        _chi_oo[c] = (_gamma_oo[c] * (-1)**(1 - _omega_o[c])
                                      * (- _z_inv_uo[c] - _z_o[c]))
        # No flow
        d = (~cond_0 & ~cond_2)
        _alpha_oo[d] = 0.0
        _beta_oo[d] = 0.0
        _chi_oo[d] = 0.0
        # Compute flow
        _Qo_next = (-1)**(1 - _omega_o) * np.sqrt(np.abs(
                _alpha_oo * _H_uo + _beta_oo * _H_do + _chi_oo))
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
        # Specify orifice heads at previous timestep
        _H_uw = H_j[_J_uw]
        _H_dw = H_j[_J_dw]
        _z_inv_uw = _z_inv_j[_J_uw]
        # Create indicator functions
        _omega_w = (_H_uw >= _H_dw).astype(float)
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
        # Compute universal coefficient
        _gamma_ww = (_Cwr * _L_w * _Hw + _Cwt * _s_w * _Hw**2)**2
        # Compute flow
        _Qw_next = (-1)**(1 - _omega_w) * np.sqrt(_gamma_ww * _Hw)
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
        # Specify pump heads at previous timestep
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
        self._Qp = _Qp_next

    def solve_superlink_depths(self):
        """
        Solve for depths at superlink ends given discharges and
        superjunction heads at time t + dt.
        """
        # Import instance variables
        _J_uk = self._J_uk              # Index of superjunction upstream of superlink k
        _J_dk = self._J_dk              # Index of superjunction downstream of superlink k
        _kappa_uk = self._kappa_uk      # Superlink head coefficient
        _kappa_dk = self._kappa_dk      # Superlink head coefficient
        _lambda_uk = self._lambda_uk    # Superlink head coefficient
        _lambda_dk = self._lambda_dk    # Superlink head coefficient
        _mu_uk = self._mu_uk            # Superlink head coefficient
        _mu_dk = self._mu_dk            # Superlink head coefficient
        _Q_uk = self._Q_uk              # Flow rate at upstream end of superlink k
        _Q_dk = self._Q_dk              # Flow rate at downstream end of superlink k
        H_j = self.H_j                  # Head at superjunction j
        min_depth = self.min_depth      # Minimum allowable depth at boundaries
        _theta_uk = self._theta_uk      # Upstream indicator variable
        _theta_dk = self._theta_dk      # Downstream indicator variable
        # Compute flow at next time step
        _h_uk_next = _kappa_uk * _Q_uk + _theta_uk * (_lambda_uk * H_j[_J_uk] + _mu_uk)
        _h_dk_next = _kappa_dk * _Q_dk + _theta_dk * (_lambda_dk * H_j[_J_dk] + _mu_dk)
        # Set minimum values
        # TODO: Is this causing the difference between normal/numba versions?
        # _h_uk_next[_h_uk_next < min_depth] = min_depth
        # _h_dk_next[_h_dk_next < min_depth] = min_depth
        # Export instance variables
        self._h_uk = _h_uk_next
        self._h_dk = _h_dk_next

    def solve_superlink_depths_alt(self):
        """
        Solve for depths at superlink ends given discharges and
        superjunction heads at time t + dt.
        """
        # Import instance variables
        _I_1k = self._I_1k              # Index of first junction in superlink k
        _I_Nk = self._I_Nk              # Index of penultimate junction in superlink k
        _U_Ik = self._U_Ik              # Forward recurrence coefficient
        _V_Ik = self._V_Ik              # Forward recurrence coefficient
        _W_Ik = self._W_Ik              # Forward recurrence coefficient
        _X_Ik = self._X_Ik              # Backward recurrence coefficient
        _Y_Ik = self._Y_Ik              # Backward recurrence coefficient
        _Z_Ik = self._Z_Ik              # Backward recurrence coefficient
        _Q_uk = self._Q_uk              # Flow rate at upstream end of superlink k
        _Q_dk = self._Q_dk              # Flow rate at downstream end of superlink k
        H_j = self.H_j                  # Head at superjunction j
        min_depth = self.min_depth      # Minimum allowable depth at superjunction j
        # Compute flow at next time step
        det = (_X_Ik[_I_1k] * _U_Ik[_I_Nk]) - (_Z_Ik[_I_1k] * _W_Ik[_I_Nk])
        det[det == 0] = np.inf
        _h_uk_next = (_U_Ik[_I_Nk] * (_Q_uk - _Y_Ik[_I_1k])
                      - _Z_Ik[_I_1k] * (_Q_dk - _V_Ik[_I_Nk])) / det
        _h_dk_next = (-_W_Ik[_I_Nk] * (_Q_uk - _Y_Ik[_I_1k])
                      + _X_Ik[_I_1k] * (_Q_dk - _V_Ik[_I_Nk])) / det
        # Set minimum values
        # _h_uk_next[_h_uk_next < min_depth] = min_depth
        # _h_dk_next[_h_dk_next < min_depth] = min_depth
        # Export instance variables
        self._h_uk = _h_uk_next
        self._h_dk = _h_dk_next

    def solve_internals_forwards(self, supercritical_only=False):
        """
        Solve for internal states of each superlink in the forward direction.
        """
        # Import instance variables
        _I_1k = self._I_1k                   # Index of first junction in superlink k
        _I_2k = self._I_2k                   # Index of second junction in superlink k
        _I_Nk = self._I_Nk                   # Index of penultimate junction in superlink k
        _I_Np1k = self._I_Np1k               # Index of last junction in superlink k
        _i_1k = self._i_1k                   # Index of first link in superlink k
        _i_nk = self._i_nk                   # Index of last link in superlink k
        _I_end = self._I_end                 # Junction is at downstream end of superlink (y/n)
        forward_I_I = self.forward_I_I       # Index of junction after junction Ik
        forward_I_i = self.forward_I_i       # Index of junction before junction Ik
        _h_Ik = self._h_Ik                   # Depth at junction Ik
        _Q_ik = self._Q_ik                   # Flow rate at link ik
        _D_Ik = self._D_Ik                   # Continuity coefficient
        _E_Ik = self._E_Ik                   # Continuity coefficient
        _U_Ik = self._U_Ik                   # Forward recurrence coefficient
        _V_Ik = self._V_Ik                   # Forward recurrence coefficient
        _W_Ik = self._W_Ik                   # Forward recurrence coefficient
        _X_Ik = self._X_Ik                   # Backward recurrence coefficient
        _Y_Ik = self._Y_Ik                   # Backward recurrence coefficient
        _Z_Ik = self._Z_Ik                   # Backward recurrence coefficient
        _Q_uk = self._Q_uk                   # Flow rate at upstream end of superlink k
        _Q_dk = self._Q_dk                   # Flow rate at downstream end of superlink k
        _h_uk = self._h_uk                   # Depth at upstream end of superlink k
        _h_dk = self._h_dk                   # Depth at downstream end of superlink k
        min_depth = self.min_depth           # Minimum allowed water depth
        _end_method = self._end_method       # Method for computing flow at pipe ends
        # max_depth = 0.3048 * 10
        if supercritical_only:
            _supercritical = self._supercritical
        # Set first elements
        if _end_method == 'o':
            _Q_ik[_i_1k] = _Q_uk
            _Q_ik[_i_nk] = _Q_dk
            _h_Ik[_I_1k] = _h_uk
            _h_Ik[_I_Np1k] = _h_dk
        else:
            _Q_ik[_i_1k] = _Q_uk - _E_Ik[_I_1k] * _h_uk + _D_Ik[_I_1k]
            _Q_ik[_i_nk] = _Q_dk + _E_Ik[_I_Np1k] * _h_dk - _D_Ik[_I_Np1k]
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
        while _I_next.size:
            _i_next = forward_I_i[_I_next]
            _im1_next = forward_I_i[_Im1_next]
            _Ip1_next = forward_I_I[_I_next]
            _h_Ik[_I_next] = self._h_Ik_next_f(_Q_ik[_im1_next], _V_Ik[_Im1_next],
                                                 _W_Ik[_Im1_next], _h_Ik[_I_1k_next],
                                                 _U_Ik[_Im1_next])
            _h_Ik[_I_next[_h_Ik[_I_next] < min_depth]] = min_depth
            # _h_Ik[_I_next[_h_Ik[_I_next] > max_depth]] = min_depth
            _Q_ik[_i_next] = self._Q_i_next_b(_X_Ik[_I_next], _h_Ik[_I_next],
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
        if _end_method == 'o':
            _Q_ik[_i_1k] = _Q_uk
            _Q_ik[_i_nk] = _Q_dk
            _h_Ik[_I_1k] = _h_uk
            _h_Ik[_I_Np1k] = _h_dk
        else:
            _Q_ik[_i_1k] = _Q_uk - _E_Ik[_I_1k] * _h_uk + _D_Ik[_I_1k]
            _Q_ik[_i_nk] = _Q_dk + _E_Ik[_I_Np1k] * _h_dk - _D_Ik[_I_Np1k]
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

    def _Q_i_next_b(self, X_Ik, h_Ik, Y_Ik, Z_Ik, h_Np1k):
        t_0 = X_Ik * h_Ik
        t_1 = Y_Ik
        t_2 = Z_Ik * h_Np1k
        return t_0 + t_1 + t_2

    def solve_internals_backwards(self, subcritical_only=False):
        """
        Solve for internal states of each superlink in the backward direction.
        """
        # Import instance variables
        _I_1k = self._I_1k                  # Index of first junction in superlink k
        _I_2k = self._I_2k                  # Index of second junction in superlink k
        _I_Nk = self._I_Nk                  # Index of penultimate junction in superlink k
        _I_Np1k = self._I_Np1k              # Index of last junction in superlink k
        _i_1k = self._i_1k                  # Index of first link in superlink k
        _i_nk = self._i_nk                  # Index of last link in superlink k
        _I_end = self._I_end                # Junction is at downstream end (y/n)
        backward_I_I = self.backward_I_I    # Index of next junction after junction Ik
        forward_I_i = self.forward_I_i      # Index of next link after junction Ik
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
        _end_method = self._end_method    # Method for computing flow at pipe ends
        # max_depth = 0.3048 * 10
        if subcritical_only:
            _subcritical = ~self._supercritical
        # Set first elements
        if _end_method == 'o':
            _Q_ik[_i_1k] = _Q_uk
            _Q_ik[_i_nk] = _Q_dk
            _h_Ik[_I_1k] = _h_uk
            _h_Ik[_I_Np1k] = _h_dk
        else:
            _Q_ik[_i_1k] = _X_Ik[_I_1k] * _h_uk + _Z_Ik[_I_1k] * _h_dk + _Y_Ik[_I_1k]
            _Q_ik[_i_nk] = _U_Ik[_I_Nk] * _h_dk + _W_Ik[_I_Nk] * _h_uk + _V_Ik[_I_Nk]
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
        while _I_next.size:
            _i_next = forward_I_i[_I_next]
            _Im1_next = backward_I_I[_I_next]
            _im1_next = forward_I_i[_Im1_next]
            _h_Ik[_I_next] = self._h_Ik_next_b(_Q_ik[_i_next], _Y_Ik[_I_next],
                                                 _Z_Ik[_I_next], _h_Ik[_I_Np1k_next],
                                                 _X_Ik[_I_next])
            # Ensure non-negative depths?
            _h_Ik[_I_next[_h_Ik[_I_next] < min_depth]] = min_depth
            # _h_Ik[_I_next[_h_Ik[_I_next] > max_depth]] = min_depth
            _Q_ik[_im1_next] = self._Q_im1k_next_f(_U_Ik[_Im1_next], _h_Ik[_I_next],
                                                     _V_Ik[_Im1_next], _W_Ik[_Im1_next],
                                                     _h_Ik[_I_1k_next])
            keep = (_Im1_next != _I_1k_next)
            _I_next = _Im1_next[keep]
            _I_1k_next = _I_1k_next[keep]
            _I_Np1k_next = _I_Np1k_next[keep]
        # Set upstream flow
        # TODO: May want to delete where this is set earlier
        if _end_method == 'o':
            _Q_ik[_i_1k] = _Q_uk
            _Q_ik[_i_nk] = _Q_dk
            _h_Ik[_I_1k] = _h_uk
            _h_Ik[_I_Np1k] = _h_dk
        else:
            _Q_ik[_i_1k] = _X_Ik[_I_1k] * _h_uk + _Z_Ik[_I_1k] * _h_dk + _Y_Ik[_I_1k]
            _Q_ik[_i_nk] = _U_Ik[_I_Nk] * _h_dk + _W_Ik[_I_Nk] * _h_uk + _V_Ik[_I_Nk]
            _h_Ik[_I_1k] = _h_uk
            _h_Ik[_I_Np1k] = _h_dk
        # Ensure non-negative depths?
        _h_Ik[_h_Ik < min_depth] = min_depth
        # _h_Ik[_h_Ik > max_depth] = max_depth
        # Export instance variables
        self._h_Ik = _h_Ik
        self._Q_ik = _Q_ik

    @safe_divide
    def _h_Ik_next_b(self, Q_ik, Y_Ik, Z_Ik, h_Np1k, X_Ik):
        num = Q_ik - Y_Ik - Z_Ik * h_Np1k
        den = X_Ik
        return num, den

    def _Q_im1k_next_f(self, U_Im1k, h_Ik, V_Im1k, W_Im1k, h_1k):
        t_0 = U_Im1k * h_Ik
        t_1 = V_Im1k
        t_2 = W_Im1k * h_1k
        return t_0 + t_1 + t_2

    def solve_internals_nnls(self):
        NK = self.NK
        nk = self.nk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _kI = self._kI
        _ki = self._ki
        _I_1k = self._I_1k
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
        for k in range(NK):
            # Set up solution matrix
            nlinks = nk[k]
            njunctions = nlinks + 1
            I_k = (_kI == k)
            i_k = (_ki == k)
            k_k = (_kk == k)
            Ak = np.zeros((nlinks, nlinks - 1))
            # Fill right-hand side matrix
            Ak.flat[::nlinks] = _U[k_k]
            Ak.flat[nlinks-1::nlinks] = -_X[k_k]
            bk = _b[i_k]
            _h_inner, _res = scipy.optimize.nnls(Ak, bk)
            _h_Ik[I_k & ~_is_start & ~_is_end] = _h_inner
        # Set depths at upstream and downstream ends
        _h_Ik[_is_start] = _h_uk
        _h_Ik[_is_end] = _h_dk
        # Solve for flows using new depths
        Q_ik_b, Q_ik_f = self.superlink_flow_from_recurrence()
        _Q_ik = (Q_ik_b + Q_ik_f) / 2
        # Export instance variables
        self._Q_ik = _Q_ik
        self._h_Ik = _h_Ik

    def solve_internals_lsq(self):
        NK = self.NK
        nk = self.nk
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        _h_Ik = self._h_Ik
        _Q_ik = self._Q_ik
        _kI = self._kI
        _ki = self._ki
        _I_1k = self._I_1k
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
        for k in range(NK):
            # Set up solution matrix
            nlinks = nk[k]
            njunctions = nlinks + 1
            I_k = (_kI == k)
            i_k = (_ki == k)
            k_k = (_kk == k)
            Ak = np.zeros((nlinks, nlinks - 1))
            # Fill right-hand side matrix
            Ak.flat[::nlinks] = _U[k_k]
            Ak.flat[nlinks-1::nlinks] = -_X[k_k]
            bk = _b[i_k]
            AA = Ak.T @ Ak
            Ab = Ak.T @ bk
            # Prevent singular matrix
            AA[AA.flat[::nlinks] == 0.0] = 1.0
            _h_inner = np.linalg.solve(AA, Ab)
            _h_Ik[I_k & ~_is_start & ~_is_end] = _h_inner
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

    def exit_conditions(self):
        """
        Determine which superlinks have exit depths below the pipe crown elevation.
        """
        # Import instance variables
        _g1_ik = self._g1_ik          # Geometry 1 of link ik (vertical)
        _i_nk = self._i_nk            # Index of last link in superlink k
        _z_inv_dk = self._z_inv_dk    # Invert offset of downstream end of superlink k
        _J_dk = self._J_dk            # Index of junction downstream of superlink k
        H_j = self.H_j                # Head at superjunction j
        # Determine which superlinks need critical depth
        _crown_elev_dk = _g1_ik[_i_nk]
        _exit_conditions = (H_j[_J_dk] - _z_inv_dk) < _crown_elev_dk
        self._exit_conditions = _exit_conditions

    def solve_critical_depth(self):
        """
        Solve for critical depth in each superlink that requires computation of exit
        hydraulics.
        """
        # Import instance variables
        _g1_ik = self._g1_ik          # Geometry 1 of link ik (vertical)
        _i_nk = self._i_nk            # Index of last link in superlink k
        _z_inv_dk = self._z_inv_dk    # Invert offset of downstream end of superlink k
        _h_dk = self._h_dk            # Depth at downstream end of superlink k
        _Q_dk = self._Q_dk            # Flow rate at downstream end of superlink k
        _h_c = self._h_c              # Critical depth
        # Determine which superlinks need critical depth computation
        _exit_conditions = self._exit_conditions
        _h_c[:] = 0.
        _crown_elev_dk = _g1_ik[_i_nk]
        _eps = np.finfo(np.float64).eps
        _iter = np.flatnonzero(_exit_conditions)
        for i in _iter:
            _h_c[i] = scipy.optimize.brentq(self._critical_depth,
                                            np.array([_eps]),
                                            _crown_elev_dk[[i]],
                                            args=(_Q_dk[[i]],
                                                  _crown_elev_dk[[i]]))
        # Export instance variables
        self._h_c = _h_c

    def solve_normal_depth(self):
        """
        Solve for normal depth in each superlink that requires computation of exit
        hydraulics.
        """
        # Import instance variables
        _g1_ik = self._g1_ik          # Geometry 1 of link ik (vertical)
        _i_nk = self._i_nk            # Index of last link in superlink k
        _z_inv_dk = self._z_inv_dk    # Invert offset of downstream end of superlink k
        _h_dk = self._h_dk            # Depth at downstream end of superlink k
        _Q_dk = self._Q_dk            # Flow rate at downstream end of superlink k
        _h_n = self._h_n              # Normal depth
        # Determine which superlinks need critical depth
        _exit_conditions = self._exit_conditions
        _h_n[:] = 0.
        _crown_elev_dk = _g1_ik[_i_nk]
        _n = self._n_ik[_i_nk[_exit_conditions]]
        _S_o = self._S_o_ik[_i_nk[_exit_conditions]]
        _h_n[_exit_conditions] = scipy.optimize.newton(self._normal_depth,
                                                       _h_dk[_exit_conditions],
                                                       args=(_Q_dk[_exit_conditions],
                                                             _crown_elev_dk[_exit_conditions],
                                                             _n, _S_o))
        # Export instance variables
        self._h_n = _h_n

    def solve_exit_hydraulics(self):
        """
        Solve for the exit depth at flow rate in each superlink.
        """
        # Import instance variables
        _h_c = self._h_c                 # Critical depth
        _h_n = self._h_n                 # Normal depth
        _h_dk = self._h_dk               # Depth at downstream end of superlink k
        _z_inv_dk = self._z_inv_dk       # Invert offset of downstream end of superlink k
        _J_dk = self._J_dk               # Index of superjunction downstream of superlink k
        H_j = self.H_j                   # Head at superjunction j
        min_depth = self.min_depth       # Minimum allowed water depth
        _exit_conditions = self._exit_conditions    # Require exit hydraulics computation (y/n)
        # Determine conditions
        _h_j = H_j[_J_dk] - _z_inv_dk
        _backwater = (_h_j > _h_c) & (_exit_conditions)
        _mild = (~_backwater & (_h_n > _h_c)) & (_exit_conditions)
        _critical = (~_backwater & (_h_n == _h_c)) & (_exit_conditions)
        _steep = (~_backwater & (_h_n < _h_c)) & (_exit_conditions)
        _h_dk[_backwater] = _h_j[_backwater]
        _h_dk[_mild] = _h_c[_mild]
        _h_dk[_critical] = _h_c[_critical]
        # Assume that _h_n is _h_dk
        # _h_dk[_steep] = _h_n[_steep]
        _h_dk[_steep] = _h_dk[_steep]
        _h_dk[_h_dk < min_depth] = min_depth
        # Export instance variables
        self._h_dk = _h_dk

    def _critical_depth(self, h, Q, d):
        # TODO: This will always be circular
        _h = np.array([h])
        return np.asscalar(np.log((Q**2) * pipedream_solver.geometry.Circular.B_ik(_h, _h, d)
                      / (9.81 * pipedream_solver.geometry.Circular.A_ik(_h, _h, d)**3)))

    def _normal_depth(self, h, Q, d, n, S_o):
        # TODO: This will always be circular
        return Q - (S_o**(1/2) * pipedream_solver.geometry.Circular.A_ik(h, h, d)**(5/3)
                    / pipedream_solver.geometry.Circular.Pe_ik(h, h, d)**(2/3) / n)

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
        _fixed = self._fixed          # Junction Ik is fixed (y/n)
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
        _k = self._k                  # List of superlinks
        # Configure function variables
        njunctions_fixed = 4
        njunctions = njunctions_fixed + 1
        nlinks = njunctions - 1
        xx = _x_Ik.reshape(-1, njunctions)
        zz = _z_inv_Ik.reshape(-1, njunctions)
        dxdx = _dx_ik.reshape(-1, nlinks)
        hh = _h_Ik.reshape(-1, njunctions)
        QQ = _Q_ik.reshape(-1, nlinks)
        _H_dk = H_j[_J_dk]
        Q_ix = np.tile(np.arange(nlinks), len(QQ)).reshape(-1, nlinks)
        # TODO: Add case where position of movable coord is exactly equal to fixed coord
        move_junction = (_H_dk > _z_inv_Ik[_I_Np1k]) & (_H_dk < _z_inv_Ik[_I_1k])
        z_m = np.where(move_junction, _H_dk, _z0)
        x_m = np.where(move_junction, (_H_dk - _b0) / _m, _x0)
        # TODO: Use instance variable
        r = np.arange(len(xx))
        c = np.array(list(map(np.searchsorted, xx, x_m)))
        frac = (x_m - xx[r, c - 1]) / (xx[r, c] - xx[r, c - 1])
        h_m = (1 - frac) * hh[r, c - 1] + (frac) * hh[r, c]
        pos_prev = np.where(~_fixed)[1]
        xx[:, :-1] = xx[_fixed].reshape(-1, njunctions - 1)
        zz[:, :-1] = zz[_fixed].reshape(-1, njunctions - 1)
        hh[:, :-1] = hh[_fixed].reshape(-1, njunctions - 1)
        xx[:, -1] = x_m
        zz[:, -1] = z_m
        hh[:, -1] = h_m
        _fixed[:, :-1] = True
        _fixed[:, -1] = False
        # TODO: Check this
        ix = np.argsort(xx)
        xx = np.take_along_axis(xx, ix, axis=-1)
        zz = np.take_along_axis(zz, ix, axis=-1)
        hh = np.take_along_axis(hh, ix, axis=-1)
        _fixed = np.take_along_axis(_fixed, ix, axis=-1)
        dxdx = np.diff(xx)
        pos_next = np.where(~_fixed)[1]
        shifted = (pos_prev != pos_next)
        Q_ix[r[shifted], pos_prev[shifted]] = pos_next[shifted]
        # TODO: This should be weighted average instead of simple average
        QQ[r[shifted], pos_prev[shifted] - 1] = (QQ[r[shifted], pos_prev[shifted]]
                                                + QQ[r[shifted], pos_prev[shifted] - 1]) / 2
        Q_ix.sort(axis=-1)
        QQ = np.take_along_axis(QQ, Q_ix, axis=-1)
        QQ = np.take_along_axis(QQ, Q_ix, axis=-1)
        # Export instance variables
        if reposition is None:
            self._h_Ik = hh.ravel()
            self._Q_ik = QQ.ravel()
            self._x_Ik = xx.ravel()
            self._z_inv_Ik = zz.ravel()
            self._dx_ik = dxdx.ravel()
            self._fixed = _fixed
        else:
            Ik = np.flatnonzero(np.repeat(reposition, njunctions))
            ik = np.flatnonzero(np.repeat(reposition, nlinks))
            self._h_Ik[Ik] = hh[reposition].ravel()
            self._Q_ik[ik] = QQ[reposition].ravel()
            self._x_Ik[Ik] = xx[reposition].ravel()
            self._z_inv_Ik[Ik] = zz[reposition].ravel()
            self._dx_ik[ik] = dxdx[reposition].ravel()
            self._fixed = _fixed

    def superlink_flow_from_recurrence(self):
        # Import instance variables
        nk = self.nk
        _is_end = self._is_end
        _is_start = self._is_start
        _h_Ik = self._h_Ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        # Compute internal flow estimates in both directions
        Q_ik_b = self._Q_i_next_b(_X_Ik[~_is_end], _h_Ik[~_is_end],
                                  _Y_Ik[~_is_end], _Z_Ik[~_is_end],
                                  np.repeat(_h_dk, nk))
        Q_ik_f = self._Q_im1k_next_f(_U_Ik[~_is_end], _h_Ik[~_is_start],
                                     _V_Ik[~_is_end], _W_Ik[~_is_end],
                                     np.repeat(_h_uk, nk))
        return Q_ik_b, Q_ik_f

    def superlink_depth_from_recurrence(self):
        # Import instance variables
        nk = self.nk
        _is_end = self._is_end
        _is_start = self._is_start
        _Q_ik = self._Q_ik
        _U_Ik = self._U_Ik
        _V_Ik = self._V_Ik
        _W_Ik = self._W_Ik
        _X_Ik = self._X_Ik
        _Y_Ik = self._Y_Ik
        _Z_Ik = self._Z_Ik
        _h_uk = self._h_uk
        _h_dk = self._h_dk
        min_depth = self.min_depth
        # Create arrays
        n_junctions = (nk + 1).sum()
        h_Ik_b = np.empty(n_junctions)
        h_Ik_f = np.empty(n_junctions)
        # Compute internal flow estimates in both directions
        h_Ik_b[~_is_end] = self._h_Ik_next_b(_Q_ik, _Y_Ik[~_is_end], _Z_Ik[~_is_end],
                                   np.repeat(_h_dk, nk), _X_Ik[~_is_end])
        h_Ik_f[~_is_start] = self._h_Ik_next_f(_Q_ik, _V_Ik[~_is_end], _W_Ik[~_is_end],
                                            np.repeat(_h_uk, nk), _U_Ik[~_is_end])
        # Ensure end depths are consistent with boundary conditions
        h_Ik_b[_is_start] = _h_uk
        h_Ik_b[_is_end] = _h_dk
        h_Ik_f[_is_start] = _h_uk
        h_Ik_f[_is_end] = _h_dk
        h_Ik_b = np.maximum(h_Ik_b, min_depth)
        h_Ik_f = np.maximum(h_Ik_f, min_depth)
        return h_Ik_b, h_Ik_f

    def superlink_inverse_courant(self):
        # Import instance variables
        _Q_k = self._Q_k
        _A_k = self._A_k
        _Q_ik = self._Q_ik
        _A_ik = self._A_ik
        _dx_k = self._dx_k
        _dt_ck = self._dt_ck
        nk = self.nk
        # Reset superlink variables
        _Q_k.fill(0.)
        _A_k.fill(0.)
        np.add.at(self._Q_k, self._ki, self._Q_ik)
        _Q_k /= nk
        np.add.at(self._A_k, self._ki, self._A_ik)
        _A_k /= nk
        _dt_ck = np.abs(_A_k * _dx_k / _Q_k)
        # Export instance variables
        self._Q_k = _Q_k
        self._A_k = _A_k
        self._dt_ck = _dt_ck

    def _augmented_system(self, _Q_0j=None, _dt=None):
        A = self.A                    # Superlink/superjunction matrix
        O = self.O                    # Orifice matrix
        W = self.W                    # Weir matrix
        P = self.P                    # Pump matrix
        D = self.D                    # Vector for storing chi coefficients
        M = self.M                    # Number of superjunctions in system
        H_j = self.H_j                # Head at superjunction j
        _A_sj = self._A_sj            # Surface area of superjunction j
        bc = self.bc                  # Superjunction j has a fixed boundary condition (y/n)
        n_o = self.n_o                # Number of orifices
        n_w = self.n_w                # Number of weirs
        n_p = self.n_p                # Number of pumps
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        # If no flow input specified, assume zero external inflow
        if _Q_0j is None:
            _Q_0j = 0
        A_1 = np.zeros((M + 1, M + 1))
        A_2 = np.zeros((M + 1, M + 1))
        Q = np.zeros(M + 1)
        has_control = n_o + n_w + n_p
        # Get right-hand size
        if has_control:
            L = A + O + W + P
        else:
            L = A
        # Fill in A_1 matrix
        A_1[:-1, :-1] = L
        A_1[-1, -1] = 1
        # Fill in A_2 matrix
        ix = np.arange(0, M)
        A_2[ix, ix] = np.where(~bc, _A_sj / _dt, 0)
        A_2[:-1, -1] = D
        A_2[-1, -1] = 1
        # Fill in Q vector
        Q[:-1] = _Q_0j
        Q[-1] = 0.0
        return A_1, A_2, Q, H_j

    def _semi_implicit_system(self, _dt=None):
        A = self.A                    # Superlink/superjunction matrix
        O = self.O                    # Orifice matrix
        W = self.W                    # Weir matrix
        P = self.P                    # Pump matrix
        b = self.b
        M = self.M                    # Number of superjunctions in system
        H_j = self.H_j                # Head at superjunction j
        _A_sj = self._A_sj            # Surface area of superjunction j
        bc = self.bc                  # Superjunction j has a fixed boundary condition (y/n)
        n_o = self.n_o                # Number of orifices
        n_w = self.n_w                # Number of weirs
        n_p = self.n_p                # Number of pumps
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        has_control = n_o + n_w + n_p
        # Get A_1
        if has_control:
            A_1 = A + O + W + P
        else:
            A_1 = A
        # Get A_2
        A_2 = np.diag(np.where(~bc, _A_sj / _dt, 0))
        return A_1, A_2, b

    def state_space_system(self, _dt=None):
        A = self.A                    # Superlink/superjunction matrix
        O = self.O                    # Orifice matrix
        W = self.W                    # Weir matrix
        P = self.P                    # Pump matrix
        D = self.D
        M = self.M                    # Number of superjunctions in system
        H_j_next = self.H_j                # Head at superjunction j
        H_j_prev = self.states['H_j']
        _A_sj = self._A_sj            # Surface area of superjunction j
        bc = self.bc                  # Superjunction j has a fixed boundary condition (y/n)
        n_o = self.n_o                # Number of orifices
        n_w = self.n_w                # Number of weirs
        n_p = self.n_p                # Number of pumps
        Q_in = self._Q_in
        # If no time step specified, use instance time step
        if _dt is None:
            _dt = self._dt
        has_control = n_o + n_w + n_p
        # Get A_1
        if has_control:
            A_1 = A + O + W + P
        else:
            A_1 = A
        # Get A_2
        A_2 = np.diag(np.where(~bc, _A_sj / _dt, 0))
        return A_1, A_2, D, H_j_next, H_j_prev, Q_in

    def save_state(self):
        """
        Save current model state to dict stored in self.states.
        """
        self.states['t'] = copy.copy(self.t)
        self.states['H_j'] = np.copy(self.H_j)
        self.states['h_Ik'] = np.copy(self.h_Ik)
        self.states['Q_ik'] = np.copy(self.Q_ik)
        self.states['Q_uk'] = np.copy(self.Q_uk)
        self.states['Q_dk'] = np.copy(self.Q_dk)
        self.states['x_Ik'] = np.copy(self.x_Ik)
        if self.n_o:
            self.states['Q_o'] = np.copy(self.Q_o)
        if self.n_w:
            self.states['Q_w'] = np.copy(self.Q_w)
        if self.n_p:
            self.states['Q_p'] = np.copy(self.Q_p)

    def load_state(self, states={}):
        """
        Load model state.

        Inputs:
        -------
        states : dict
            Dict of model states. If empty, load current state stored in self.states dict.
        """
        # If no states given, load previous states
        if not states:
            states = self.states
        for key, value in states.items():
            setattr(self, key, value)
        # Ensure consistency of internal states
        self.link_hydraulic_geometry()
        self.upstream_hydraulic_geometry()
        self.downstream_hydraulic_geometry()
        self.compute_storage_areas()
        self.node_velocities()

    def spinup(self, n_steps=100, dt=10, Q_in=None, Q_0Ik=None, reposition_junctions=True,
               reset_counters=True, **kwargs):
        """
        Spin up solver for a given number of steps to avoid running a completely dry model.
        """
        if Q_in is None:
            Q_in = 1e-6 * np.ones(self.M)
        if Q_0Ik is None:
            Q_0Ik = 1e-6 * np.ones(self._I.size)
        for _ in range(n_steps):
            self.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, **kwargs)
            if reposition_junctions:
                self.reposition_junctions()
        if reset_counters:
            self.t = 0.
            self.iter_count = 0

    def plot_profile(self, js, ax=None, width=1, superlink_kwargs={},
                     superjunction_kwargs={}):
        return (pipedream_solver
                .visualization
                .plot_profile(self, js=js, ax=ax, width=width,
                              superlink_kwargs=superlink_kwargs,
                              superjunction_kwargs=superjunction_kwargs))

    def plot_network_2d(self, ax=None, superjunction_kwargs={}, junction_kwargs={},
                    link_kwargs={}, orifice_kwargs={}, weir_kwargs={}, pump_kwargs={}):
        return (pipedream_solver
                .visualization
                .plot_network_2d(self, ax=ax,
                                 superjunction_kwargs=superjunction_kwargs,
                                 junction_kwargs=junction_kwargs,
                                 link_kwargs=link_kwargs,
                                 orifice_kwargs=orifice_kwargs,
                                 weir_kwargs=weir_kwargs,
                                 pump_kwargs=pump_kwargs))

    def plot_network_3d(self, ax=None, superjunction_signal=None, junction_signal=None,
                        superjunction_stems=True, junction_stems=True,
                        border=True, fill=True, base_line_kwargs={}, superjunction_stem_kwargs={},
                        junction_stem_kwargs={}, border_kwargs={}, fill_kwargs={},
                        orifice_kwargs={}, weir_kwargs={}, pump_kwargs={}):
        return (pipedream_solver
                .visualization
                .plot_network_3d(self, ax=ax,
                                 superjunction_signal=superjunction_signal,
                                 junction_signal=junction_signal,
                                 superjunction_stems=superjunction_stems,
                                 junction_stems=junction_stems,
                                 border=border,
                                 fill=fill,
                                 base_line_kwargs=base_line_kwargs,
                                 superjunction_stem_kwargs=superjunction_stem_kwargs,
                                 junction_stem_kwargs=junction_stem_kwargs,
                                 border_kwargs=border_kwargs,
                                 fill_kwargs=fill_kwargs,
                                 orifice_kwargs=orifice_kwargs,
                                 weir_kwargs=weir_kwargs,
                                 pump_kwargs=pump_kwargs))

    def _setup_step(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
             first_time=False, implicit=True, banded=False, first_iter=True):
        if first_iter:
            self.save_state()
        if dt is None:
            dt = self._dt
        self._Q_in = Q_in
        self._Q_0Ik = Q_0Ik
        if not implicit:
            raise NotImplementedError
        self.link_hydraulic_geometry()
        self.upstream_hydraulic_geometry()
        self.downstream_hydraulic_geometry()
        self.compute_storage_areas()
        self.compute_storage_volumes()
        self.node_velocities()
        if self.inertial_damping:
            self.compute_flow_regime()
        self.link_coeffs(_dt=dt, first_iter=first_iter)
        self.node_coeffs(_Q_0Ik=Q_0Ik, _dt=dt, first_iter=first_iter)
        self.forward_recurrence()
        self.backward_recurrence()
        self.superlink_upstream_head_coefficients()
        self.superlink_downstream_head_coefficients()
        self.superlink_flow_coefficients()
        if self.orifices is not None:
            self.orifice_hydraulic_geometry(u=u_o)
            self.orifice_flow_coefficients(u=u_o)
        if self.weirs is not None:
            self.weir_flow_coefficients(u=u_w)
        if self.pumps is not None:
            self.pump_flow_coefficients(u=u_p)
        self.sparse_matrix_equations(H_bc=H_bc, _Q_0j=Q_in,
                                     first_time=first_time, _dt=dt,
                                     implicit=implicit)

    def _solve_step(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
             first_time=False, implicit=True, banded=False, first_iter=True):
        _method = self._method
        _exit_hydraulics = self._exit_hydraulics
        if banded:
            self.solve_banded_matrix(implicit=implicit)
        else:
            self.solve_sparse_matrix(implicit=implicit)
        self.solve_superlink_flows()
        if self.orifices is not None:
            self.solve_orifice_flows(dt=dt, u=u_o)
        if self.weirs is not None:
            self.solve_weir_flows(u=u_w)
        if self.pumps is not None:
            self.solve_pump_flows(u=u_p)
        self.solve_superlink_depths()
        if _exit_hydraulics:
            self.exit_conditions()
            self.solve_critical_depth()
            # self.solve_normal_depth()
            self.solve_exit_hydraulics()
        if _method == 'b':
            self.solve_internals_backwards()
        elif _method == 'f':
            self.solve_internals_forwards()
        elif _method == 'nnls':
            self.solve_internals_nnls()
        elif _method == 'lsq':
            self.solve_internals_lsq()
        self.iter_count += 1
        self.t += dt

    def step(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
             first_time=False, implicit=True, banded=False, first_iter=True,
             num_iter=1, head_tol=0.0015):
        """
        Advance model forward to next time step, computing hydraulic states.

        Inputs:
        -------
        H_bc : np.ndarray (M)
            Boundary stage at each superjunction (m)
        Q_in : np.ndarray (M)
            Direct inflow at each superjunction (m^3/s)
        Q_0Ik : np.ndarray (MK)
            Direct inflow at each junction (m^3/s)
        u_o : np.ndarray (o)
            Orifice control signal. Represents fraction of orifice open (0-1).
        u_w : np.ndarray (w)
            Weir control signal. Represents fraction of weir open (0-1).
        u_p : np.ndarray (p)
            Pump control signal. Represents fraction of maximum pump flow (0-1).
        dt : float
            Time step to advance (s)
        first_time : bool
            Set True if this is the first step the model has performed.
        banded : bool
            If True, use banded matrix solver.
        first_iter : bool
            True if this is the first iteration when iterating towards convergence.
        num_iter : int
            Number of iterations to perform when iterating towards convergence.
        head_tol : float
            Maximum allowable head tolerance when iterating towards convergence (m).
        implicit : bool
            (Deprecated)
        """
        self._setup_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                         first_time=first_time, implicit=implicit, banded=banded,
                         first_iter=first_iter)
        self._solve_step(H_bc=H_bc, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                         first_time=first_time, implicit=implicit, banded=banded,
                         first_iter=first_iter)
        # Perform fixed-point iteration until convergence
        num_iter -= 1
        if (num_iter > 0):
            H_j_prev = self.states['H_j']
            H_j_next = np.copy(self.H_j)
            h_Ik_prev = self.states['h_Ik']
            Q_ik_prev = self.states['Q_ik']
            Q_uk_prev = self.states['Q_uk']
            Q_dk_prev = self.states['Q_dk']
            if self.n_o:
                Q_o_prev = self.states['Q_o']
            if self.n_w:
                Q_w_prev = self.states['Q_w']
            if self.n_p:
                Q_p_prev = self.states['Q_p']
            residual = np.abs(H_j_next - H_j_prev)
            if not (residual < head_tol).all():
                for _ in range(num_iter):
                    self.iter_count -= 1
                    self.t -= dt
                    self.H_j = H_j_prev
                    self._h_Ik = (h_Ik_prev + self._h_Ik) / 2
                    self._Q_ik = (Q_ik_prev + self._Q_ik) / 2
                    self._Q_uk = (Q_uk_prev + self._Q_uk) / 2
                    self._Q_dk = (Q_dk_prev + self._Q_dk) / 2
                    if self.n_o:
                        self._Qo = (Q_o_prev + self._Qo) / 2
                    if self.n_w:
                        self._Qw = (Q_w_prev + self._Qw) / 2
                    if self.n_p:
                        self._Qp = (Q_p_prev + self._Qp) / 2
                    self._setup_step(H_bc=H_bc, Q_in=Q_in, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                     first_time=first_time, implicit=implicit, banded=banded,
                                     first_iter=False)
                    self._solve_step(H_bc=H_bc, Q_in=Q_in, u_o=u_o, u_w=u_w, u_p=u_p, dt=dt,
                                     first_time=first_time, implicit=implicit, banded=banded,
                                     first_iter=False)
                    residual = np.abs(H_j_next - self.H_j)
                    if (residual < head_tol).all():
                        break
                    H_j_next = np.copy(self.H_j)
