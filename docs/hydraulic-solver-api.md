# Hydraulic solver API documentation

This section enumerates all the methods of the `SuperLink` class.

## Instantiating the `SuperLink` model

A hydraulic model is instantiated using the `pipedream_solver.hydraulics.SuperLink` class.

### Initialization parameters

The `SuperLink` class is initialized with the following parameters:

|------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Attribute        | Type         | Description                                                                                                                                                                                          |
|------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `superlinks`     | pd.DataFrame | Table containing all superlinks in the network along with their attributes. See [model inputs: superlinks](/pipedream/model-inputs.html#superlinks) for specification.                               |
| `superjunctions` | pd.DataFrame | Table containing all superjunctions in the network along with their attributes. See [model inputs: superjunctions](/pipedream/model-inputs.html#superjunctions) for specification.                   |
| `links`          | pd.DataFrame | Table containing all links in the network along with their attributes. See [model inputs: links](/pipedream/model-inputs.html#links) for specification.                                              |
| `junctions`      | pd.DataFrame | Table containing all junctions in the network along with their attributes. See [model inputs: junctions](/pipedream/model-inputs.html#junctions) for specification.                                  |
| `transects`      | dict         | Dictionary describing nonfunctional channel cross-sectional geometries. See [model inputs: transects](/pipedream/model-inputs.html#transects) for specification.                                     |
| `storages`       | dict         | Dictionary describing tabular storages for superjunctions. See [model inputs: storages](/pipedream/model-inputs.html#storages) for specification.                                                    |
| `orifices`       | pd.DataFrame | Table containing orifice control structures, and their attributes. See [model inputs: orifices](/pipedream/model-inputs.html#orifices) for specification.                                            |
| `weirs`          | pd.DataFrame | Table containing weir control structures, and their attributes. See [model inputs: weirs](/pipedream/model-inputs.html#weirs) for specification.                                                     |
| `pumps`          | pd.DataFrame | Table containing pump control structures, and their attributes. See [model inputs: pumps](/pipedream/model-inputs.html#pumps) for specification.                                                     |
| `dt`             | float        | Default timestep of model (in seconds). Defaults to 60 seconds.                                                                                                                                      |
| `min_depth`      | float        | Minimum depth allowed at junctions and superjunctions (in meters). Defaults to 1e-5 m.                                                                                                               |
| `method`         | str          | Method for computing internal states in superlinks. Must be one of the following: <br><br>- `b` : Backwards (default) <br>- `f` : Forwards<br>- `lsq` : Least-squares<br><br> Defaults to `b`.                          |
| `auto_permute`   | bool         | If `True`, permute the superjunctions to enable use of a banded matrix solver and increase solver speed. Superjunctions are permuted using the Reverse Cuthill-McKee algorithm. Defaults to `False`. |
| `internal_links` | int          | If junctions/links are not provided, this gives the number of internal links that will be generated inside each superlink. Defaults to `4`.                                                          |
|------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Running the `SuperLink` model with `step`

The hydraulic model is advanced forward in time using the `SuperLink.step` method:

<b>`step`</b>`(self, H_bc=None, Q_in=None, Q_0Ik=None, u_o=None, u_w=None, u_p=None, dt=None,
               first_time=False, implicit=True, banded=False, first_iter=True,
               num_iter=1, head_tol=0.0015)`

> Advance model forward to next time step, computing hydraulic states.

  Parameters

|--------------|------------------|----------------------------------------------------------------------------------------|
| Argument     | Type             | Description                                                                            |
|--------------|------------------|----------------------------------------------------------------------------------------|
| `H_bc`       | np.ndarray (M)   | Boundary stage at each superjunction (m)                                               |
| `Q_in`       | np.ndarray (M)   | Direct inflow at each superjunction (m^3/s)                                            |
| `Q_0Ik`      | np.ndarray (NIk) | Direct inflow at each junction (m^3/s)                                                 |
| `u_o`        | np.ndarray (NO)  | Orifice control signal. Represents fraction of orifice open (0-1)                      |
| `u_w`        | np.ndarray (NW)  | Weir control signal. Represents fraction of weir open (0-1)                            |
| `u_p`        | np.ndarray (NP)  | Pump control signal. Represents fraction maximum pump flow (0-1)                       |
| `dt`         | float            | Time step to advance                                                                   |
| `first_time` | bool             | Set `True` if this is the first step the model has performed                           |
| `banded`     | bool             | If `True`, use banded matrix solver                                                    |
| `first_iter` | bool             | `True` if this is the first iteration of this step when iterating towards convergence. |
| `num_iter`   | int              | Maximum number of iterations to perform when iterating towards convergence             |
| `head_tol`   | float            | Maximum allowable head tolerance when iterating towards convergence (m)                |
|--------------|------------------|----------------------------------------------------------------------------------------|

  Returns
  
  `None`


## Attributes of the `SuperLink` model

### Model dimensions

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `M`       | int  | Number of superjunctions (M) |
| `NK`      | int  | Number of superlinks (NK) |
| `NIk`     | int  | Number of junctions (NIK) |
| `Nik`     | int  | Number of links (NIK) |
|----------|------|-------------|

### Model states

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `t`       | float | Current time (s) |
| `H_j`     | np.ndarray (M) | Superjunction heads (m) |
| `h_Ik`    | np.ndarray (NIk) | Junction depths (m) |
| `Q_ik`    | np.ndarray (Nik) | Link flows (m^3/s) |
| `Q_uk`    | np.ndarray (NK)  | Flows into upstream ends of superlinks (m^3/s) |
| `Q_dk`    | np.ndarray (NK)  | Flows out of downstream ends of superlinks (m^3/s) |
| `Q_o`     | np.ndarray (NO)  | Orifice flows (m^3/s) |
| `Q_w`     | np.ndarray (NW)  | Weir flows (m^3/s) |
| `Q_p`     | np.ndarray (NP)  | Pump flows (m^3/s) |
| `A_ik`    | np.ndarray (Nik) | Cross-sectional area of flow in links (m^2) |
| `Pe_ik`   | np.ndarray (Nik) | Wetted perimeter in links (m) |
| `R_ik`    | np.ndarray (Nik) | Hydraulic radius in links (m) |
| `B_ik`    | np.ndarray (Nik) | Top width of flow in links (m) |
| `A_sj`    | np.ndarray (M) | Superjunction surface area (m^2) |
| `V_sj`    | np.ndarray (M) | Superjunction stored volumes (m^3) |
|----------|------|-------------|

### Model indexing

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `_i`   | np.ndarray (Nik) | Array indicating the index of each link |
| `_I`   | np.ndarray (NIk) | Array indicating the index of each junction |
| `_ki`  | np.ndarray (Nik) | Array indicating the index of the superlink that each link belongs to |
| `_kI`  | np.ndarray (NIk) | Array indicating the index of the superlink that each junction belongs to |
|----------|------|-------------|

