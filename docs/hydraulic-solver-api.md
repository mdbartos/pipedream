# Hydraulic solver API documentation

This section enumerates all the methods of the `SuperLink` class

## Instantiating the `SuperLink` model

A hydraulic model is instantiated using the `pipedream_solver.hydraulics.SuperLink` class.

### Initialization parameters

The `SuperLink` class is initialized with the following parameters:

<b>`superlinks`</b>: pd.DataFrame
> Table containing all superlinks in the network along with their attributes.

> See [model inputs: superlinks](/pipedream/model-inputs.html#superlinks) for specification.

<b>`superjunctions`</b>: pd.DataFrame
> Table containing all superjunctions in the network along with their attributes.

> See [model inputs: superjunctions](/pipedream/model-inputs.html#superjunctions) for specification.

<b>`links`</b>: pd.DataFrame (optional)
> Table containing all links in the network along with their attributes.

> See [model inputs: links](/pipedream/model-inputs.html#links) for specification.

<b>`junctions`</b>: pd.DataFrame (optional)
> Table containing all junctions in the network along with their attributes.

> See [model inputs: junctions](/pipedream/model-inputs.html#junctions) for specification.

<b>`transects`</b>: dict (optional)
> Dictionary describing nonfunctional channel cross-sectional geometries.

> See [model inputs: transects](/pipedream/model-inputs.html#transects) for specification.

<b>`storages`</b>: dict (optional)
> Dictionary describing tabular storages for superjunctions.

> See [model inputs: storages](/pipedream/model-inputs.html#storages) for specification.

<b>`orifices`</b>: pd.DataFrame (optional)
> Table containing orifice control structures, and their attributes.

> See [model inputs: orifices](/pipedream/model-inputs.html#orifices) for specification.

<b>`weirs`</b>: pd.DataFrame (optional)
> Table containing weir control structures, and their attributes.

> See [model inputs: weirs](/pipedream/model-inputs.html#weirs) for specification.

<b>`pumps`</b>: pd.DataFrame (optional)
> Table containing pump control structures and their attributes.

> See [model inputs: pumps](/pipedream/model-inputs.html#pumps) for specification.

<b>`dt`</b>: float
> Default timestep of model (in seconds). Defaults to 60 seconds.

<b>`min_depth`</b>: float
> Minimum depth allowed at junctions and superjunctions (in meters). Defaults to 1e-5 m.

<b>`method`</b>: str
> Method for computing internal states in superlinks. Must be one of the following:

- `b`   : Backwards (default)
- `f`   : Forwards
- `lsq` : Least-squares

> Defaults to `b`.

<b>`auto_permute`</b>: bool
>If `True`, permute the superjunctions to enable use of a banded matrix solver and increase solver speed. Superjunctions are permuted using the Reverse Cuthill-McKee algorithm. Defaults to `False`.

<b>`internal_links`</b>: int
>If junctions/links are not provided, this gives the number of internal links that will be generated inside each superlink. Defaults to `4`.

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

<b>`M`</b>: int
> Number of superjunctions (M).

<b>`NK`</b>: int
> Number of superlinks (NK).

<b>`NIk`</b>: int
> Number of junctions (NIk).

<b>`Nik`</b>: int
> Number of junctions (Nik).

### Model states

<b>`t`</b>: float
> Current time (s)

<b>`H_j`</b>: np.ndarray (M)
> Superjunction heads (m)

<b>`h_Ik`</b>: np.ndarray (NIk)
> Junction depths (m)

<b>`Q_ik`</b>: np.ndarray (Nik)
> Link flows (m^3/s)

<b>`Q_uk`</b>: np.ndarray (NK)
> Flows into upstream ends of superlinks (m^3/s)

<b>`Q_dk`</b>: np.ndarray (NK)
> Flows into downstream ends of superlinks (m^3/s)

<b>`Q_o`</b>: np.ndarray (NO)
> Orifice flows (m^3/s)

<b>`Q_w`</b>: np.ndarray (NW)
> Weir flows (m^3/s)

<b>`Q_p`</b>: np.ndarray (NP)
> Pump flows (m^3/s)

<b>`A_ik`</b>: np.ndarray (Nik)
> Cross-sectional area of flow in links (m^2)

<b>`Pe_ik`</b>: np.ndarray (Nik)
> Wetted perimeter in links (m)

<b>`R_ik`</b>: np.ndarray (Nik)
> Hydraulic radius in links (m)

<b>`B_ik`</b>: np.ndarray (Nik)
> Top width of flow in links (m)

<b>`A_sj`</b>: np.ndarray (M)
> Superjunction surface areas (m^2)

<b>`V_sj`</b>: np.ndarray (M)
> Superjunction stored volumes (m^3)

### Model indexing

<b>`_ki`</b>: np.ndarray (Nik)
> Array indicating the index of the superlink that each link belongs to.

<b>`_kI`</b>: np.ndarray (NIk)
> Array indicating the index of the superlink that each junction belongs to.

### Other attributes

<!-- z_inv_j  : Superjunction invert elevation (m) -->
<!-- z_inv_uk : Offset of superlink upstream invert above superjunction (m) -->
<!-- z_inv_dk : Offset of superlink downstream invert above superjunction (m) -->

## Methods of the `SuperLink` model

