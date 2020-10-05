# Water quality solver API documentation

This section enumerates all the methods of the `QualityBuilder` class

## Instantiating the `QualityBuilder` model

A water quality model is instantiated using the `pipedream_solver.transport.QualityBuilder` class.

### Initialization parameters

The `QualityBuilder` class is initialized with the following parameters:

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `hydraulics` | `SuperLink` | A SuperLink instance describing the system hydraulics.  |
| `superjunction_params` | `pd.DataFrame` | Table containing superjunction water quality parameters. See [model inputs: superjunction parameters](/pipedream/model-inputs.html#superjunction-parameters) for specification. |
| `superlink_params` | `pd.DataFrame` | Table containing superlink water quality parameters. See [model inputs: superlink parameters](/pipedream/model-inputs.html#superlink-parameters) for specification. |
| `junction_params` | `pd.DataFrame` | Table containing junction water quality parameters (optional). See [model inputs: junction parameters](/pipedream/model-inputs.html#junction-parameters) for specification.   |
| `link_params` | `pd.DataFrame` | Table containing link water quality parameters (optional). See [model inputs: link parameters](/pipedream/model-inputs.html#link-parameters) for specification. |
| `c_min` | float | Minimum allowed contaminant concentration (default 0). |
| `c_max` | float | Maximum allowed contaminant concentration (default `inf`). |
|----------|------|-------------|

## Running the `QualityBuilder` model with `step`

The hydraulic model is advanced forward in time using the `SuperLink.step` method:

<b>`step`</b>`(self, dt=None, c_bc=None, c_0j=None, Q_0j=None, c_0Ik=None, Q_0Ik=None, u_j_frac=0.0)`

> Advance model forward to next time step, computing hydraulic states.

  Parameters

|--------------|------------------|----------------------------------------------------------------------------------------|
| Argument     | Type             | Description                                                                            |
|--------------|------------------|----------------------------------------------------------------------------------------|
| `dt`         | float            | Time step to advance                                                                   |
| `c_bc`       | np.ndarray (M)   | Boundary concentration at each superjunction (*/m^3)                                 |
| `c_0j`       | np.ndarray (M)   | Contaminant concentration in direct superjunction inflow `Q_in` (*/m^3). Defaults to 0.  |
| `Q_0j`       | np.ndarray (M)   | Direct inflow at each superjunction (m^3/s). Defaults to `_Q_in` of underlying `SuperLink` model.  |
| `c_0Ik`      | np.ndarray (NIk) | Contaminant concentration in direct junction inflow `Q_0Ik`. Defaults to 0. |
| `Q_0Ik`      | np.ndarray (NIk) | Direct inflow at each junction (m^3/s). Defaults to `_Q_0Ik` of underlying `SuperLink` model. |
| `u_j_frac` | float | (Deprecated). |
|--------------|------------------|----------------------------------------------------------------------------------------|

  Returns
  
  `None`


## Attributes of the `QualityBuilder` model

### Model states

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `c_j`     | np.ndarray (M) | Contaminant concentration in superjunctions (*/m^3) |
| `c_Ik` | np.ndarray (NIk) | Contaminant concentration in junctions (*/m^3) |
| `c_ik`    | np.ndarray (Nik) | Contaminant concentration in links (*/m^3) |
| `c_uk` | np.ndarray (NK) | Contaminant concentration entering upstream end of superlinks (*/m^3) |
| `c_dk` | np.ndarray (NK) | Contaminant concentration exiting downstream end of superlinks (*/m^3) |
|----------|------|-------------|

### Other attributes

<!-- z_inv_j  : Superjunction invert elevation (m) -->
<!-- z_inv_uk : Offset of superlink upstream invert above superjunction (m) -->
<!-- z_inv_dk : Offset of superlink downstream invert above superjunction (m) -->

