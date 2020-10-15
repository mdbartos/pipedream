# Infiltration solver API documentation

This section enumerates all the methods of the `GreenAmpt` class.

## Instantiating the `GreenAmpt` model

An infiltration/runoff model is instantiated using the `pipedream_solver.hydrology.GreenAmpt` class.

### Initialization parameters

The `GreenAmpt` class is initialized with the following parameters:

|---------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Attribute     | Type         | Description                                                                                                                                                           |
|---------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `soil_params` | pd.DataFrame | Table containing soil parameters for each computational element. See [model inputs: soil parameters](/pipedream/model-inputs.html#soil-parameters) for specification. |
|---------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Running the `GreenAmpt` model with `step`

The hydraulic model is advanced forward in time using the `SuperLink.step` method:

<b>`step`</b>`(self, dt, i)`

> Advance model forward in time, computing infiltration rate and cumulative infiltration.

  Parameters

|----------|------|-------------|
| Argument| Type | Description |
|----------|------|-------------|
| `dt` | float | Time step (s) |
| `i` | np.ndarray (N) | Precipitation rate (m/s) |
|----------|------|-------------|

  Returns
  
  `None`


## Attributes of the `GreenAmpt` model

### Model dimensions

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `N`        | int  | Number of subcatchments (N) |
|----------|------|-------------|

### Model states

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `f` | np.ndarray (N) | Infiltration rate (m/s) |
| `F` | np.ndarray (N) | Cumulative infiltration depth (m) |
| `d` | np.ndarray (N) | Ponded depth (m) |
| `T` | np.ndarray (N) | Recovery time (s) |
|----------|------|-------------|

### Other attributes

|----------|------|-------------|
| Attribute | Type | Description |
|----------|------|-------------|
| `is_saturated` | np.ndarray (N) | Indicates whether soil element is currently saturated (`True`/`False`) |
|----------|------|-------------|

