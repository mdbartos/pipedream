# Hydraulic solver API documentation

This section enumerates all the methods of the `SuperLink` class

## Instantiating the `SuperLink` model

A hydraulic model is instantiated using the `pipedream_solver.SuperLink` class.
This class is initialized with the following parameters:

<b>`superlinks`</b>: pd.DataFrame
> Table containing all superlinks in the network along with their attributes.

> See [model inputs](/pipedream/model-inputs.html#superlinks) for specification.

<b>`superjunctions`</b>: pd.DataFrame
> Table containing all superjunctions in the network along with their attributes.

> See [model inputs](/pipedream/model-inputs.html#superjunctions) for specification.

<b>`links`</b>: pd.DataFrame (optional)
> Table containing all links in the network along with their attributes.

> See [model inputs](/pipedream/model-inputs.html#links) for specification.

<b>`junctions`</b>: pd.DataFrame (optional)
> Table containing all junctions in the network along with their attributes.

> See [model inputs](/pipedream/model-inputs.html#junctions) for specification.

<b>`transects`</b>: dict (optional)
> Dictionary describing nonfunctional channel cross-sectional geometries.

> See [model inputs](/pipedream/model-inputs.html#transects) for specification.

<b>`storages`</b>: dict (optional)
> Dictionary describing tabular storages for superjunctions.

> See [model inputs](/pipedream/model-inputs.html#storages) for specification.

<b>`orifices`</b>: pd.DataFrame (optional)
> Table containing orifice control structures, and their attributes.

> See [model inputs](/pipedream/model-inputs.html#orifices) for specification.

<b>`weirs`</b>: pd.DataFrame (optional)
> Table containing weir control structures, and their attributes.

> See [model inputs](/pipedream/model-inputs.html#weirs) for specification.

<b>`pumps`</b>: pd.DataFrame (optional)
> Table containing pump control structures and their attributes.

> See [model inputs](/pipedream/model-inputs.html#pumps) for specification.

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

## Running the `SuperLink` model

The hydraulic model is advanced forward in time using the `SuperLink.step` method.

## Attributes of the `SuperLink` model

## Methods of the `SuperLink` model

