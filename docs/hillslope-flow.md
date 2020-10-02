# Flow on a hillslope

This tutorial demonstrates how to use the *pipedream* solver to simulate the flow of water on a simple hillslope.

## Import modules


```python
import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
```

## Load model data

Model structure can be saved to or loaded from CSV files. In this case, our model consists of a set of superjunctions and a set of superlinks.


```python
input_path = '../data/hillslope'
superjunctions = pd.read_csv(f'{input_path}/hillslope_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/hillslope_superlinks.csv')
```

#### Superjunctions

Superjunctions are basic finite volumes, and may represent manholes, retention ponds, or other bodies of water.


```python
superjunctions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe" style="font-size: 8pt">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>id</th>
      <th>z_inv</th>
      <th>h_0</th>
      <th>bc</th>
      <th>storage</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>max_depth</th>
      <th>map_x</th>
      <th>map_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.00001</td>
      <td>False</td>
      <td>functional</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>200.0</td>
      <td>inf</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0.00001</td>
      <td>False</td>
      <td>functional</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>inf</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Superlinks

Superlinks are sections of channel or conduit that connect superjunctions together. Each superlink consists of a number of internal junctions and nodes connected in a linear fashion. In this case, we have a single superlink that connects the uphill superjunction to the downhill superjunction.


```python
superlinks
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe" style="font-size: 8pt">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>id</th>
      <th>sj_0</th>
      <th>sj_1</th>
      <th>in_offset</th>
      <th>out_offset</th>
      <th>dx</th>
      <th>n</th>
      <th>shape</th>
      <th>g1</th>
      <th>g2</th>
      <th>g3</th>
      <th>g4</th>
      <th>Q_0</th>
      <th>h_0</th>
      <th>ctrl</th>
      <th>A_s</th>
      <th>A_c</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000</td>
      <td>0.035</td>
      <td>rect_open</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00001</td>
      <td>False</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Internal links and junctions can be specified explicitly, or generated automatically inside each superlink. In this case, we will use 24 automatically-generated internal links.


```python
internal_links = 24
```

## Instantiate model

The hydraulic model is instantiated with the `SuperLink` class.


```python
model = SuperLink(superlinks, superjunctions, 
                  internal_links=internal_links)
```

Next, we specify the model parameters, including the default time step `dt`, the inflow into each superjunction `Q_in`, and the inflow into each internal junction `Q_0Ik`.


```python
dt = 10                            # Model time step (s)
Q_in = 1e-3 * np.ones(model.M)     # Flow into each internal junction (cms)
Q_0Ik = 1e-3 * np.ones(model.NIk)  # Flow into each internal junction (cms)
```

Note that `model.M` is the number of superjunctions, and `model.NIk` is the number of internal junctions.

We can visualize the model with `plot_profile`. Note that the hillslope at the start of the model run is dry.


```python
# Plot profile from superjunctions 0 to 1 (uphill to downhill)
_ = model.plot_profile([0, 1], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/hillslope-flow/hillslope-flow-0.png)


## Run model

#### Apply rain to hillslope for 4 hours

Applying inflow to each superjunction and internal link causes water to flow down the hill.


```python
# For each timestep...
for _ in range(4 * 360):
    # Advance model forward in time
    model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
```


```python
_ = model.plot_profile([0, 1], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/hillslope-flow/hillslope-flow-1.png)


#### Progress forward in time for 3 hours with no rain

Ceasing the flow input and advancing the model in time causes the water to settle at the bottom of the hill.


```python
for _ in range(3 * 360):
    model.step(dt=dt)
```


```python
_ = model.plot_profile([0, 1], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/hillslope-flow/hillslope-flow-2.png)

