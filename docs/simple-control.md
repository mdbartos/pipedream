# Simple example of dynamic control

## Import modules


```python
import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation

import matplotlib.pyplot as plt
import seaborn as sns
```

## Load model data


```python
input_path = '../data/hillslope'
superjunctions = pd.read_csv(f'{input_path}/hillslope_control_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/hillslope_control_superlinks.csv')
```


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
<table border="1" class="dataframe">
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
      <td>2</td>
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
      <td>1</td>
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
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0.00001</td>
      <td>False</td>
      <td>functional</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>inf</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0.00001</td>
      <td>False</td>
      <td>functional</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>inf</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




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
<table border="1" class="dataframe">
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
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
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



## Create a control orifice


```python
orifices = {
    0 : {
        'id' : 0,
        'sj_0' : 1,
        'sj_1' : 2,
        'A' : 0.3048**2,
        'orientation' : 'side',
        'z_o' : 0,
        'y_max' : 0.3048,
        'C' : 0.67
    }
}

orifices = pd.DataFrame.from_dict(orifices, orient='index')
```


```python
orifices
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sj_0</th>
      <th>sj_1</th>
      <th>A</th>
      <th>orientation</th>
      <th>z_o</th>
      <th>y_max</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0.092903</td>
      <td>side</td>
      <td>0</td>
      <td>0.3048</td>
      <td>0.67</td>
    </tr>
  </tbody>
</table>
</div>



## Instantiate model


```python
internal_links = 24
model = SuperLink(superlinks, superjunctions, orifices=orifices,
                  internal_links=internal_links)
model.spinup(n_steps=1000)
```


```python
dt = 10                            # Model time step (s)
Q_in = 1e-3 * np.ones(model.M)     # Flow into each internal junction (cms)
Q_0Ik = 1e-3 * np.ones(model.NIk)  # Flow into each internal junction (cms)
u_0 = np.zeros(len(orifices))      # Orifice closed signal
u_1 = np.ones(len(orifices))       # Orifice open signal
```


```python
# Plot profile from superjunctions 0 to 1 (uphill to downhill)
_ = model.plot_profile([0, 1, 2, 3], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simple-control/simple-control-0.png)


## Run model with orifice closed


```python
# For each timestep...
while model.t < (8 * 3600):
    if model.t < (4 * 3600):
        model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_0)
    else:
        model.step(dt=dt, u_o=u_0)
```


```python
_ = model.plot_profile([0, 1, 2, 3], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simple-control/simple-control-1.png)


## Continue running model with orifice open


```python
while model.t < (12 * 3600):
    model.step(dt=dt, u_o=u_1)
```


```python
_ = model.plot_profile([0, 1, 2, 3], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simple-control/simple-control-2.png)


## Run full simulation with control


```python
model = SuperLink(superlinks, superjunctions, orifices=orifices,
                  internal_links=internal_links)
model.spinup(n_steps=1000)

# Create simulation context manager
with Simulation(model, t_start=0, t_end=(24 * 3600)) as simulation:
    # While simulation time has not expired...
    while simulation.t <= simulation.t_end:
        if simulation.t < (8 * 3600):
            simulation.step(dt=dt, Q_in=Q_in, Q_Ik=Q_0Ik, u_o=u_0)
        elif (simulation.t > 8 * 3600) and (simulation.t < 12 * 3600):
            simulation.step(dt=dt, u_o=u_0)
        else:
            simulation.step(dt=dt, u_o=u_1)
        # Record internal depth and flow states
        simulation.record_state()
        # Reposition junctions
        simulation.model.reposition_junctions()
        # Print progress bar
        simulation.print_progress()
```

    [==================================================] 100.0% [4.77 s]

## Visualize results

### Profile at end of simulation


```python
_ = model.plot_profile([0, 1, 2, 3], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simple-control/simple-control-3.png)


### Superjunction heads


```python
sns.set_palette('husl')
simulation.states.H_j.plot()
plt.axvline(12 * 3600, c='k', alpha=0.3, linestyle='--')
plt.title('Superjunction heads')
plt.ylabel('Head (m)')
_ = plt.xlabel('Time (s)')
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simple-control/simple-control-4.png)


### Internal superlink depths


```python
fig, ax = plt.subplots(2, figsize=(10, 10))

(simulation.states.h_Ik[np.flatnonzero(model._kI == 0)]
 .plot(ax=ax[0], legend=False, alpha=0.8, cmap='rainbow_r'))
(simulation.states.h_Ik[np.flatnonzero(model._kI == 1)]
 .plot(ax=ax[1], legend=False, alpha=0.8, cmap='rainbow_r'))

ax[0].axvline(12 * 3600, c='k', alpha=0.3, linestyle='--')
ax[1].axvline(12 * 3600, c='k', alpha=0.3, linestyle='--')

ax[0].set_title('Depth in internal junctions (upstream of orifice)')
ax[1].set_title('Depth in internal junctions (downstream of orifice)')
ax[0].xaxis.set_ticklabels([])
ax[0].set_ylabel('Depth (m)')
ax[1].set_ylabel('Depth (m)')
ax[1].set_xlabel('Time (s)')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simple-control/simple-control-5.png)


### Internal superlink flows


```python
fig, ax = plt.subplots(2, figsize=(10, 10))

(simulation.states.Q_ik[np.flatnonzero(model._ki == 0)]
 .plot(ax=ax[0], legend=False, alpha=0.8, cmap='rainbow_r'))
(simulation.states.Q_ik[np.flatnonzero(model._ki == 1)]
 .plot(ax=ax[1], legend=False, alpha=0.8, cmap='rainbow_r'))

ax[0].axvline(12 * 3600, c='k', alpha=0.3, linestyle='--')
ax[1].axvline(12 * 3600, c='k', alpha=0.3, linestyle='--')

ax[0].set_title('Flow in internal links (upstream of orifice)')
ax[1].set_title('Flow in internal links (downstream of orifice)')
ax[0].xaxis.set_ticklabels([])
ax[0].set_ylabel('Flow (cms)')
ax[1].set_ylabel('Flow (cms)')
ax[1].set_xlabel('Time (s)')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simple-control/simple-control-6.png)

