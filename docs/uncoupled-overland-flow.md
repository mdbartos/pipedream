# Uncoupled infiltration and flow routing

## Import modules


```python
import numpy as np
import pandas as pd
from pipedream_solver.hydrology import GreenAmpt
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation

import matplotlib.pyplot as plt
import seaborn as sns
```

## Load model data


```python
input_path = '../data/hillslope'
superjunctions = pd.read_csv(f'{input_path}/hillslope_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/hillslope_superlinks.csv')
soil_params = pd.read_csv('../data/hillslope/hillslope_soil_params.csv')
```

### Inspect soil parameters


```python
soil_params.head()
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
      <th>psi_f</th>
      <th>Ks</th>
      <th>theta_s</th>
      <th>theta_i</th>
      <th>A_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.020529</td>
      <td>0.000011</td>
      <td>0.37</td>
      <td>0.15</td>
      <td>108.695652</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.020529</td>
      <td>0.000011</td>
      <td>0.37</td>
      <td>0.15</td>
      <td>217.391304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.020529</td>
      <td>0.000011</td>
      <td>0.37</td>
      <td>0.15</td>
      <td>217.391304</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.020529</td>
      <td>0.000011</td>
      <td>0.37</td>
      <td>0.15</td>
      <td>217.391304</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.020529</td>
      <td>0.000011</td>
      <td>0.37</td>
      <td>0.15</td>
      <td>217.391304</td>
    </tr>
  </tbody>
</table>
</div>



## Instantiate hydraulic/infiltration models


```python
internal_links = 24
superlink = SuperLink(superlinks, superjunctions, internal_links=internal_links)
greenampt = GreenAmpt(soil_params)
```

### Specify precipitation inputs


```python
# Specify precipitation on each soil element in (m/s)
i_0 = 50 / 1000 / 3600 * np.ones(superlink.NIk)
i_1 = np.zeros(superlink.NIk)
# Specify time step
dt = 60
```

## Run infiltration model


```python
# Create dicts to store state data
infiltration_rate = {}
runoff_rate = {}
ponded_depth = {}

# Run simulation for 24 hours...
while greenampt.t < (24 * 3600):
    # For first 12 hours...
    if greenampt.t < (12 * 3600):
        # Compute infiltration with active rainfall
        greenampt.step(dt=dt, i=i_0)
    # For last 12 hours...
    else:
        # Compute infiltration when rainfall rate is zero
        greenampt.step(dt=dt, i=i_1)
    # Export system states
    infiltration_rate[greenampt.t] = greenampt.f.copy()
    runoff_rate[greenampt.t] = greenampt.Q.copy()
    ponded_depth[greenampt.t] = greenampt.d.copy()
  
# Convert state dicts to dataframes
infiltration_rate = pd.DataFrame.from_dict(infiltration_rate, orient='index')
runoff_rate = pd.DataFrame.from_dict(runoff_rate, orient='index')
ponded_depth = pd.DataFrame.from_dict(ponded_depth, orient='index')
```

## Visualize infiltration results


```python
fig, ax = plt.subplots(3, figsize=(10, 10))
infiltration_rate.mean(axis=1).plot(ax=ax[0], legend=False, color='r')
runoff_rate.sum(axis=1).plot(ax=ax[1], legend=False, color='b')
ponded_depth.mean(axis=1).plot(ax=ax[2], legend=False, color='k')

ax[0].set_title('Infiltration rate (m/s)')
ax[1].set_title('Runoff rate (m^3/s)')
ax[2].set_title('Ponded depth (m)')

ax[0].xaxis.set_ticklabels([])
ax[1].xaxis.set_ticklabels([])
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/uncoupled-overland-flow/uncoupled-overland-flow-0.png)


## Run hydraulic model with runoff input


```python
# Set initial timestep
dt = 30

# Spin up model
superlink.spinup(n_steps=200)

# Create simulation context manager
with Simulation(superlink, Q_in=runoff_rate, t_end=(32 * 3600)) as simulation:
    # While simulation time has not expired...
    while simulation.t <= simulation.t_end:
        # Step model forward in time
        simulation.step(dt=dt)
        # Record internal depth and flow states
        simulation.record_state()
        simulation.model.reposition_junctions()
        # Print progress bar
        simulation.print_progress()
```

    [==================================================] 100.0% [2.35 s]

## Visualize hydraulic modeling results


```python
sns.set_palette('cool')
_ = simulation.model.plot_profile([0, 1], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/uncoupled-overland-flow/uncoupled-overland-flow-1.png)



```python
sns.set_palette('rainbow_r', internal_links + 1)
simulation.states.h_Ik.plot(legend=False)

plt.title('Internal junction depths')
plt.ylabel('Depth (m)')
plt.xlabel('Time (s)')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/uncoupled-overland-flow/uncoupled-overland-flow-2.png)



```python
sns.set_palette('rainbow_r', internal_links)
simulation.states.Q_ik.plot(legend=False)

plt.title('Internal link flows')
plt.ylabel('Flow (cms)')
plt.xlabel('Time (s)')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/uncoupled-overland-flow/uncoupled-overland-flow-3.png)

