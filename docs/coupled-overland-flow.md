# Fully coupled infiltration and runoff routing

## Import modules


```python
import numpy as np
import pandas as pd
from pipedream_solver.hydrology import GreenAmpt
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('husl')

%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 5)
```

## Load model data


```python
input_path = '../data/hillslope'
superjunctions = pd.read_csv(f'{input_path}/hillslope_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/hillslope_superlinks.csv')
soil_params = pd.read_csv(f'{input_path}/hillslope_soil_params.csv')
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
```


```python
# Create dict to collect infiltration data
f = {}

# Set initial timestep
dt = 15

# Spin up model
superlink.spinup(n_steps=200)

# Create simulation context manager
with Simulation(superlink, t_start=0, t_end=(18 * 3600)) as simulation:
    # While simulation time has not expired...
    while simulation.t <= simulation.t_end:
        greenampt.d = superlink.h_Ik
        if simulation.t < (12 * 3600):
            greenampt.step(dt=dt, i=i_0)
        else:
            greenampt.step(dt=dt, i=i_1)
        Q_Ik = greenampt.Q
        # Step hydraulic model forward in time
        simulation.step(dt=dt, Q_Ik=Q_Ik)
        # Record internal depth and flow states
        simulation.record_state()
        f[simulation.t] = greenampt.f.copy()
        # Print progress bar
        simulation.print_progress()
```

    [==================================================] 100.0% [4.33 s]

## Visualize hydraulic modeling results


```python
sns.set_palette('cool')
_ = simulation.model.plot_profile([0, 1], width=100)
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/coupled-overland-flow/coupled-overland-flow-0.png)



```python
sns.set_palette('rainbow_r', internal_links + 1)
simulation.states.h_Ik.plot(legend=False)

plt.title('Internal junction depths')
plt.ylabel('Depth (m)')
plt.xlabel('Time (s)')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/coupled-overland-flow/coupled-overland-flow-1.png)



```python
sns.set_palette('rainbow_r', internal_links)
simulation.states.Q_ik.plot(legend=False)

plt.title('Internal link flows')
plt.ylabel('Flow (cms)')
plt.xlabel('Time (s)')
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/coupled-overland-flow/coupled-overland-flow-2.png)



```python
f = pd.DataFrame.from_dict(f, orient='index')

f.plot(legend=False)
plt.title('Infiltration in channel segments')
plt.xlabel('Time (s)')
plt.ylabel('Infiltration rate (m/s)')
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/coupled-overland-flow/coupled-overland-flow-3.png)

