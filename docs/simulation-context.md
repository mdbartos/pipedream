
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
superjunctions = pd.read_csv(f'{input_path}/hillslope_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/hillslope_superlinks.csv')
```

## Instantiate model


```python
dt = 10
internal_links = 24

model = SuperLink(superlinks, superjunctions, 
                  internal_links=internal_links)
```

## Create tables of input data


```python
Q_in = pd.DataFrame.from_dict(
    {
        0 :  np.zeros(model.M),
        3600: np.zeros(model.M),
        3601: 1e-3 * np.ones(model.M),
        18000 : 1e-3 * np.ones(model.M),
        18001 : np.zeros(model.M),
        28000 : np.zeros(model.M)
    }, orient='index')

Q_Ik = pd.DataFrame.from_dict(
    {
        0 :  np.zeros(model.NIk),
        3600: np.zeros(model.NIk),
        3601: 1e-3 * np.ones(model.NIk),
        18000 : 1e-3 * np.ones(model.NIk),
        18001 : np.zeros(model.NIk),
        28000 : np.zeros(model.NIk)
    }, orient='index'
)
```

Inspecting the table of superjunction inflows:


```python
Q_in
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3600</th>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3601</th>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>18000</th>
      <td>0.001</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>18001</th>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>28000</th>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>



We can visualize the inflows to the system as well:


```python
fig, ax = plt.subplots(2)

Q_in.sum(axis=1).plot(ax=ax[0], title='Superjunction inflow', c='k')
Q_Ik.sum(axis=1).plot(ax=ax[1], title='Internal junction inflow', c='k')
ax[0].set_ylabel('Inflow (cms)')
ax[1].set_ylabel('Inflow (cms)')
ax[1].set_xlabel('Time (s)')
plt.tight_layout()
```


![png](output_11_0.png)


## Run simulation using context manager

The simulation context manager provides a convenient tool for running simulations. It provides a systematic way to:

- Handle input data
- Record simulation outputs
- Provide information on the status of the simulation
- Adaptively modify the timestep of the simulation

In this case, we will use the simulation context manager to read and parse our flow input tables, print the simulation progress, and record the internal depths and flows in the system.


```python
# Create simulation context manager
with Simulation(model, Q_in=Q_in, Q_Ik=Q_Ik) as simulation:
    # While simulation time has not expired...
    while simulation.t <= simulation.t_end:
        # Step model forward in time
        simulation.step(dt=dt)
        # Record internal depth and flow states
        simulation.record_state()
        # Print progress bar
        simulation.print_progress()
```

    [==================================================] 100.0% [2.37 s]

We can see the end state of the system by plotting the model profile.


```python
sns.set_palette('cool')
_ = model.plot_profile([0, 1], width=100)
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simulation-context/simulation-context-0.png)


## Visualize simulation results

The simulation context manager makes it easy to access and visualize model outputs. We can access any outputs recorded by `simulation.record_state` by accessing the `simulation.states` attribute. By default, `simulation.states` contains the following states:

- `H_j`: Superjunction heads (m)
- `Q_ik`: Flow in internal links (cms)
- `h_Ik`: Depth of water in internal junctions (m)
- `Q_uk`: Flow into upstream ends of superlinks (cms)
- `Q_dk`: Flow out of downstream ends of superlinks (cms)

#### Plot hydraulic head at superjunctions

The hydraulic heads at each superjunction can be plotted by accessing the `simulation.states.H_j` dataframe. Note that superjunction `0` corresponds to the uphill superjunction, and superjunction `1` corresponds to the downhill superjunction.


```python
sns.set_palette('husl', 2)
simulation.states.H_j.plot()

plt.title('Hydraulic head at superjunctions')
plt.xlabel('Time (s)')
plt.ylabel('Head (m)')
```




    Text(0, 0.5, 'Head (m)')




![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simulation-context/simulation-context-1.png)


#### Plot flow in internal links

The flows in each internal link can be plotted by accessing the `simulation.states.Q_ik` dataframe. Note that the colormap goes from red (uphill) to violet (downhill).


```python
sns.set_palette('husl', internal_links)
simulation.states.Q_ik.plot(legend=False)

plt.title('Flow in internal links')
plt.xlabel('Time (s)')
plt.ylabel('Flow (cms)')
```




    Text(0, 0.5, 'Flow (cms)')




![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simulation-context/simulation-context-2.png)


#### Plot water depth at internal junctions

The water depth in each internal junction can be plotted by accessing the `simulation.states.h_Ik` dataframe.


```python
sns.set_palette('husl', internal_links + 1)
simulation.states.h_Ik.plot(legend=False)

plt.title('Depth in internal junctions')
plt.xlabel('Time (s)')
plt.ylabel('Depth (m)')
```




    Text(0, 0.5, 'Depth (m)')




![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/simulation-context/simulation-context-3.png)

