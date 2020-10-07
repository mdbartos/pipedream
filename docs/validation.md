# Validation with real-world stormwater network

## Import modules


```python
import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = (10, 5)
```

## Read input data


```python
# Provide base URL where data is stored
baseurl = 'http://pipedream-solver.s3.us-east-2.amazonaws.com/data/validation'

# Read model structure files
superjunctions = pd.read_csv(f'{baseurl}/validation_superjunctions.csv')
superlinks = pd.read_csv(f'{baseurl}/validation_superlinks.csv')
weirs = pd.read_csv(f'{baseurl}/validation_weirs.csv')
```

## Create model


```python
# Instantiate superlink object
superlink = SuperLink(superlinks, superjunctions, weirs=weirs,
                      auto_permute=True, internal_links=4, min_depth=0.0)

# Spin-up model to avoid running dry
superlink.reposition_junctions()
superlink.spinup(n_steps=2000)
```


```python
# Visualize network structure
fig, ax = plt.subplots(figsize=(8, 6))

_ = superlink.plot_network_2d(ax=ax, junction_kwargs={'s' : 0},
                              superjunction_kwargs={'c' : '0.25'},
                              link_kwargs={'color' : '0.5'},
                              weir_kwargs={'color' : 'r'})
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/validation/validation_0.png)


## Run model


```python
# Load forcing data
Q_in = pd.read_csv(f'{baseurl}/validation_flow_input_1.csv', index_col=0)

# Create "open" control signal for weirs
u = np.ones(len(weirs))
```


```python
# Set initial timestep
dt = 5
tol = 0.25
t_end = Q_in.index[-1] + 40000

# Create simulation context manager
with Simulation(superlink, Q_in=Q_in, dt=dt, min_dt=1,
                max_dt=30, t_end=t_end, interpolation_method='nearest') as simulation:
    coeffs = simulation.h0321
    # While simulation time has not expired...
    for step in simulation.steps:
        if simulation.t >= simulation.t_end:
            break
        # Advance model forward in time
        simulation.step(dt=dt, u_w=u, subdivisions=4)
        # Reposition junctions to capture backwater effects
        simulation.model.reposition_junctions()
        # Adjust step size
        dt = simulation.filter_step_size(tol=tol, coeffs=coeffs)
        # Record internal depth and flow states
        simulation.record_state()
        # Print progress bar
        simulation.print_progress()
```

    [==================================================] 100.0% [14.84 s]

## Get data from model and plot


```python
# Specify desired site names
sites = ['Pond 1', 'Pond 2', 'Pond 3', 'Outlet Flume']

# Convert hydraulic head states to water depth
h_j = (simulation.states.H_j - simulation.states.H_j.iloc[0])[sites]

# Convert index to epoch time
h_j.index = pd.to_datetime((simulation.states.H_j.index * 1e9).astype(int))
```


```python
# Plot depth states of desired sites
sns.set_palette('husl')
fig, ax = plt.subplots()

h_j.plot(ax=ax)
plt.title('Depth hydrographs')
plt.ylabel('Depth (m)')
plt.xlabel('Time')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/validation/validation_1.png)


## Compare against sensor data


```python
# Specify desired sensor sites
sensors = ['site_1', 'site_2', 'site_3', 'site_4']
sensor_data = {}

# Read sensor data into dataframes
for sensor in sensors:
    data = pd.read_csv(f'{baseurl}/depth_{sensor}.csv', index_col=0)
    data.index = pd.to_datetime(data.index)
    sensor_data[sensor] = data
```


```python
# Plot model output vs. sensor data
fig, ax = plt.subplots(2, 2)

t_start, t_end = h_j.index[0], h_j.index[-1] 

for index, site, sensor in zip(range(4), sites, sensors):
    sensor_data[sensor].loc[t_start:t_end].plot(ax=ax.flat[index], legend=False, color='0.7')
    h_j[site].plot(ax=ax.flat[index], label='simulated', color='r')
    ax.flat[index].set_title(site.title())
    if index < 2:
        ax.flat[index].xaxis.set_ticklabels([])
        ax.flat[index].set_xlabel('')   
    if not (index % 2):
        ax.flat[index].set_ylabel('Depth (m)')
    
plt.tight_layout()
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/validation/validation_2.png)

