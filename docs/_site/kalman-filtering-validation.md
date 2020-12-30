# Kalman Filtering Holdout Analysis

## Import modules

```python
import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation
from pipedream_solver.nutils import interpolate_sample

import matplotlib.pyplot as plt
import seaborn as sns
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

## Run model without Kalman filtering


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

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/kalman-filtering-validation/kalman-filtering-validation-0.png)

### Run model

```python
# Load forcing data
Q_in = pd.read_csv(f'{baseurl}/validation_flow_input_2.csv', index_col=0)

# Create "open" control signal for weirs
u = np.ones(len(weirs))
```


```python
# Set initial, end, minimum and maximum timesteps
dt = 5
min_dt = 3
max_dt = 10
t_end = Q_in.index[-1] + 40000

# Create simulation context manager
with Simulation(superlink, Q_in=Q_in, dt=dt, min_dt=1,
                max_dt=30, t_end=t_end, interpolation_method='nearest') as simulation:
    # While simulation time has not expired...
    for step in simulation.steps:
        if simulation.t >= simulation.t_end:
            break
        # Advance model forward in time
        simulation.step(dt=dt, u_w=u, subdivisions=1)
        simulation.model.superlink_inverse_courant()
        # Reposition junctions to capture backwater effects
        simulation.model.reposition_junctions()
        # Adjust step size
        dt = min(max(superlink._dt_ck.min(), min_dt), max_dt)
        # Record internal depth and flow states
        simulation.record_state()
        # Print progress bar
        simulation.print_progress()
```

    [==================================================] 100.0% [16.33 s]

### Get data from model and plot


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
fig, ax = plt.subplots(figsize=(14,7))

h_j.plot(ax=ax)
plt.title('Depth hydrographs (simulated)', size=14)
plt.ylabel('Depth (m)')
plt.xlabel('Time')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/kalman-filtering-validation/kalman-filtering-validation-1.png)


## Run simulation with Kalman filtering


```python
# Instantiate superlink object
superlink = SuperLink(superlinks, superjunctions, weirs=weirs,
                      auto_permute=False, internal_links=4, min_depth=0.0)

# Spin-up model to avoid running dry
superlink.reposition_junctions()
superlink.spinup(n_steps=2000)
```

### Read sensor data

```python
# Specify desired sensor sites
sensors = ['site_1', 'site_2', 'site_3', 'site_4']
sensor_data = {}

# Read sensor data into dataframes
for sensor in sensors:
    data = pd.read_csv(f'{baseurl}/depth_{sensor}.csv', index_col=0)
    data.columns = ['sensor']
    data.index = pd.to_datetime(data.index)
    sensor_data[sensor] = data
```

```python
# Format measurement data for assimilation
site_indices = superlink.superjunctions.set_index('name').loc[['Pond 1', 'Pond 3'], 'id'].values.tolist()
measurements = (pd.concat([sensor_data[sensor] for sensor in ['site_1', 'site_3']], axis=1)
                .interpolate().dropna()) + superlink.H_j[site_indices]
measurements.index = measurements.index.astype(int) / 1e9
```

### Run simulation and fuse sensor data with Kalman filter

```python
# Set up Kalman filtering parameters
n = superlink.M
p = n
m = 2

process_std_dev = 1e-1
measurement_std_dev = 8e-3

H = np.zeros((m, n))
H[[0,1], site_indices] = 1.
Qcov = (process_std_dev**2)*np.eye(p)
Rcov = (measurement_std_dev**2)*np.eye(m)

C = np.zeros((n, p))
C[np.arange(n), np.arange(p)] = 1.
```

```python
# Set initial, end, minimum and maximum timesteps
dt = 5
min_dt = 3
max_dt = 10
t_end = Q_in.index[-1] + 40000

# Create simulation context manager
with Simulation(superlink, Q_in=Q_in, Qcov=Qcov, Rcov=Rcov,
                C=C, H=H, t_end=t_end,
                interpolation_method='nearest') as filtered_simulation:
    # While simulation time has not expired...
    for step in filtered_simulation.steps:
        if filtered_simulation.t >= filtered_simulation.t_end:
            break
        # Step model forward in time
        filtered_simulation.step(dt=dt, u_w=u, subdivisions=1)
        filtered_simulation.model.superlink_inverse_courant()
        # Get "measured" value
        next_measurement = interpolate_sample(filtered_simulation.t,
                                              measurements.index.values,
                                              measurements.values)
        # Apply Kalman filter with measured value
        filtered_simulation.kalman_filter(next_measurement, u_w=u, dt=dt)
        # Reposition junctions to capture backwater effects
        filtered_simulation.model.reposition_junctions()
        # Adjust step size using inverse courant number
        dt = min(max(superlink._dt_ck.min(), min_dt), max_dt)
        # Record internal depth and flow states
        filtered_simulation.record_state()
        # Print progress bar
        filtered_simulation.print_progress()
```

### Get data from Kalman filtered model and plot

```python
# Convert hydraulic head states to water depth
h_j_filtered = (filtered_simulation.states.H_j - filtered_simulation.states.H_j.iloc[0])[sites]

# Convert index to epoch time
h_j_filtered.index = pd.to_datetime((filtered_simulation.states.H_j.index * 1e9).astype(int))
```

```python
# Plot depth states of desired sites
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(14,7))

h_j_filtered.plot(ax=ax)
plt.title('Depth hydrographs (filtered)', size=14)
plt.ylabel('Depth (m)')
plt.xlabel('Time')
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/kalman-filtering-validation/kalman-filtering-validation-2.png)

### Compare simulation, Kalman filtering, and sensor data

```python
# Plot model output vs. sensor data
fig, ax = plt.subplots(2, 2, figsize=(14, 7))

t_start, t_end = h_j.index[0], h_j.index[-1] 

for index, site, sensor in zip(range(4), sites, sensors):
    sensor_data[sensor].loc[t_start:t_end].plot(ax=ax.flat[index], legend=False, color='0.7')
    h_j[site].plot(ax=ax.flat[index], label='simulated', color='k', linestyle='--')
    h_j_filtered[site].plot(ax=ax.flat[index], label='filtered', color='r')
    ax.flat[index].set_title(site.title())
    if index < 2:
        ax.flat[index].xaxis.set_ticklabels([])
        ax.flat[index].set_xlabel('')   
    if not (index % 2):
        ax.flat[index].set_ylabel('Depth (m)')
    
plt.tight_layout()
plt.legend(fontsize=12)
ax.flat[0].set_ylim(-0.015, 0.28)
ax.flat[1].set_ylim(-0.015, 0.28)
ax.flat[2].set_ylim(-0.015, 0.28)
ax.flat[3].set_ylim(-0.015, 0.28)
```

![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/kalman-filtering-validation/kalman-filtering-validation-3.png)
