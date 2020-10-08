# Adaptive step size control

## Import modules


```python
import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation

import matplotlib.pyplot as plt
import seaborn as sns
```

## Read input data


```python
# Specify data path
model_input_path = '../data/adaptive_step'
data_input_path = '../data/six_pipes'

# Get model components
superjunctions = pd.read_csv(f'{model_input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{model_input_path}/superlinks.csv')

# Read input data
Q_in = pd.read_csv(f'{data_input_path}/flow_input.csv', index_col=0)
H_bc = pd.read_csv(f'{data_input_path}/boundary_stage.csv', index_col=0)
```

## Configure model


```python
# Set number of internal links
internal_links = 24
# Set surface area of internal junctions
superlinks['A_s'] = 1.16 / internal_links
```

## Instantiate hydraulic model


```python
# Instantiate hydraulic model
superlink = SuperLink(superlinks, superjunctions, min_depth=0.0, 
                      internal_links=internal_links, auto_permute=True)
# Spin up model to avoid running dry
superlink.spinup(n_steps=1000)
```

## Run hydraulic model with adaptive time step


```python
# Set initial timestep
dt = 60
# Create lists to store error and timestep
errs = {}
dts = {}

# Create simulation context manager
with Simulation(superlink, Q_in=Q_in, H_bc=H_bc) as simulation:
    coeffs = simulation.h0321
    tol = 0.25
    # While simulation time has not expired...
    for step in simulation.steps:
        if simulation.t >= simulation.t_end:
            break
        # Step model forward in time
        simulation.step(dt=dt, subdivisions=2,
                        retries=10, norm=-1)
        # Get truncation error
        err = simulation.err
        # Record truncation error and time step size
        errs[simulation.t] = err
        dts[simulation.t] = dt
        # Record internal depth and flow states
        simulation.record_state()
        # Reposition junctions for backwater effects
        simulation.model.reposition_junctions()
        # Adjust step size
        dt = simulation.filter_step_size(tol=tol, coeffs=coeffs)
        # Print progress bar
        simulation.print_progress()
        
# Convert error and timesteps to dataframes
errs = pd.DataFrame.from_dict(errs, orient='index')
dts = pd.DataFrame.from_dict(dts, orient='index')
```

    [==================================================] 100.0% [0.88 s]

## Plot results


```python
# Instantiate plot
fig, ax = plt.subplots(3, figsize=(10, 12))

# Compute average discharge in superlinks
simulation.states.Q_k = (simulation.states.Q_uk + simulation.states.Q_dk) / 2

# Plot results
simulation.states.Q_k.plot(ax=ax[0], title='Superlink discharge (cms)')
errs.plot(ax=ax[1], color='r', marker='o', legend=False)
dts.plot(ax=ax[2], color='k', marker='o', legend=False)
ax[1].set_title('Scaled error (-)')
ax[2].set_title('Time step size (s)')

# Configure plots
ax[0].set_ylabel('Input discharge (cms)')
ax[1].set_ylabel('Error (m)')
ax[2].set_ylabel('Time step size (s)')
ax[2].set_xlabel('Time (s)')
ax[0].title.set_size(14)
ax[1].title.set_size(14)
ax[2].title.set_size(14)
ax[0].xaxis.set_ticklabels([])
ax[1].xaxis.set_ticklabels([])
plt.tight_layout()
```


![png](https://pipedream-solver.s3.us-east-2.amazonaws.com/img/adaptive-stepsize/adaptive-stepsize-0.png)

