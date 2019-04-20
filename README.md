# superlink
🚰 Implementation of the SUPERLINK hydraulic solver by [Ji (1998).](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%281998%29124%3A3%28307%29)

```
Ji, Z. (1998). General Hydrodynamic Model for Sewer/Channel Network Systems.
Journal of Hydraulic Engineering, 124(3), 307–315.
doi:10.1061/(asce)0733-9429(1998)124:3(307)
```

## Example

### Import modules and load data

```python
# Import modules
import json
import numpy as np
import pandas as pd
from superlink.superlink import SuperLink

# Load input data
with open('../data/six_pipes.json', 'r') as j:
    inp = json.load(j)  
superjunctions = pd.DataFrame(inp['superjunctions'])
superlinks = pd.DataFrame(inp['superlinks'])
junctions = pd.DataFrame(inp['junctions'])
links = pd.DataFrame(inp['links'])

# Define function to create triangular input hydrographs
def triangle(t, lm=1, rm=1, tc=0, yc=1):
    """
    Generate a triangular signal at time t.
    """
    lm = np.asarray(lm)
    rm = np.asarray(rm)
    tc = np.asarray(tc)
    yc = np.asarray(yc)
    if t < tc:
        return np.maximum(yc - (lm)*(-t + tc), 0)
    else:
        return np.maximum(yc - (rm)*(t - tc), 0)
```
### Specify parameters and boundary conditions

```python
# Instantiate superlink object
superlink = SuperLink(superlinks, superjunctions, links, junctions)

# Specify time step
dt = 30

# Specify superjunctions with boundary conditions
bc_junctions = np.asarray([0, 0, 0, 1, 0, 1], dtype=float)
# Specify rate of head rise at boundary junctions
dHdt_rise = (1 / 60 / 60) * bc_junctions
dHdt_fall = (2.5 / 240 / 60) * bc_junctions
# Specify time at which maximum boundary head is reached...
t_Hc = 2 * 3600
# ...and the maximum boundary head
H_j_0 = np.copy(superlink.H_j)
H_max = (dHdt_rise * 2 * 3600) + H_j_0

# Specify rate of inflow at input junctions
dQdt = np.array([0.5, 0, 0.3, 0, 0, 0]) / 60 / 60
# Specify time at which maximum inflow is reached...
t_Qc = 3 * 3600
# ...and the maximum inflow
Q_max = dQdt * 1 * 3600

# Initialize time at zero
t = 0
# End simulation at six hours
T = 6 * 3600
```

### Run superlink

```python
# Create lists to store results
hs = []
Hs = []
Qs = []
ts = []
Q_ins = []
H_bcs = []

# For each timestep...
for _ in range(T // dt):
    # Increment time
    t += dt
    # Specify stage boundary conditions at time t
    H_bc = triangle(t, lm=dHdt_rise, rm=dHdt_fall,
                    tc=t_Hc, yc=H_max)
    # Specify inflow at time t
    Q_0j = triangle(t, lm=dQdt, rm=dQdt, tc=t_Qc, yc=Q_max)
    # Run superlink algorithm
    superlink.step(H_bc=H_bc, Q_0j=Q_0j, dt=dt)
    # Store results
    ts.append(t)
    hs.append(np.copy(superlink._h_Ik))
    Hs.append(superlink.H_j)
    Qs.append((superlink._Q_uk + superlink._Q_dk) / 2)
    Q_ins.append(Q_0j)
    H_bcs.append(H_bc)
```

![Superlink Example](https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/superlink_test.png)
