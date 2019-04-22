# superlink
🚰 Implementation of the SUPERLINK hydraulic solver by [Ji (1998).](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%281998%29124%3A3%28307%29)

```
Ji, Z. (1998). General Hydrodynamic Model for Sewer/Channel Network Systems.
Journal of Hydraulic Engineering, 124(3), 307–315.
doi: 10.1061/(asce)0733-9429(1998)124:3(307)
```

## Example

### Diagram of test case

![Example network](https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/example_network_ji.png)

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

# Read input hydrographs and stage boundary conditions
H_bc = pd.read_csv('../data/boundary_stage.csv', index_col=0)
Q_in = pd.read_csv('../data/flow_input.csv', index_col=0)
time_range = H_bc.index.values
```

### Run superlink

```python
# Instantiate superlink object
superlink = SuperLink(superlinks, superjunctions, links, junctions)

# Create lists to store results
hs = []
Hs = []
Qs = []
ts = []
Q_ins = []
H_bcs = []

# Set initial time
t_prev = t_range[0]

# For each time step...
for t_next in time_range[1:]:
    # Compute time difference between steps
    dt = t_next - t_prev
    # Get next stage boundary condition
    H_bc_next = H_bc.loc[t_next].values
    # Get next flow input
    Q_in_next = Q_in.loc[t_next].values
    # Run superlink algorithm
    superlink.step(H_bc=H_bc_next, Q_in=Q_in_next, dt=dt)
    # Store previous timestamp
    t_prev = t_next
    # Store results
    ts.append(t_next)
    hs.append(np.copy(superlink._h_Ik))
    Hs.append(superlink.H_j)
    Qs.append((superlink._Q_uk + superlink._Q_dk) / 2)
    Q_ins.append(Q_in_next)
    H_bcs.append(H_bc_next)
```

![Superlink Example](https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/superlink_test.png)
