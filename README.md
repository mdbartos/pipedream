# superlink
🚰 Implementation of the SUPERLINK hydraulic solver by [Ji (1998).](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%281998%29124%3A3%28307%29)

```
Ji, Z. (1998). General Hydrodynamic Model for Sewer/Channel Network Systems.
Journal of Hydraulic Engineering, 124(3), 307–315.
doi:10.1061/(asce)0733-9429(1998)124:3(307)
```

## Example

```python
import json
import pandas as pd

# Read input file
with open('../data/six_pipes.json', 'r') as j:
    inp = json.load(j)
superjunctions = pd.DataFrame(inp['superjunctions'])
superlinks = pd.DataFrame(inp['superlinks'])
junctions = pd.DataFrame(inp['junctions'])
links = pd.DataFrame(inp['links'])

# Add initial conditions
superjunctions['h_0'] += 1e-5
junctions['h_0'] += 1e-5
links['Q_0'] += 0

# Instantiate superlink
superlink = SuperLink(superlinks, superjunctions, links, junctions, dt=1e-6)

# Run superlink with boundary conditions and inputs (see notebook in test folder)...
```

![Superlink Example](https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/superlink_test.png)
