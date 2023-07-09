---
layout: default
---

# pipedream <sub>🚰</sub> <sup>💭</sup>
[![Build Status](https://travis-ci.org/mdbartos/pipedream.svg?branch=master)](https://travis-ci.org/mdbartos/pipedream) [![Coverage Status](https://coveralls.io/repos/github/mdbartos/pipedream/badge.svg?branch=master)](https://coveralls.io/github/mdbartos/pipedream?branch=master) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

An interactive hydrodynamic solver for sewer/stormwater networks.

# A Minimal Example

<img src="https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/example_network_ji.png" width="700">

### Import modules and load data

```python
# Import modules
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation

# Specify data path
input_path = '../data/six_pipes'

# Get model components
superjunctions = pd.read_csv(f'{input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/superlinks.csv')
junctions = pd.read_csv(f'{input_path}/junctions.csv')
links = pd.read_csv(f'{input_path}/links.csv')

# Read input data
Q_in = pd.read_csv(f'{input_path}/flow_input.csv', index_col=0)
H_bc = pd.read_csv(f'{input_path}/boundary_stage.csv', index_col=0)
```

### Run superlink

```python
# Instantiate superlink model
superlink = SuperLink(superlinks, superjunctions, links, junctions)

# Set constant timestep (in seconds)
dt = 30

# Create simulation context manager
with Simulation(superlink, Q_in=Q_in, H_bc=H_bc) as simulation:
    # While simulation time has not expired...
    while simulation.t <= simulation.t_end:
        # Step model forward in time
        simulation.step(dt=dt, num_iter=1)
        # Record internal depth and flow states
        simulation.record_state()
        # Print progress bar
        simulation.print_progress()
```

> `[==================================================] 100.0% [0.82 s]`


### Plot results

See plotting code [here](https://github.com/mdbartos/pipedream/blob/master/notebooks/six_pipe_test.ipynb).

![Superlink Example](https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/superlink_test.png)

# Acknowledgments

Hydraulic solver based on the SUPERLINK scheme proposed by [Zhong Ji (1998).](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%281998%29124%3A3%28307%29)

```
Ji, Z. (1998). General Hydrodynamic Model for Sewer/Channel Network Systems.
Journal of Hydraulic Engineering, 124(3), 307–315.
doi: 10.1061/(asce)0733-9429(1998)124:3(307)
```
