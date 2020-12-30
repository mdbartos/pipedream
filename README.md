# pipedream <sub>🚰</sub> <sup>💭</sup>
[![Build Status](https://travis-ci.org/mdbartos/pipedream.svg?branch=master)](https://travis-ci.org/mdbartos/pipedream) [![Coverage Status](https://coveralls.io/repos/github/mdbartos/pipedream/badge.svg?branch=master)](https://coveralls.io/github/mdbartos/pipedream?branch=master) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Docs](https://img.shields.io/badge/docs-read%20here-ff69b4)](https://mdbartos.github.io/pipedream/) [![Paper](https://img.shields.io/badge/EarthArXiv-10.31223/osf.io/d8ca6-orange)](https://eartharxiv.org/d8ca6)

An interactive hydrodynamic solver for sewer/stormwater networks.

## About

*Pipedream* is a physically-based sewer/stormwater model designed for real-time applications. The *pipedream* toolkit consists of four major components:

- A *hydraulic solver* based on the 1D Saint-Venant equations.
- An *infiltration solver* based on the Green-Ampt formulation.
- A *water quality solver* based on the 1D advection-reaction-diffusion equation (experimental).
- An *interactive simulation manager* that facilitates real-time data assimilation and control.

Example use-cases for *pipedream* include:
- Real-time detection and forecasting of urban flooding.
- Implementation of real-time control strategies for combined sewer overflows.
- Stormwater asset management and detection of maintenance emergencies.
- Data-driven water quality assessment.

## Documentation

- Read the docs [here 📖](https://mdbartos.github.io/pipedream/).

- Read the paper [here 📄](https://eartharxiv.org/d8ca6).

## Installation

Use `pip` to install `pipedream-solver` via pypi:

```shell
$ pip install pipedream-solver
```

Currently, only Python 3 is supported.

### Dependencies

The following dependencies are required to install and use the *pipedream* toolkit:

- [numpy](http://www.numpy.org/) (>= 1.18)
- [pandas](https://pandas.pydata.org/) (>= 0.25)
- [scipy](https://www.scipy.org/) (>= 1.5)
- [numba](https://numba.pydata.org/) (>= 0.40)
- [matplotlib](https://matplotlib.org/) (>= 3.0)

Listed version numbers have been tested and are known to work (this does not necessarily preclude older versions).

## A Minimal Example

<img src="https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/example_network_ji.png" width="700">

### Import modules and load data

```python
# Import modules
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation

# Specify data path
input_path = './data/six_pipes'

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
        simulation.step(dt=dt)
        # Record internal depth and flow states
        simulation.record_state()
        # Print progress bar
        simulation.print_progress()
```

> `[==================================================] 100.0% [0.82 s]`


### Plot results

See plotting code [here](https://github.com/mdbartos/superlink/blob/master/test/six_pipe_test.ipynb).

![Superlink Example](https://s3.us-east-2.amazonaws.com/mdbartos-img/superlink/superlink_test.png)

## Acknowledgments

Hydraulic solver based on the SUPERLINK scheme proposed by [Zhong Ji (1998).](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%281998%29124%3A3%28307%29)

```
Ji, Z. (1998). General Hydrodynamic Model for Sewer/Channel Network Systems.
Journal of Hydraulic Engineering, 124(3), 307–315.
doi: 10.1061/(asce)0733-9429(1998)124:3(307)
```
