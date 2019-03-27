import numpy as np
import pandas as pd
from superlink.superlink import SuperLink
from data import *

superjunctions = pd.DataFrame.from_dict(superjunctions, orient='index')
superlinks = pd.DataFrame.from_dict(superlinks, orient='index')
links = pd.concat({k: pd.DataFrame.from_dict(v, orient='index')
                   for k,v in links.items()}, axis=0)
junctions = pd.concat({k: pd.DataFrame.from_dict(v, orient='index')
                       for k,v in junctions.items()}, axis=0)

links = links.reset_index().set_index('level_1')
junctions = junctions.reset_index().set_index('level_1')

superjunctions['h_0'] += 1e-5
junctions['h_0'] += 1e-5
links['Q_0'] += 1e-10
superjunctions['A_sj'] = 1e-5
junctions['A_s'] = 1e-5

superlink = SuperLink(superlinks, superjunctions, links, junctions, dt=1e-6)

t = 0
dt = 1e-3
dHdt = 1 / 60 / 60
bc = superlink.bc.astype(float)
H_bc = superlink.H_j
data = {}
for _ in range(10):
    t += dt
    H_bc = H_bc + bc*dHdt*dt
    superlink.step(H_bc=H_bc, _dt=dt)
    data[t] = superlink.H_j

dt = 1.
for _ in range(20):
    t += dt
    H_bc = H_bc + bc*dHdt*dt
    superlink.step(H_bc=H_bc, _dt=dt)
    data[t] = superlink.H_j

H_bc = H_bc + bc*dHdt*dt
superlink.step(H_bc=H_bc, _dt=dt)
data[t] = superlink.H_j

