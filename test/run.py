import numpy as np
import pandas as pd
from superlink import SuperLink
from data import *

superjunctions = pd.DataFrame.from_dict(superjunctions, orient='index')
superlinks = pd.DataFrame.from_dict(superlinks, orient='index')
links = pd.concat({k: pd.DataFrame.from_dict(v, orient='index')
                   for k,v in links.items()}, axis=0)
junctions = pd.concat({k: pd.DataFrame.from_dict(v, orient='index')
                       for k,v in junctions.items()}, axis=0)

links = links.reset_index().set_index('level_1')
junctions = junctions.reset_index().set_index('level_1')

superjunctions['h_0'] += 1e-1
junctions['h_0'] += 1e-1
links['Q_0'] += 1e-10
superjunctions['A_sj'] = 1e-5
junctions['A_s'] = 1e-5

superlink = SuperLink(superlinks, superjunctions, links, junctions, dt=1e-6)
superlink.step(_dt=1e-6)
