import copy
import numpy as np
import pandas as pd
import pipedream_solver
from pipedream_solver.superlink import SuperLink
from pipedream_solver.nsuperlink import nSuperLink
from pipedream_solver.simulation import Simulation
from pipedream_solver.infiltration import GreenAmpt
from pipedream_solver.ninfiltration import nGreenAmpt
from pipedream_solver.nquality import QualityBuilder

hillslope_superjunctions = pd.read_csv('data/hillslope/hillslope_superjunctions.csv')
hillslope_superlinks = pd.read_csv('data/hillslope/hillslope_superlinks.csv')
hillslope_soil_params = pd.read_csv('data/hillslope/hillslope_soil_params.csv')
hillslope_superlink_wq_params = pd.read_csv('data/hillslope/hillslope_superlink_wq_params.csv')
hillslope_superjunction_wq_params = pd.read_csv('data/hillslope/hillslope_superjunction_wq_params.csv')
njunctions_fixed = 24

hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                      hillslope_superjunctions,
                                      njunctions_fixed=njunctions_fixed)

hillslope_nsuperlink_model = nSuperLink(hillslope_superlinks,
                                       hillslope_superjunctions,
                                       njunctions_fixed=njunctions_fixed)

hillslope_greenampt_model = GreenAmpt(hillslope_soil_params)
hillslope_ngreenampt_model = nGreenAmpt(hillslope_soil_params)

hillslope_water_quality_model = QualityBuilder(hillslope_nsuperlink_model,
                                               superjunction_params=hillslope_superjunction_wq_params,
                                               superlink_params=hillslope_superlink_wq_params)

initial_nsuperlink_states = copy.deepcopy(hillslope_nsuperlink_model.states)

def test_superlink_step():
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_superlink_model.NIk)
    hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)

def test_nsuperlink_step():
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_nsuperlink_model.NIk)
    hillslope_nsuperlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)

def test_plot_profile():
    hillslope_superlink_model.plot_profile([0, 1], width=100)
    hillslope_nsuperlink_model.plot_profile([0, 1], width=100)

def test_plot_network_2d():
    hillslope_superlink_model.plot_network_2d(junction_kwargs={'s' : 4})
    hillslope_nsuperlink_model.plot_network_2d(junction_kwargs={'s' : 4})

def test_greenampt_step():
    dt = 120
    i = 50 / 1000 / 3600 * np.ones(hillslope_superlink_model.NIk)
    infiltration_model = hillslope_greenampt_model
    for _ in range(100):
        infiltration_model.step(dt=dt, i=i)

def test_ngreenampt_step():
    dt = 10
    i = 50 / 1000 / 3600 * np.ones(hillslope_superlink_model.NIk)
    hydraulic_model = hillslope_nsuperlink_model
    infiltration_model = hillslope_ngreenampt_model
    hydraulic_model.load_state(initial_nsuperlink_states)
    for _ in range(1000):
        infiltration_model.d = hydraulic_model.h_Ik
        infiltration_model.step(dt=dt, i=i)
        Q_0Ik = infiltration_model.Q
        hydraulic_model.step(dt=dt, Q_0Ik=Q_0Ik)

def test_water_quality_step():
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_nsuperlink_model.NIk)
    c_0j = 10. * np.asarray([1., 0.])
    for _ in range(100):
        hillslope_nsuperlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
        hillslope_water_quality_model.step(dt=dt, c_0j=c_0j)

