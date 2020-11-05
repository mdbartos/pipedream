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

internal_links = 24

hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                      hillslope_superjunctions,
                                      internal_links=internal_links)

hillslope_nsuperlink_model = nSuperLink(hillslope_superlinks,
                                       hillslope_superjunctions,
                                       internal_links=internal_links)

hillslope_greenampt_model = GreenAmpt(hillslope_soil_params)
hillslope_ngreenampt_model = nGreenAmpt(hillslope_soil_params)

hillslope_water_quality_model = QualityBuilder(hillslope_nsuperlink_model,
                                               superjunction_params=hillslope_superjunction_wq_params,
                                               superlink_params=hillslope_superlink_wq_params)

initial_nsuperlink_states = copy.deepcopy(hillslope_nsuperlink_model.states)

def test_superlink_step():
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                        hillslope_superjunctions,
                                        internal_links=4)
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_superlink_model.NIk)
    hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
    hillslope_superlink_model.reposition_junctions()

def test_nsuperlink_step():
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_nsuperlink_model.NIk)
    hillslope_nsuperlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
    hillslope_nsuperlink_model.reposition_junctions()

def test_superlink_spinup():
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                        hillslope_superjunctions,
                                        internal_links=4)
    hillslope_superlink_model.spinup(n_steps=100)

def test_superlink_convergence():
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                        hillslope_superjunctions,
                                        internal_links=4)
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_superlink_model.NIk)
    hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, num_iter=8)

def test_superlink_banded_step():
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                          hillslope_superjunctions,
                                          internal_links=4, auto_permute=True)
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_superlink_model.NIk)
    hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)

def test_superlink_recurrence_method():
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                          hillslope_superjunctions,
                                          internal_links=4, method='f')
    dt = 10
    Q_in = 1e-2 * np.asarray([1., 0.])
    Q_0Ik = 1e-3 * np.ones(hillslope_superlink_model.NIk)
    hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                          hillslope_superjunctions,
                                          internal_links=4, method='nnls')
    hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                          hillslope_superjunctions,
                                          internal_links=4, method='lsq')
    hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)

def test_simulation_manager():
    dt = 10
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                          hillslope_superjunctions,
                                          internal_links=24)
    Q_in = pd.DataFrame.from_dict(
        {
            0 :  np.zeros(hillslope_superlink_model.M),
            3600: np.zeros(hillslope_superlink_model.M),
            3601: 1e-3 * np.ones(hillslope_superlink_model.M),
            18000 : 1e-3 * np.ones(hillslope_superlink_model.M),
            18001 : np.zeros(hillslope_superlink_model.M),
            28000 : np.zeros(hillslope_superlink_model.M)
        }, orient='index')

    Q_Ik = pd.DataFrame.from_dict(
        {
            0 :  np.zeros(hillslope_superlink_model.NIk),
            3600: np.zeros(hillslope_superlink_model.NIk),
            3601: 1e-3 * np.ones(hillslope_superlink_model.NIk),
            18000 : 1e-3 * np.ones(hillslope_superlink_model.NIk),
            18001 : np.zeros(hillslope_superlink_model.NIk),
            28000 : np.zeros(hillslope_superlink_model.NIk)
        }, orient='index'
    )
    # Create simulation context manager
    with Simulation(hillslope_superlink_model, Q_in=Q_in, Q_Ik=Q_Ik) as simulation:
        # While simulation time has not expired...
        while simulation.t <= simulation.t_end:
            # Step hillslope_superlink_model forward in time
            simulation.step(dt=dt)
            # Record internal depth and flow states
            simulation.record_state()
            # Print progress bar
            simulation.print_progress()

def test_adaptive_timestep():
    dt = 10
    hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                          hillslope_superjunctions,
                                          internal_links=24)
    Q_in = pd.DataFrame.from_dict(
        {
            0 :  np.zeros(hillslope_superlink_model.M),
            3600: np.zeros(hillslope_superlink_model.M),
            3601: 1e-3 * np.ones(hillslope_superlink_model.M),
            18000 : 1e-3 * np.ones(hillslope_superlink_model.M),
            18001 : np.zeros(hillslope_superlink_model.M),
            28000 : np.zeros(hillslope_superlink_model.M)
        }, orient='index')

    Q_Ik = pd.DataFrame.from_dict(
        {
            0 :  np.zeros(hillslope_superlink_model.NIk),
            3600: np.zeros(hillslope_superlink_model.NIk),
            3601: 1e-3 * np.ones(hillslope_superlink_model.NIk),
            18000 : 1e-3 * np.ones(hillslope_superlink_model.NIk),
            18001 : np.zeros(hillslope_superlink_model.NIk),
            28000 : np.zeros(hillslope_superlink_model.NIk)
        }, orient='index'
    )
    # Create simulation context manager
    with Simulation(hillslope_superlink_model, Q_in=Q_in, Q_Ik=Q_Ik) as simulation:
        coeffs = simulation.h0321
        tol = 0.25
        # While simulation time has not expired...
        while simulation.t <= simulation.t_end:
            # Step hillslope_superlink_model forward in time
            simulation.step(dt=dt, subdivisions=2,
                            retries=10)
            # Record internal depth and flow states
            simulation.record_state()
            # Adjust step size
            dt = simulation.filter_step_size(tol=tol, coeffs=coeffs)
            # Print progress bar
            simulation.print_progress()

def test_superlink_geometry():
    geoms = ['circular', 'rect_closed', 'rect_open', 'triangular', 'trapezoidal', 'wide']
    hillslope_superlinks['g1'] = 1
    hillslope_superlinks['g2'] = 1
    hillslope_superlinks['g3'] = 1
    for geom in geoms:
        hillslope_superlinks['shape'] = geom
        hillslope_superlink_model = SuperLink(hillslope_superlinks,
                                            hillslope_superjunctions,
                                            internal_links=4)
        dt = 10
        Q_in = 1e-2 * np.asarray([1., 0.])
        Q_0Ik = 1e-3 * np.ones(hillslope_superlink_model.NIk)
        hillslope_superlink_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
    hillslope_superlinks['shape'] = 'rect_open'
    hillslope_superlinks['g1'] = 10
    hillslope_superlinks['g2'] = 5

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

def test_orifice():
    superjunctions = copy.deepcopy(hillslope_superjunctions)
    superlinks = hillslope_superlinks
    superjunctions.loc[2] = superjunctions.loc[0]
    superjunctions.loc[2, ['name', 'id']] = 2
    superjunctions.loc[2, 'h_0'] = 2.0
    orifices = {
        'id' : 0,
        'sj_0' : 2,
        'sj_1' : 0,
        'A' : 0.3048**2,
        'orientation' : 'side',
        'z_o' : 0,
        'y_max' : 0.3048,
        'C' : 0.67}
    orifices = pd.DataFrame(orifices, index=[0])
    hydraulic_model = SuperLink(superlinks, superjunctions, orifices=orifices,
                                internal_links=internal_links)
    dt = 10
    Q_in = 1e-2 * np.asarray([0., 0., 0.])
    Q_0Ik = 1e-3 * np.ones(hydraulic_model.NIk)
    for _ in range(100):
        if (hydraulic_model.t > 200):
            u_o = 0.5 * np.ones(1)
        else:
            u_o = np.zeros(1)
        hydraulic_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o)

def test_norifice():
    superjunctions = copy.deepcopy(hillslope_superjunctions)
    superlinks = hillslope_superlinks
    superjunctions.loc[2] = superjunctions.loc[0]
    superjunctions.loc[2, ['name', 'id']] = 2
    superjunctions.loc[2, 'h_0'] = 2.0
    orifices = {
        'id' : 0,
        'sj_0' : 2,
        'sj_1' : 0,
        'A' : 0.3048**2,
        'orientation' : 'side',
        'z_o' : 0,
        'y_max' : 0.3048,
        'C' : 0.67}
    orifices = pd.DataFrame(orifices, index=[0])
    hydraulic_model = nSuperLink(superlinks, superjunctions, orifices=orifices,
                                internal_links=internal_links)
    dt = 10
    Q_in = 1e-2 * np.asarray([0., 0., 0.])
    Q_0Ik = 1e-3 * np.ones(hydraulic_model.NIk)
    for _ in range(1000):
        if (hydraulic_model.t > 2000):
            u_o = 0.5 * np.ones(1)
        else:
            u_o = np.zeros(1)
        hydraulic_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, u_o=u_o)

def test_weir():
    superjunctions = copy.deepcopy(hillslope_superjunctions)
    superlinks = hillslope_superlinks
    superjunctions.loc[2] = superjunctions.loc[0]
    superjunctions.loc[2, ['name', 'id']] = 2
    superjunctions.loc[2, 'h_0'] = 2.0
    weirs = {
        'id' : 0,
        'sj_0' : 2,
        'sj_1' : 0,
        'z_w' : 0,
        'y_max' : 0.3048,
        'Cr' : 0.67,
        'Ct' : 0.67,
        'L' : 0.3048,
        's' : 0.01
    }
    weirs = pd.DataFrame(weirs, index=[0])
    hydraulic_model = SuperLink(superlinks, superjunctions, weirs=weirs,
                                internal_links=internal_links)
    dt = 10
    Q_in = 1e-2 * np.asarray([0., 0., 0.])
    Q_0Ik = 1e-3 * np.ones(hydraulic_model.NIk)
    for _ in range(100):
        if (hydraulic_model.t > 200):
            u_w = 0.5 * np.ones(1)
        else:
            u_w = np.zeros(1)
        hydraulic_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, u_w=u_w)

def test_nweir():
    superjunctions = copy.deepcopy(hillslope_superjunctions)
    superlinks = hillslope_superlinks
    superjunctions.loc[2] = superjunctions.loc[0]
    superjunctions.loc[2, ['name', 'id']] = 2
    superjunctions.loc[2, 'h_0'] = 2.0
    weirs = {
        'id' : 0,
        'sj_0' : 2,
        'sj_1' : 0,
        'z_w' : 0,
        'y_max' : 0.3048,
        'Cr' : 0.67,
        'Ct' : 0.67,
        'L' : 0.3048,
        's' : 0.01
    }
    weirs = pd.DataFrame(weirs, index=[0])
    hydraulic_model = nSuperLink(superlinks, superjunctions, weirs=weirs,
                                internal_links=internal_links)
    dt = 10
    Q_in = 1e-2 * np.asarray([0., 0., 0.])
    Q_0Ik = 1e-3 * np.ones(hydraulic_model.NIk)
    for _ in range(1000):
        if (hydraulic_model.t > 2000):
            u_w = 0.5 * np.ones(1)
        else:
            u_w = np.zeros(1)
        hydraulic_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, u_w=u_w)

def test_pump():
    superjunctions = copy.deepcopy(hillslope_superjunctions)
    superlinks = hillslope_superlinks
    superjunctions.loc[2] = superjunctions.loc[0]
    superjunctions.loc[2, ['name', 'id']] = 2
    superjunctions.loc[2, 'h_0'] = 2.0
    pumps = {
        'id' : 0,
        'sj_0' : 0,
        'sj_1' : 2,
        'z_p' : 0,
        'a_q' : 2.0,
        'a_h' : 0.1,
        'dH_min' : 0.5,
        'dH_max' : 2.0
    }
    pumps = pd.DataFrame(pumps, index=[0])
    hydraulic_model = SuperLink(superlinks, superjunctions, pumps=pumps,
                                internal_links=internal_links)
    dt = 10
    Q_in = 1e-2 * np.asarray([0., 0., 0.])
    Q_0Ik = 1e-3 * np.ones(hydraulic_model.NIk)
    for _ in range(100):
        if (hydraulic_model.t > 200):
            u_p = 0.5 * np.ones(1)
        else:
            u_p = np.zeros(1)
        hydraulic_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, u_p=u_p)

def test_npump():
    superjunctions = copy.deepcopy(hillslope_superjunctions)
    superlinks = hillslope_superlinks
    superjunctions.loc[2] = superjunctions.loc[0]
    superjunctions.loc[2, ['name', 'id']] = 2
    superjunctions.loc[2, 'h_0'] = 2.0
    pumps = {
        'id' : 0,
        'sj_0' : 0,
        'sj_1' : 2,
        'z_p' : 0,
        'a_q' : 2.0,
        'a_h' : 0.1,
        'dH_min' : 0.5,
        'dH_max' : 2.0
    }
    pumps = pd.DataFrame(pumps, index=[0])
    hydraulic_model = nSuperLink(superlinks, superjunctions, pumps=pumps,
                                 internal_links=internal_links)
    dt = 10
    Q_in = 1e-2 * np.asarray([0., 0., 0.])
    Q_0Ik = 1e-3 * np.ones(hydraulic_model.NIk)
    for _ in range(1000):
        if (hydraulic_model.t > 2000):
            u_p = 0.5 * np.ones(1)
        else:
            u_p = np.zeros(1)
        hydraulic_model.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik, u_p=u_p)

