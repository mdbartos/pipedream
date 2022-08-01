import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation
from pipedream_solver.nquality import QualityBuilder

import matplotlib.pyplot as plt

# Provide base directory where data is stored
input_path = '../data/Simple_Test_3_Open'

# Get model components
superjunctions = pd.read_csv(f'{input_path}/Model_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/Model_superlinks.csv')
junctions = pd.read_csv(f'{input_path}/Model_junctions.csv')
links = pd.read_csv(f'{input_path}/Model_links.csv')

# Load forcing data
Q_in = pd.read_csv(f'{input_path}/Model_flow_input.csv', index_col=0)

# Instantiate superlink object
superlink = SuperLink(superlinks, superjunctions, links, junctions)

Vel_ik = []
Depth_ik = []
A_ik = []

# Set initial timestep
dt = 10
tol = 0.25
t_end = Q_in.index[-1]

T=0
# Create simulation context manager
with Simulation(superlink, Q_in=Q_in, dt=dt, min_dt=10,
                max_dt = 10, t_end=t_end, interpolation_method='nearest') as simulation:
    coeffs = simulation.h0321
    # While simulation time has not expired...
    for step in simulation.steps:
        if simulation.t >= simulation.t_end:
            break
        # Advance model forward in time
        simulation.step(dt=dt, subdivisions=4)
                
        Vel_ik.append(superlink.Q_ik[:].copy()/superlink.A_ik[:].copy())
        Depth_ik.append(superlink.A_ik[:].copy()/superlink.B_ik[:].copy())
        A_ik.append(superlink.A_ik[:].copy())
        
        # Adjust step size
        dt = simulation.filter_step_size(tol=tol, coeffs=coeffs)
        # Record internal depth and flow states
        simulation.record_state()
        # Print progress bar
        simulation.print_progress()

# Convert hydraulic head states to water depth
h_j = (simulation.states.H_j - superlink.z_inv_j)
Q_ik = simulation.states.Q_ik

plt.figure(1)
plt.figure(figsize= (10,6))
plt.plot(h_j[['J2']], label = 'J2', color = 'r')
plt.plot(h_j[['J3']], label = 'J3', color = 'g')
plt.plot(h_j[['J4']], label = 'J4', color = 'b')
plt.ylabel('Depth(head) of the super junctions', fontsize = 15)
plt.xlabel('Time (s)', fontsize = 15)
plt.ylim(0,10)
plt.xlim(0,80000)
plt.legend(fontsize = 15)
plt.grid()

plt.figure(2)
plt.figure(figsize= (10,6))
plt.plot(Q_ik[:][6], label = 'ik = 6', color = 'r')
plt.plot(Q_ik[:][11], label = 'ik = 11', color = 'g')
plt.plot(Q_ik[:][16], label = '', color = 'b')
plt.ylabel('Flow rate of the internal links', fontsize = 15)
plt.xlabel('Time (s)', fontsize = 15)
plt.ylim(0,20)
plt.xlim(0,80000)
plt.legend(fontsize = 15)
plt.grid()

Vel_ik = pd.DataFrame(Vel_ik)
plt.figure(3)
plt.figure(figsize= (10,6))
plt.plot(Vel_ik[:][6], label = 'ik = 6', color = 'r')
plt.plot(Vel_ik[:][11], label = 'ik = 11', color = 'g')
plt.plot(Vel_ik[:][16], label = 'ik = 16', color = 'b')
plt.ylabel('Velocity in the internal links', fontsize = 15)
plt.xlabel('Time (10s)', fontsize = 15)
plt.ylim(0,10)
plt.xlim(0,8000)
plt.legend(fontsize = 15)
plt.grid()

A_ik = pd.DataFrame(A_ik)
plt.figure(4)
plt.figure(figsize= (10,6))
plt.plot(A_ik[:][6], label = 'ik = 6', color = 'r')
plt.plot(A_ik[:][11], label = 'ik = 11', color = 'g')
plt.plot(A_ik[:][16], label = 'ik = 16', color = 'b')
plt.ylabel('Cross-sectional area in the internal links', fontsize = 15)
plt.xlabel('Time (10s)', fontsize = 15)
plt.ylim(0,5)
plt.xlim(0,8000)
plt.legend(fontsize = 15)
plt.grid()
