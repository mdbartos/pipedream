# Import modules
import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation
from pipedream_solver.nquality import QualityBuilder
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn as sns

# Specify data path
input_path = '../data/WQ_Kalman_Test_Six_pipes'

# Get model components
superjunctions = pd.read_csv(f'{input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/superlinks.csv')
junctions = pd.read_csv(f'{input_path}/junctions.csv')
links = pd.read_csv(f'{input_path}/links.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/superjunction_wq_params.csv')

# Read input data
Q_in = pd.read_csv(f'{input_path}/flow_input.csv', index_col=0)
H_bc = pd.read_csv(f'{input_path}/boundary_stage.csv', index_col=0)

# Instantiate superlink model
superlink = SuperLink(superlinks, superjunctions, links, junctions, bc_method = 'b', min_depth=10e-3)
waterquality = QualityBuilder(superlink, 
                              superjunction_params=superjunction_wq_params,
                              superlink_params=superlink_wq_params)

# Set the time step, dt, from the file input
dt = 5
Q_0Ik = 0*np.ones(superlink.NIk)    
c_Ik = []
c_ik = []
c_j = []

mass_in = 0.
mass_over_time = []
true_mass_over_time = []
self = waterquality

c_Ik = []
c_ik = []
c_dk = []
c_j = []

# Variables for Kalman filter
x_hats = []
measure = pd.read_csv(f'{input_path}/measurement.csv')
measure = pd.DataFrame(measure).to_numpy()

T = 0
with Simulation(superlink, Q_in=Q_in, H_bc=H_bc) as simulation:
    
    Num_T = int(simulation.t_end/dt)+1
    Estimated_out = np.zeros(Num_T)
    Original_out = np.zeros(Num_T)
    Observed_out = np.zeros(Num_T)
    
    # While simulation time has not expired...
    while simulation.t <= simulation.t_end:
        # If time is between 7200 and 14400 sec...
        if (superlink.t > 7200) and (superlink.t < 14400):
            # Apply contaminant to uphill superjunction
            c_0j = 12. * np.asarray([1,0,1,0,0,0])
        else:
            c_0j = np.zeros(6)
    
        # Step model forward in time
        simulation.step(dt=dt)
        # Record internal depth and flow states
        simulation.record_state()
        # Advance water quality model 
        waterquality.step(dt=dt, c_0j=c_0j)
        
        # Applying the Kalman filter
        '''
        Next parameters can be set in 'superlink_WQ_paras.py'.
         : Observation point(# of superjunction), σ_measurement and σ_process noise
        Options for Kalman filter
         - KF = 1 : Making an artificial measurement data with the random noise.
         - KF = 2 : Data Assimilation with the input measurement data
         - Other numbers = No Kalman filter application
        '''
        KF = 2
        if KF == 1:
            waterquality.Kalman_Filter_WQ1(_dt=dt)
            waterquality.solve_boundary_states()
            waterquality.solve_internals_backwards()
            Observed_out[T] = waterquality.Z
        elif KF == 2:
            waterquality.Kalman_Filter_WQ2(measure=measure[T,1], _dt=dt)
            waterquality.solve_boundary_states()
            waterquality.solve_internals_backwards()
            Observed_out[T] = measure[T,1]
        else:
            pass
        x_hats.append(waterquality.x_hat.copy())
        Original_out[T] = waterquality.c_j[waterquality.Observation_point]
        Estimated_out[T] = waterquality.x_hat[waterquality.Observation_point]
        
        # Check the mass balance
        mass_in += dt * (c_0j * Q_in).sum()
        mass_in_sys = ((superlink._dx_ik * superlink._A_ik * waterquality.c_ik).sum()
                 + (waterquality.c_Ik * superlink.h_Ik * superlink._A_SIk).sum()
                 + (waterquality.c_j * superlink.A_sj*(superlink.H_j - superlink.z_inv_j)).sum() 
                 + (waterquality.c_dk * waterquality._A_dk_next * waterquality._dx_dk).sum()
                 + (waterquality.c_uk * waterquality._A_uk_next * waterquality._dx_uk).sum())
        mass_over_time.append(mass_in_sys)
        true_mass_over_time.append(mass_in)
        
        # Record nodal contaminant concentrations
        c_Ik.append(waterquality.c_Ik.copy())
        c_ik.append(waterquality.c_ik.copy())
        c_j.append(waterquality.c_j.copy())
    
        # save the result of estimate at observation point
        Original_out[T] = waterquality.c_j[waterquality.Observation_point]
        Estimated_out[T] = waterquality.x_hat[waterquality.Observation_point]        
        if KF == 1:
            Observed_out[T] = waterquality.Z
        elif KF ==2:
            Observed_out[T] = measure[T,1]
        else:
            pass
        x_hats.append(waterquality.x_hat.copy())

        # add number of time step for saving the results.(This should be modified.)
        T += 1
        simulation.print_progress()

# Stack results into single array
x_hats = np.vstack(x_hats)

plt.figure(1)    
sns.set_palette('viridis')
_ = plt.plot(c_Ik)
plt.ylabel('Concentration $(g/m^3)$')
plt.xlabel('Time (s)')
plt.title('Concentration time series at internal junction')
plt.ylim(0,20)

plt.figure(2)    
sns.set_palette('viridis')
_ = plt.plot(c_ik)
plt.ylabel('Concentration $(g/m^3)$')
plt.xlabel('Time (s)')
plt.title('Concentration time series at internal link')
plt.ylim(0,20)

plt.figure(3)    
sns.set_palette('viridis', 2)
_ = plt.plot(c_j)
plt.ylabel('Concentration $(g/m^3)$')
plt.xlabel('Time (s)')
plt.title('Concentration time series at superjunctions')
plt.ylim(0,20)

plt.figure(6)    
sns.set_palette('viridis', 2)
_ = plt.plot(x_hats)
plt.ylabel('Concentration $(g/m^3)$')
plt.xlabel('Time (s)')
plt.title('Estimated concentrations at superjunctions - x_hat')
plt.ylim(0,20)


plt.figure(7)
plt.figure(figsize= (12,6))
plt.plot(Observed_out,'-', label='Measurement(process+measurement noise)', color='r')
#plt.plot(Z,'-', label='Measurement(Original model+measurement noise)', color='r')
plt.plot(Original_out,'-', label='Model output(Original model+process noise)', color='y')
plt.plot(Estimated_out,'--', label='Estimation by Kalman filter', color='b')
plt.xlabel("Time(t)", fontsize = 14)
plt.ylabel("Concentration", fontsize = 14)
plt.ylim(0, 12)
plt.legend(fontsize = 14)