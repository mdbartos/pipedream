import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn as sns
from pipedream_solver.simulation import Simulation

input_path = '../data/WQ_Kalman_Test_11superlinks'
superjunctions = pd.read_csv(f'{input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/superjunction_wq_params.csv')

internal_links = 20
superlink = SuperLink(superlinks, superjunctions, internal_links=internal_links, bc_method='b', min_depth=10e-3)
waterquality = QualityBuilder(superlink, 
                              superjunction_params=superjunction_wq_params,
                              superlink_params=superlink_wq_params)

Q_0Ik = 0*np.ones(superlink.NIk)

c_Ik = []
c_ik = []
c_j = []
Q_ik = []
Q_j_out = []

# Kalman Filter
x_hats = []
measure = pd.read_csv(f'{input_path}/measurement_data.csv')
measure = pd.DataFrame(measure).to_numpy()

# For checking the mass and volume balances
mass_in = 0.
mass_over_time = []
true_mass_over_time = []
volume_in = 0.0
volume_over_time = []
true_volume_over_time = []
volume_initial = ( (superlink.A_sj*(superlink.H_j-superlink.z_inv_j)).sum()
                 + (superlink.A_ik*superlink._dx_ik).sum()
                 + (superlink._A_SIk*superlink.h_Ik).sum())

# Timestep and time-series input data - application of the arbitrary Q & C
# In this test, only dt=1 can be applied because 'T' index variable is used for the input data.
dt = 1
T = 0
T_max = 10000
Q_in_time = np.zeros([2,T_max])
c_0j_in_time = np.zeros([2,T_max])
c_0j_in_time_noise = np.zeros([2,T_max])

# Making the arbitrary time-series input data
for t in range(T_max):
    if t < 500:
        Q_in_time[0,t] = 0.1
    elif t < 1000:
        Q_in_time[0,t] = 0.1 + 0.002*(t-500)
    elif t < 1500:
        Q_in_time[0,t] = 0.1 + 0.002*(1500-t)
    elif t< T_max:
        Q_in_time[0,t] = 0.1
        
for t in range(T_max):
    if t < 500:
        c_0j_in_time[0,t] = 0
    elif t < 1000:
        c_0j_in_time[0,t] = 15#+0.02*(t-500)
    elif t < 1500:
        c_0j_in_time[0,t] = 15#+0.02*(1500-t)
    elif t< T_max:
        c_0j_in_time[0,t] = 0

with Simulation(superlink, dt=dt, min_dt=1,
                max_dt=1, t_end=T_max, interpolation_method='linear') as simulation:
    coeffs = simulation.h0321
    
    Num_T = int(simulation.t_end/dt)+1
    Estimated_out = np.zeros(Num_T)
    Original_out = np.zeros(Num_T)
    Observed_out = np.zeros(Num_T)

    while superlink.t < T_max:
        c_0j = c_0j_in_time[0,T] * np.asarray([1,0.5,0,0,0,0,1,0.5,0,0])
        Q_in = Q_in_time[0,T] * np.asarray([1,1,0,0,0,0,1,1,0,0])
        
        # Advance hydraulic model
        simulation.step(dt=dt, Q_in=Q_in)
                
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
        
        # Check the mass and volume balances
        mass_in += dt * (c_0j * Q_in).sum()
        mass_in_sys = ((superlink._dx_ik * superlink._A_ik * waterquality.c_ik).sum()
                 + (waterquality.c_Ik * superlink.h_Ik * superlink._A_SIk).sum()
                 + (waterquality.c_j * superlink.A_sj*(superlink.H_j - superlink.z_inv_j)).sum() 
                 + (waterquality.c_dk * waterquality._A_dk_next * waterquality._dx_dk).sum()
                 + (waterquality.c_uk * waterquality._A_uk_next * waterquality._dx_uk).sum())
        mass_over_time.append(mass_in_sys)
        true_mass_over_time.append(mass_in)
        volume_in += dt * (Q_in[:]).sum()
        volume_in_sys = ( (superlink.A_sj*(superlink.H_j-superlink.z_inv_j)).sum()
                         + (superlink.A_ik*superlink._dx_ik).sum()
                         + (superlink._A_SIk*superlink.h_Ik).sum()
                         - volume_initial)
        volume_over_time.append(volume_in_sys)
        true_volume_over_time.append(volume_in)
        
        # Record nodal contaminant concentrations
        c_Ik.append(waterquality.c_Ik.copy())
        c_ik.append(waterquality.c_ik.copy())   
        c_j.append(waterquality.c_j.copy())
        Q_ik.append(waterquality._Q_ik_next.copy())
       
        simulation.print_progress()
        
        T += 1
    
# Plot the results
fig, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2, figsize=(12,12))
plt.rc('font', size=12)
sns.set_palette('viridis', internal_links + 1)
_ = ax1.plot(c_Ik)
ax1.set_ylabel('Concentration $(g/m^3)$')
ax1.set_xlabel('Time (s)')
ax1.set_ylim(0,20)
ax1.set_title("Concentrations of internal junctions")
ax2.plot(c_ik)
ax2.set_ylabel('Concentration $(g/m^3)$')
ax2.set_xlabel('Time (s)')
ax2.set_ylim(0,20)
ax2.set_title("Concentrations of internal Links")
ax3.plot(true_volume_over_time, c='k', label='True volume input')
ax3.plot(volume_over_time, c='r', linestyle='--', label='volume in system')
ax3.set_ylabel('Total volume of water')
ax3.set_xlabel('Time (s)')
ax3.set_title("Volume balance")
ax3.legend(loc=4)
ax4.plot(true_mass_over_time, c='k', label='True mass input')
ax4.plot(mass_over_time, c='r', linestyle='--', label='Mass in system')
ax4.set_ylabel('Total mass')
ax4.set_xlabel('Time (s)')
ax4.set_title("Mass balance")
ax4.legend(loc=4)
ax5.plot(Q_ik)
ax5.set_ylabel('Flow rate $(m^3/sec)$')
ax5.set_xlabel('Time (s)')
ax5.set_title("Flow rates of internal links")
ax6.plot(c_0j_in_time[0,:])
ax6.set_ylabel('Concentration $(g/m^3)$')
ax6.set_xlabel('Time (s)')
ax6.set_title("Input concentration")
fig.tight_layout()
plt.show()

percent_mass = 100*(mass_in_sys/mass_in)
percent_volume = 100*(volume_in_sys/volume_in)
print("\nMass(%, in system/input)  = " , percent_mass)
print("Volume(%, in system/input)  = " , percent_volume)

plt.figure(1)
sns.set_palette('viridis', 2)
_ = plt.plot(x_hats)
plt.ylabel('Concentration $(g/m^3)$')
plt.xlabel('Time (s)')
plt.title('Estimated concentration time series at superjunctions - x_hat')
plt.ylim(0,20)

plt.figure(2)
plt.figure(figsize= (12,6))
plt.plot(Observed_out,'-', label='Measurement(process+measurement noise)', color='r')
plt.plot(Original_out,'-', label='Model output(Original model+process noise)', color='y')
plt.plot(Estimated_out,'--', label='Estimation by Kalman filter', color='b')
plt.xlabel("Time(t)", fontsize = 14)
plt.ylabel("Concentration", fontsize = 14)
plt.ylim(0, 20)
plt.legend(fontsize = 14)