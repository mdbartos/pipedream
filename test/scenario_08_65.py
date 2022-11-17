import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn as sns
from pipedream_solver.simulation import Simulation

input_path = '../data/scenario_08_65'
superjunctions = pd.read_csv(f'{input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/superjunction_wq_params.csv')
orifices = pd.read_csv(f'{input_path}/orifices.csv')

# Load comparison data
Data_swmm_depth = pd.read_csv(f'{input_path}/N_comparison(5sec)/Data_swmm_depth.csv')
Data_swmm_flow = pd.read_csv(f'{input_path}/N_comparison(5sec)/Data_swmm_flow.csv')
Data_swmm_tss = pd.read_csv(f'{input_path}/N_comparison(5sec)/Data_swmm_tss.csv')

dt = 5
internal_links = 70
superlink = SuperLink(superlinks, superjunctions,orifices=orifices,
                      internal_links=internal_links, bc_method = 'b', min_depth=10e-4, mobile_elements=False)
waterquality = QualityBuilder(superlink, 
                              superjunction_params=superjunction_wq_params,
                              superlink_params=superlink_wq_params)

# Visualize network structure
fig, ax = plt.subplots(figsize=(12, 9))
_ = superlink.plot_network_2d(ax=ax, junction_kwargs={'s' : 5},
                              superjunction_kwargs={'c' : '0.25'},
                              link_kwargs={'color' : '0.5'},
                              orifice_kwargs={'color' : 'r'})

Q_0Ik = 0*np.ones(superlink.NIk)

c_Ik = []
c_ik = []
c_j = []
Q_ik = []
Q_ori = []

# Kalman Filter
x_hats = []
measure = pd.read_csv(f'{input_path}/measurement_input.csv')
measure = pd.DataFrame(measure).to_numpy()

mass_in = 0.0
mass_over_time = []
true_mass_over_time = []
volume_in = 0.0
volume_over_time = []
true_volume_over_time = []
volume_initial = ( (superlink.A_sj*(superlink.H_j-superlink.z_inv_j)).sum()
                 + (superlink.A_ik*superlink._dx_ik).sum()
                 + (superlink._A_SIk*superlink.h_Ik).sum()
                 + (waterquality.c_dk * waterquality._A_dk_next * waterquality._dx_dk).sum()
                 + (waterquality.c_uk * waterquality._A_uk_next * waterquality._dx_uk).sum() )


# Load forcing data
Q_in_file = pd.read_csv(f'{input_path}/N_Data_flow_input_5sec.csv', index_col=0)
C_in = pd.read_csv(f'{input_path}/N_Data_conc_input_5sec.csv', index_col=0)
#H_bc = pd.read_csv(f'{input_path}/boundary_stage.csv', index_col=0)

t_end = Q_in_file.index[-1]
T = 0
     
# Spin-up model to avoid running dry
#superlink.reposition_junctions()
#superlink.spinup(n_steps=2000)
    
# Create "open" control signal for orifices
u = np.ones(len(orifices))

with Simulation(superlink, dt=dt, t_end=t_end, interpolation_method='linear') as simulation:
    coeffs = simulation.h0321

    Num_T = int(simulation.t_end/dt)+1
    Estimated_out = np.zeros(Num_T)
    Original_out = np.zeros(Num_T)
    Observed_out = np.zeros(Num_T)

    while superlink.t < t_end:
        c_0j = C_in.iloc[T][:]
        c_0j = np.array(c_0j)
        Q_in = Q_in_file.iloc[T][:]
        Q_in = np.array(Q_in)
        # Advance hydraulic model
        simulation.step(dt=dt,u_o=u, Q_in = Q_in)
        # Reposition junctions to capture backwater effects
        #simulation.model.reposition_junctions()
        # Advance water quality model
        waterquality.step(dt=dt, c_0j=c_0j)
        # mass and volume balance
        mass_in += dt * (c_0j * Q_in).sum()
        mass_in_sys = ((superlink._dx_ik * superlink._A_ik * waterquality.c_ik).sum()
                 + (waterquality.c_Ik * superlink.h_Ik * superlink._A_SIk).sum()
                 + (waterquality.c_j * superlink.A_sj*(superlink.H_j - superlink.z_inv_j)).sum() 
                 + (waterquality.c_dk * waterquality._A_dk_next * waterquality._dx_dk).sum()
                 + (waterquality.c_uk * waterquality._A_uk_next * waterquality._dx_uk).sum())
        mass_over_time.append(mass_in_sys)
        true_mass_over_time.append(mass_in)
        volume_in += dt *(Q_in_file.iloc[T][:]).sum()
        volume_in_sys = ( (superlink.A_sj*(superlink.H_j-superlink.z_inv_j)).sum()
                         + (superlink.A_ik*superlink._dx_ik).sum()
                         + (superlink._A_SIk*superlink.h_Ik).sum()
                         - volume_initial)
        volume_over_time.append(volume_in_sys)
        true_volume_over_time.append(volume_in)
        '''
        # Applying the Kalman filter
  
        Next parameters can be set in 'superlink_WQ_paras.py'.
         : Observation point(# of superjunction), σ_measurement and σ_process noise
        Options for Kalman filter
         - KF = 1 : Making an artificial measurement data with the random noise.
         - KF = 2 : Data Assimilation with the input measurement data
         - Other numbers = No Kalman filter application
        '''
        KF = 0
        if KF == 1:
            waterquality.Kalman_Filter_WQ1(_dt=dt)
            Observed_out[T] = waterquality.Z
        elif KF == 2:
            waterquality.Kalman_Filter_WQ2(measure=measure[T,1], _dt=dt)
            waterquality.solve_boundary_states()
            waterquality.solve_internals_backwards()
            Observed_out[T] = measure[T,1]
        else:
            pass
        x_hats.append(waterquality.x_hat.copy())
        #print('time =', T*dt)
        #print('error =', waterquality.x_hat - waterquality.c_j)
        Original_out[T] = waterquality.c_j[waterquality.Observation_point]
        Estimated_out[T] = waterquality.x_hat[waterquality.Observation_point] 
        # Record nodal contaminant concentrations
        c_Ik.append(waterquality.c_Ik.copy())
        c_ik.append(waterquality.c_ik.copy())
        c_j.append(waterquality.c_j.copy())
        Q_ik.append(waterquality._Q_ik_next.copy())
        Q_ori.append(superlink.Q_o.copy())
        
        simulation.record_state()
        simulation.print_progress()
        T += 1


# Convert hydraulic head states to water depth
h_j = (simulation.states.H_j - simulation.states.H_j.iloc[0])
Q_ori = simulation.states.Q_o

# Plot the results
fig, ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(3, 2, figsize=(12,12))
plt.rc('font', size=12)
sns.set_palette('viridis', internal_links + 1)
_ = ax1.plot(c_j)
ax1.set_ylabel('Concentration $(g/m^3)$')
ax1.set_xlabel('Time (s)')
ax1.set_ylim(0,150)
ax1.set_title("Concentrations of super junctions")
ax2.plot(c_ik)
ax2.set_ylabel('Concentration $(g/m^3)$')
ax2.set_xlabel('Time (s)')
ax2.set_ylim(0,150)
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
ax5.set_ylim(0,0.5)
ax5.set_ylabel('Flow rate $(m^3/sec)$')
ax5.set_xlabel('Time (s)')
ax5.set_title("Flow rates of internal links")
ax6.plot(Q_ori)
ax6.set_ylim(0,0.5)
ax6.set_ylabel('Flow rate $(m^3/sec)$')
ax6.set_xlabel('Time (s)')
ax6.set_title("Flow rates of Orifices")
fig.tight_layout()
plt.show()

percent_mass = 100*(mass_in_sys/mass_in)
percent_volume = 100*(volume_in_sys/volume_in)
print("End")
print("Mass(%, in system/input)  = " , percent_mass)
print("Volume(%, in system/input)  = " , percent_volume)

#####
plt.figure(1)
plt.figure(figsize= (10,6))
plt.plot(h_j[['93-49743']], label = 'Pipedream(93-49743)', color = 'r')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-49743']]*0.3048, ":", label = 'SWMM(93-49743)', color = 'r')

plt.plot(h_j[['93-49868']], label = 'Pipedream(93-49868)', color = 'g')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-49868']]*0.3048, ":", label = 'SWMM(93-49868)', color = 'g')

plt.plot(h_j[['93-49919']], label = 'Pipedream(93-49919)', color = 'b')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-49919']]*0.3048,  ":", label = 'SWMM(93-49919)', color = 'b')
plt.title('Depth of the storages')
plt.ylabel('Storage Depth (m)')
plt.ylim(0,5)
plt.xlabel('Time')
plt.legend()

plt.figure(2)
plt.figure(figsize= (10,6))
plt.plot(h_j[['93-49921']], label = 'Pipedream', color = 'r')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-49921']]*0.3048, ":", label = 'SWMM(93-49921)', color = 'r')

plt.plot(h_j[['93-50074']], label = 'Pipedream', color = 'g')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50074']]*0.3048, ":", label = 'SWMM(93-50074)', color = 'g')

plt.plot(h_j[['93-50076']], label = 'pipedream', color = 'b')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50076']]*0.3048, ":", label = 'SWMM(93-50076)', color = 'b')
plt.title('Depth of the storages')
plt.ylabel('Storage Depth (m)')
plt.ylim(0,3)
plt.xlabel('Time')
plt.legend()

plt.figure(3)
plt.figure(figsize= (10,6))
plt.plot(h_j[['93-50077']], label = 'Pipedream', color = 'r')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50077']]*0.3048,":", label = 'SWMM(93-50077)', color = 'r')

plt.plot(h_j[['93-50081']], label = 'Pipedream', color = 'g')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50081']]*0.3048,":", label = 'SWMM(93-50081)', color = 'g')

plt.plot(h_j[['93-50225']], label = 'Pipedream', color = 'b')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50225']]*0.3048,":", label = 'SWMM(93-50225)', color = 'b')
plt.title('Depth of the storages')
plt.ylabel('Storage Depth (m)')
plt.ylim(0,2)
plt.xlabel('Time')
plt.legend()

plt.figure(4)
plt.figure(figsize= (10,6))
plt.plot(h_j[['93-90357']], label = 'pipedream', color = 'r')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-90357']]*0.3048, ":", label = 'SWMM(93-90357)', color = 'r')

plt.plot(h_j[['93-90358']], label = 'pipedream', color = 'g')
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-90358']]*0.3048, ":", label = 'SWMM(93-90358)', color = 'g')
plt.title('Depth of the storages')
plt.ylabel('Storage Depth (m)')
plt.ylim(0,2)
plt.xlabel('Time')
plt.legend()


## Orifice flow
plt.figure(5)
plt.figure(figsize= (10,6))
plt.plot(Q_ori[[0]], label = 'pipedream', color = 'r')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR34']]*0.028317, ":", label = 'SWMM(OR34)', color = 'r')

plt.plot(Q_ori[[1]], label = 'pipedream', color = 'g')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR35']]*0.028317, ":",label = 'SWMM(OR35)', color = 'g')

plt.plot(Q_ori[[2]], label = 'pipedream', color = 'b')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR36']]*0.028317,":", label = 'SWMM(OR36)', color = 'b')
plt.title('Flow rate of the orifices')
plt.ylabel('Flow rate(㎥/sec)')
plt.ylim(0,0.7)
plt.xlabel('Time')
plt.legend()

plt.figure(6)
plt.figure(figsize= (10,6))
plt.plot(Q_ori[[3]], label = 'pipedream', color = 'r')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR38']]*0.028317,":", label = 'SWMM(OR38)', color = 'r')

plt.plot(Q_ori[[4]], label = 'pipedream', color = 'g')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR39']]*0.028317,":", label = 'SWMM(OR39)', color = 'g')

plt.plot(Q_ori[[5]], label = 'pipedream', color = 'b')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR43']]*0.028317,":", label = 'SWMM(OR43)', color = 'b')
plt.title('Flow rate of the orifices')
plt.ylabel('Flow rate(㎥/sec)')
plt.ylim(0,0.5)
plt.xlabel('Time')
plt.legend()

plt.figure(7)
plt.figure(figsize= (10,6))
plt.plot(Q_ori[[6]], label = 'pipedream', color = 'r')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR44']]*0.028317, ":",label = 'SWMM(OR44)', color = 'r')

plt.plot(Q_ori[[7]], label = 'pipedream', color = 'g')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR45']]*0.028317,":", label = 'SWMM(OR45)', color = 'g')

plt.plot(Q_ori[[8]], label = 'pipedream', color = 'b')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR46']]*0.028317,":", label = 'SWMM(OR46)', color = 'b')
plt.title('Flow rate of the orifices')
plt.ylabel('Flow rate(㎥/sec)')
plt.ylim(0,0.5)
plt.xlabel('Time')
plt.legend()

plt.figure(8)
plt.figure(figsize= (10,6))
plt.plot(Q_ori[[9]], label = 'pipedream', color = 'r')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR47']]*0.028317,":", label = 'SWMM(OR47)', color = 'r')

plt.plot(Q_ori[[10]], label = 'pipedream', color = 'g')
plt.plot(Data_swmm_flow[['time']], Data_swmm_flow[['OR48']]*0.028317,":", label = 'SWMM(OR48)', color = 'g')
plt.title('Flow rate of the orifices')
plt.ylabel('Flow rate(㎥/sec)')
plt.ylim(0,0.5)
plt.xlabel('Time')
plt.legend()


## TSS output
c_j = pd.DataFrame(c_j)  
plt.figure(9)
plt.figure(figsize= (10,6))
plt.plot(c_j[:][0], label = 'Pipedream(93-49743)', color = 'r')
plt.plot(Data_swmm_tss[['93-49743']], ":", label = 'SWMM(93-49743)', color = 'r')

plt.plot(c_j[:][1], label = 'Pipedream(93-49868)', color = 'g')
plt.plot(Data_swmm_tss[['93-49868']], ":", label = 'SWMM(93-49868)', color = 'g')

plt.plot(c_j[:][2], label = 'Pipedream(93-49919)', color = 'b')
plt.plot(Data_swmm_tss[['93-49919']],  ":", label = 'SWMM(93-49919)', color = 'b')
plt.title('TSS of the storages')
plt.ylabel('TSS (mg/L)')
plt.ylim(0,120)
plt.xlabel('Time')
plt.legend()

plt.figure(10)
plt.figure(figsize= (10,6))
plt.plot(c_j[:][3], label = 'Pipedream', color = 'r')
plt.plot(Data_swmm_tss[['93-49921']], ":", label = 'SWMM(93-49921)', color = 'r')

plt.plot(c_j[:][4], label = 'Pipedream', color = 'g')
plt.plot(Data_swmm_tss[['93-50074']], ":", label = 'SWMM(93-50074)', color = 'g')

plt.plot(c_j[:][5], label = 'pipedream', color = 'b')
plt.plot(Data_swmm_tss[['93-50076']], ":", label = 'SWMM(93-50076)', color = 'b')
plt.title('TSS of the storages')
plt.ylabel('TSS (mg/L)')
plt.ylim(0,100)
plt.xlabel('Time')
plt.legend()

plt.figure(11)
plt.figure(figsize= (10,6))
plt.plot(c_j[:][6], label = 'Pipedream', color = 'r')
plt.plot(Data_swmm_tss[['93-50077']],":", label = 'SWMM(93-50077)', color = 'r')

plt.plot(c_j[:][7], label = 'Pipedream', color = 'g')
plt.plot(Data_swmm_tss[['93-50081']],":", label = 'SWMM(93-50081)', color = 'g')

plt.plot(c_j[:][8], label = 'Pipedream', color = 'b')
plt.plot(Data_swmm_tss[['93-50225']],":", label = 'SWMM(93-50225)', color = 'b')
plt.title('TSS of the storages')
plt.ylabel('TSS (mg/L)')
plt.ylim(0,50)
plt.xlabel('Time')
plt.legend()

plt.figure(12)
plt.figure(figsize= (10,6))
plt.plot(c_j[:][9], label = 'pipedream', color = 'r')
plt.plot(Data_swmm_tss[['93-90357']], ":", label = 'SWMM(93-90357)', color = 'r')

plt.plot(c_j[:][10], label = 'pipedream', color = 'g')
plt.plot(Data_swmm_tss[['93-90358']], ":", label = 'SWMM(93-90358)', color = 'g')
plt.title('TSS of the storages')
plt.ylabel('TSS (mg/L)')
plt.ylim(0,80)
plt.xlabel('Time')
plt.legend()
plt.show()

plt.figure(13)    
plt.figure(figsize= (10,6))
sns.set_palette('viridis', 2)
_ = plt.plot(x_hats)
plt.ylabel('Concentration $(g/m^3)$')
plt.xlabel('Time (s)')
plt.title('Estimated concentration time series at superjunctions - x_hat')
plt.ylim(0,60)

c_j.to_csv(f'{input_path}/c_j_output.csv', mode='w')
