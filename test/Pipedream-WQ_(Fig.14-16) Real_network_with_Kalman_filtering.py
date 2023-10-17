import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
import matplotlib.pyplot as plt
from pipedream_solver.simulation import Simulation

input_path = '../data/real network'
superjunctions = pd.read_csv(f'{input_path}/superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/superjunction_wq_params.csv')
orifices = pd.read_csv(f'{input_path}/orifices.csv')

# Load comparison data
Data_swmm_depth = pd.read_csv(f'{input_path}/SWMM_result/Data_swmm_depth.csv')
Data_swmm_flow = pd.read_csv(f'{input_path}/SWMM_result/Data_swmm_flow.csv')
Data_swmm_tss = pd.read_csv(f'{input_path}/SWMM_result/Data_swmm_tss.csv')

dt = 5
internal_links = 60
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

# Kalman Filter variables and data
x_hats = []
measure = pd.read_csv(f'{input_path}/measurement_input.csv')
measure = pd.DataFrame(measure).to_numpy()

# Scenario names of the base and 10 disturbed input timeseries of contaminant concentration
scenario_list = ['Base', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']

# Load the input wq data of scenarios
wq_in_data = {}
wq_input_file_prefix_ = 'N_Data_conc_input_'
for i in range(0,len(scenario_list)):
    wq_in_data[scenario_list[i]] = pd.read_csv(f'{input_path}/{wq_input_file_prefix_+scenario_list[i]}.csv', index_col=0)
    wq_in_data[scenario_list[i]] = pd.DataFrame(wq_in_data[scenario_list[i]]).to_numpy()

# Load forcing data
Q_in_file = pd.read_csv(f'{input_path}/N_Data_flow_input.csv', index_col=0)

# Create "open" control signal for orifices
u = np.ones(len(orifices))

# Model only simulation for 11 different scenarios
c_j_model_only = {}
for scenario_ in scenario_list:
    superlink = SuperLink(superlinks, superjunctions,orifices=orifices,
                          internal_links=internal_links, bc_method = 'b', min_depth=10e-4, mobile_elements=False)
    waterquality = QualityBuilder(superlink, 
                                  superjunction_params=superjunction_wq_params,
                                  superlink_params=superlink_wq_params)
    C_in = wq_in_data[scenario_]
    t_end = Q_in_file.index[-1]
    T = 0

    c_j_temp = []
    with Simulation(superlink, dt=dt, t_end=t_end, interpolation_method='linear') as simulation:
        coeffs = simulation.h0321
        Num_T = int(simulation.t_end/dt)+1
        Estimated_out = np.zeros(Num_T)
        Original_out = np.zeros(Num_T)
        Observed_out = np.zeros(Num_T)
    
        while superlink.t < t_end:
            c_0j = C_in[T,:]
            c_0j = np.array(c_0j)
            Q_in = Q_in_file.iloc[T][:]
            Q_in = np.array(Q_in)
    
            # Advance hydraulic model
            simulation.step(dt=dt, u_o=u, Q_in = Q_in)
    
            # Advance water quality model
            waterquality.step(dt=dt, c_0j=c_0j)

            # Record nodal contaminant concentrations
            c_j_temp.append(waterquality.c_j.copy())
            
            simulation.record_state()
            simulation.print_progress()
            T += 1
    c_j_model_only[scenario_] = c_j_temp
    print(f'\n▶▶▶▶▶ [Model only] Scenario "{scenario_}" is finished.')

# Model + Kalman Filtering simulation for 11 different scenarios: without correlation in process noise covariance matrix
c_j_KF1 = {}
for scenario_ in scenario_list:
    superlink = SuperLink(superlinks, superjunctions,orifices=orifices,
                          internal_links=internal_links, bc_method = 'b', min_depth=10e-4, mobile_elements=False)
    waterquality = QualityBuilder(superlink, 
                                  superjunction_params=superjunction_wq_params,
                                  superlink_params=superlink_wq_params)
    C_in = wq_in_data[scenario_]
    t_end = Q_in_file.index[-1]
    T = 0

    c_j_temp = []
    with Simulation(superlink, dt=dt, t_end=t_end, interpolation_method='linear') as simulation:
        coeffs = simulation.h0321
        Num_T = int(simulation.t_end/dt)+1
        Estimated_out = np.zeros(Num_T)
        Original_out = np.zeros(Num_T)
        Observed_out = np.zeros(Num_T)
    
        while superlink.t < t_end:
            c_0j = C_in[T,:]
            c_0j = np.array(c_0j)
            Q_in = Q_in_file.iloc[T][:]
            Q_in = np.array(Q_in)
    
            # Advance hydraulic model
            simulation.step(dt=dt, u_o=u, Q_in = Q_in)
    
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
                Observed_out[T] = waterquality.Z
            elif KF == 2:
                corr_option = 0   # uncorrelated process noise covariance matix
                waterquality.Kalman_Filter_WQ2(measure=measure[T,1], _dt=dt, process_correlation=corr_option)
                waterquality.solve_boundary_states()
                waterquality.solve_internals_backwards()
                Observed_out[T] = measure[T,1]
            else:
                pass
            x_hats.append(waterquality.x_hat.copy())
    
            # Record nodal contaminant concentrations
            c_j_temp.append(waterquality.c_j.copy())
            
            simulation.record_state()
            simulation.print_progress()
            T += 1
    c_j_KF1[scenario_] = c_j_temp
    print(f'\n▶▶▶▶▶ [Model + Kalman Filtering: with correlation] Scenario "{scenario_}" is finished.')

# Model + Kalman Filtering simulation for 11 different scenarios: with correlation in process noise covariance matrix
c_j_KF2 = {}
for scenario_ in scenario_list:
    superlink = SuperLink(superlinks, superjunctions,orifices=orifices,
                          internal_links=internal_links, bc_method = 'b', min_depth=10e-4, mobile_elements=False)
    waterquality = QualityBuilder(superlink, 
                                  superjunction_params=superjunction_wq_params,
                                  superlink_params=superlink_wq_params)
    C_in = wq_in_data[scenario_]
    t_end = Q_in_file.index[-1]
    T = 0

    c_j_temp = []
    with Simulation(superlink, dt=dt, t_end=t_end, interpolation_method='linear') as simulation:
        coeffs = simulation.h0321
        Num_T = int(simulation.t_end/dt)+1
        Estimated_out = np.zeros(Num_T)
        Original_out = np.zeros(Num_T)
        Observed_out = np.zeros(Num_T)
    
        while superlink.t < t_end:
            c_0j = C_in[T,:]
            c_0j = np.array(c_0j)
            Q_in = Q_in_file.iloc[T][:]
            Q_in = np.array(Q_in)
    
            # Advance hydraulic model
            simulation.step(dt=dt, u_o=u, Q_in = Q_in)
    
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
                Observed_out[T] = waterquality.Z
            elif KF == 2:
                corr_option = 1  # correlated process noise covariance matix
                waterquality.Kalman_Filter_WQ2(measure=measure[T,1], _dt=dt, process_correlation=corr_option)
                waterquality.solve_boundary_states()
                waterquality.solve_internals_backwards()
                Observed_out[T] = measure[T,1]
            else:
                pass
            x_hats.append(waterquality.x_hat.copy())
    
            # Record nodal contaminant concentrations
            c_j_temp.append(waterquality.c_j.copy())
            
            simulation.record_state()
            simulation.print_progress()
            T += 1
    c_j_KF2[scenario_] = c_j_temp
    print(f'\n▶▶▶▶▶ [Model + Kalman Filtering: with correlation] Scenario "{scenario_}" is finished.')

# Convert hydraulic head states to water depth
h_j = (simulation.states.H_j - simulation.states.H_j.iloc[0])

## Fig.14 in the paper: TSS output
c_j2 = np.array(c_j_model_only['Base'])
plt.figure(2)
plt.figure(figsize= (10,6))
x_axis_ = np.linspace(dt,dt*len(h_j),len(h_j))
plt.plot(x_axis_[0:-1],c_j2[:,3], label = 'Pipedream', color = 'black', alpha = 0.5)
plt.plot(x_axis_,Data_swmm_tss[['93-49921']], ":", label = 'SWMM(93-49921)', color = 'black')
plt.plot(x_axis_[0:-1],c_j2[:,5], label = 'pipedream', color = 'b', alpha = 0.5)
plt.plot(x_axis_,Data_swmm_tss[['93-50076']], ":", label = 'SWMM(93-50076)', color = 'b')
plt.plot(x_axis_[0:-1],c_j2[:,7], label = 'Pipedream', color = 'r', alpha = 0.5)
plt.plot(x_axis_,Data_swmm_tss[['93-50081']],":", label = 'SWMM(93-50081)', color = 'r')
plt.plot(x_axis_[0:-1],c_j2[:,6], label = 'Pipedream', color = 'g', alpha = 0.5)
plt.plot(x_axis_,Data_swmm_tss[['93-50077']],":", label = 'SWMM(93-50077)', color = 'g')
plt.title('TSS of the storages')
plt.ylabel('TSS (mg/L)')
plt.ylim(0,80)
plt.xlabel('Time (seconds)')
plt.legend()

# Fig.14 in the paper : Depth of storages
plt.figure(1)
plt.figure(figsize= (10,6))
plt.plot(h_j[['93-49921']], label = 'Pipedream', color = 'black', alpha = 0.5)
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-49921']]*0.3048, ":", label = 'SWMM(93-49921)', color = 'black')
plt.plot(h_j[['93-50076']], label = 'pipedream', color = 'b', alpha = 0.5)
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50076']]*0.3048, ":", label = 'SWMM(93-50076)', color = 'b')
plt.plot(h_j[['93-50081']], label = 'Pipedream', color = 'r', alpha = 0.5)
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50081']]*0.3048,":", label = 'SWMM(93-50081)', color = 'r')
plt.plot(h_j[['93-50077']], label = 'Pipedream', color = 'g', alpha = 0.5)
plt.plot(Data_swmm_depth[['time']], Data_swmm_depth[['93-50077']]*0.3048,":", label = 'SWMM(93-50077)', color = 'g')
plt.title('Depth of the storages')
plt.ylabel('Storage Depth (m)')
plt.ylim(0,3)
plt.xlabel('Time (seconds)')
plt.legend()

## Fig. 15 in the paper: without correlation in process noise covariance matrix
out_name = [3,5,6,7]
storage_name = ['Storage 1','Storage 2','Storage 3','Storage 4']
y_limit_ = [40,40,80,40]
scenario_list2 = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']
for i in range(len(out_name)):
    plt.figure(int(out_name[i]))
    plt.figure(figsize= (12,7))
    if int(out_name[i])==3:
        plt.plot(measure[:,0],measure[:,1], color = 'r', linewidth = 0.5, alpha = 0.5, label = 'Artificial measurement')
    for scenario_ in scenario_list2:
        c_j2 = np.array(c_j_KF1[scenario_])
        plt.plot(x_axis_[0:-1],c_j2[:,int(out_name[i])], color = 'b', linewidth = 0.5, label = f'(Scenario: {scenario_})')
    for scenario_ in scenario_list2:
        c_j3 = np.array(c_j_model_only[scenario_])
        plt.plot(x_axis_[0:-1],c_j3[:,int(out_name[i])], ':', color = 'gray', alpha = 0.5, label = f'(Scenario: {scenario_})')
    c_j2 = np.array(c_j_model_only['Base'])
    plt.plot(x_axis_[0:-1],c_j2[:,int(out_name[i])], color = 'r', linewidth = 2, label = 'Ground truth')
    plt.title(f'TSS concentrations in {storage_name[i]} : without correlation')
    plt.ylabel('TSS (mg/L)')
    plt.ylim(0,int(y_limit_[i]))
    plt.xlabel('Time (seconds)')
    plt.legend()

## Fig. 16 in the paper: without correlation in process noise covariance matrix
out_name = [5,5]
storage_name = ['Storage 1','Storage 2','Storage 3','Storage 4']
y_limit_ = [50,50]
x_axis_ = np.linspace(dt,dt*len(h_j),len(h_j))
scenario_list2 = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']

# Storage 10 : uncorrelated
plt.figure()
plt.figure(figsize= (12,7))
for scenario_ in scenario_list2:
    c_j2 = np.array(c_j_KF1[scenario_])
    plt.plot(x_axis_[0:-1],c_j2[:,4], color = 'b', linewidth = 0.5, label = f'(Scenario: {scenario_})')
for scenario_ in scenario_list2:
    c_j3 = np.array(c_j_model_only[scenario_])
    plt.plot(x_axis_[0:-1],c_j3[:,4], ':', color = 'gray', alpha = 0.5, label = f'(Scenario: {scenario_})')
c_j2 = np.array(c_j_model_only['Base'])
plt.plot(x_axis_[0:-1],c_j2[:,4], color = 'r', linewidth = 2, label = 'Ground truth')
plt.title("TSS concentrations in Storage 10 : without correlation")
plt.ylabel('TSS (mg/L)')
plt.ylim(0,50)
plt.xlabel('Time (seconds)')
plt.legend()

# Storage 10 : correlated
plt.figure()
plt.figure(figsize= (12,7))
for scenario_ in scenario_list2:
    c_j2 = np.array(c_j_KF2[scenario_])
    plt.plot(x_axis_[0:-1],c_j2[:,4], color = 'b', linewidth = 0.5, label = f'(Scenario: {scenario_})')
for scenario_ in scenario_list2:
    c_j3 = np.array(c_j_model_only[scenario_])
    plt.plot(x_axis_[0:-1],c_j3[:,4], ':', color = 'gray', alpha = 0.5, label = f'(Scenario: {scenario_})')
c_j2 = np.array(c_j_model_only['Base'])
plt.plot(x_axis_[0:-1],c_j2[:,4], color = 'r', linewidth = 2, label = 'Ground truth')
plt.title("TSS concentrations in Storage 10 : with correlation")
plt.ylabel('TSS (mg/L)')
plt.ylim(0,50)
plt.xlabel('Time (seconds)')
plt.legend()

# Storage 11 : uncorrelated
plt.figure()
plt.figure(figsize= (12,7))
for scenario_ in scenario_list2:
    c_j2 = np.array(c_j_KF1[scenario_])
    plt.plot(x_axis_[0:-1],c_j2[:,2], color = 'b', linewidth = 0.5, label = f'(Scenario: {scenario_})')
for scenario_ in scenario_list2:
    c_j3 = np.array(c_j_model_only[scenario_])
    plt.plot(x_axis_[0:-1],c_j3[:,2], ':', color = 'gray', alpha = 0.5, label = f'(Scenario: {scenario_})')
c_j2 = np.array(c_j_model_only['Base'])
plt.plot(x_axis_[0:-1],c_j2[:,2], color = 'r', linewidth = 2, label = 'Ground truth')
plt.title("TSS concentrations in Storage 11 : without correlation")
plt.ylabel('TSS (mg/L)')
plt.ylim(0,80)
plt.xlabel('Time (seconds)')
plt.legend()

# Storage 11 : correlated
plt.figure()
plt.figure(figsize= (12,7))
for scenario_ in scenario_list2:
    c_j2 = np.array(c_j_KF2[scenario_])
    plt.plot(x_axis_[0:-1],c_j2[:,2], color = 'b', linewidth = 0.5, label = f'(Scenario: {scenario_})')
for scenario_ in scenario_list2:
    c_j3 = np.array(c_j_model_only[scenario_])
    plt.plot(x_axis_[0:-1],c_j3[:,2], ':', color = 'gray', alpha = 0.5, label = f'(Scenario: {scenario_})')
c_j2 = np.array(c_j_model_only['Base'])
plt.plot(x_axis_[0:-1],c_j2[:,2], color = 'r', linewidth = 2, label = 'Ground truth')
plt.title("TSS concentrations in Storage 11 : with correlation")
plt.ylabel('TSS (mg/L)')
plt.ylim(0,80)
plt.xlabel('Time (seconds)')
plt.legend()