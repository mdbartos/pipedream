import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
from pipedream_solver.simulation import Simulation
import matplotlib.pyplot as plt
import time

input_path = '../data/simple network_100m'
superjunctions = pd.read_csv(f'{input_path}/Simple_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/Simple_superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/Simple_superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/Simple_superjunction_wq_params.csv')

# input boundary 
H_bc = pd.read_csv(f'{input_path}/Simple_boundary_stage.csv', index_col=0)

# Set scenarios
No_internal_link_list = [1,2,5,20,50,80,100]
dt_ = 0.5
c_j_out_all = {}

for No_link in No_internal_link_list:
    internal_links = No_link
    superlink = SuperLink(superlinks, superjunctions, internal_links=internal_links)
    WQ = QualityBuilder(superlink, superjunction_params=superjunction_wq_params,
                                  superlink_params=superlink_wq_params)
    T_max = 1200
    C_start = 0
    C_end = 600
    C_input = 10
    Q_0Ik = 0*np.ones(superlink.NIk)
    c_j = []
    mass_in = 0.0
    mass_in_sys = 0.0
    mass_over_time = []
    true_mass_over_time = []
    volume_in = 0.0 
    volume_over_time = []
    true_volume_over_time = []
    volume_initial = ( (superlink.A_sj*(superlink.H_j-superlink.z_inv_j)).sum()
                     + (superlink.A_ik*superlink._dx_ik).sum()
                     + (superlink._A_SIk*superlink.h_Ik).sum()
                     + (superlink.A_uk*superlink._dx_uk).sum()
                     + (superlink.A_dk*superlink._dx_dk).sum() )
    
    start_t = time.perf_counter()
    with Simulation(superlink, dt=dt_, H_bc=H_bc) as simulation:
        coeffs = simulation.h0321
        
        while simulation.t <= T_max:
            # setting the contaminant input duration
            if (superlink.t > C_start) and (superlink.t <= C_end):
                # Apply contaminant to uphill superjunction
                Q_in = 1 * np.asarray([1.,0.,0.])
                c_0j = C_input * np.asarray([1., 0.,0.])
            else:     # Otherwise, no contaminant input
                Q_in = 1 * np.asarray([1.,0.,0.])
                c_0j = np.zeros(3)
                
            # Advance hydraulic model
            superlink.step(dt=dt_, Q_in=Q_in, Q_0Ik=Q_0Ik)
            
            # Advance water quality model
            WQ.step(dt=dt_, c_0j=c_0j)
            
            # Record nodal contaminant concentrations
            c_j.append(WQ.c_j.copy())
    c_j_out_all[No_link] = c_j

# Analytical solution: time series
from scipy import special
K = WQ._K_ik[1]                                  # Reaction
D = WQ._D_ik[1]                                  # Diffusion coefficient
Vx = np.mean(superlink.Q_ik/superlink.A_ik)      # velocity
t = np.linspace(0,T_max,361)
Gamma = np.sqrt(1+(4*K*D/pow(Vx,2)))
tau = C_end - 0.0000001
C0 = C_input

X_location = 100
C_An_Temp = np.zeros(361)
C_Analytical_time = np.zeros(361)
for i in range(0,361):
    if t[i] < C_end:
        C_Analytical_time[i] = 0.5*C0*( np.exp((1-Gamma)*Vx*X_location/(2*D))*special.erfc((X_location-Vx*t[i]*Gamma)/(2*np.sqrt(D*t[i])))
                                       + np.exp((1+Gamma)*Vx*X_location/(2*D))*special.erfc((X_location+Vx*t[i]*Gamma)/(2*np.sqrt(D*t[i]))))
    else:
        C_An_Temp[i] = special.erfc((X_location-Vx*t[i]*Gamma)/(2*np.sqrt(D*t[i])))
        C_An_Temp[i] += -special.erfc((X_location-Vx*(t[i]-tau)*Gamma)/(2*np.sqrt(D*(t[i]-tau))))
        C_Analytical_time[i] = 0.5*C0*np.exp(Vx*X_location*(1-Gamma)/(2*D))*C_An_Temp[i]
        C_An_Temp[i] = special.erfc((X_location+Vx*t[i]*Gamma)/(2*np.sqrt(D*t[i])))
        C_An_Temp[i] += -special.erfc((X_location+Vx*(t[i]-tau)*Gamma)/(2*np.sqrt(D*(t[i]-tau))))
        C_Analytical_time[i] += np.exp(Vx*X_location*Gamma*(1+Gamma)/(2*D)) * C_An_Temp[i]

# Print the results
plt.figure()
plt.figure(figsize= (12,6))
plt.plot(t,C_Analytical_time, color = 'r', linewidth = 3, label = 'Analytical solution')
t2 = np.linspace(0,T_max+dt_,2401)
for No_link in No_internal_link_list:
    c_j = np.array(c_j_out_all[No_link])
    label_ = 100/No_link
    plt.plot(t2, c_j[:,1], '-', color = 'b', linewidth = 1.5, alpha = 0.5, label = f'Δx = {label_ : 5.2f} m')
plt.title('Pipedream-WQ vs. Analytical solution under various spatial resolutions')
plt.xlabel('time (s)')
plt.ylabel('Contaminant concentration (mg/l)')
plt.legend()
plt.grid()
plt.show()