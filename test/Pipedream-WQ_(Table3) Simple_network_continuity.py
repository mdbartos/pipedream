import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
from pipedream_solver.simulation import Simulation
import time

input_path = '../data/simple network_100m'
superjunctions = pd.read_csv(f'{input_path}/simple_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/simple_superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/simple_superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/simple_superjunction_wq_params.csv')

# input boundary 
H_bc = pd.read_csv(f'{input_path}/simple_boundary_stage.csv', index_col=0)

# Set scenarios
No_internal_link_list = [1,2,10,20,100]
dt_list = [0.5,1,5,10,60]

for No_link in No_internal_link_list:
    for dt_ in dt_list:
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
                
                mass_in += dt_ * (c_0j * Q_in).sum()
                mass_in_sys = ((superlink._dx_ik * superlink._A_ik * WQ.c_ik).sum()
                         + (WQ.c_Ik * superlink.h_Ik * superlink._A_SIk).sum()
                         + (WQ.c_j * superlink.A_sj*(superlink.H_j - superlink.z_inv_j)).sum() 
                         + (WQ.c_dk * WQ._A_dk_next * WQ._dx_dk).sum()
                         + (WQ.c_uk * WQ._A_uk_next * WQ._dx_uk).sum())
                mass_over_time.append(mass_in_sys)
                true_mass_over_time.append(mass_in)
                volume_in += (dt_ * Q_in).sum()
                volume_in_sys = ( (superlink.A_sj*(superlink.H_j-superlink.z_inv_j)).sum()
                                 + (superlink.A_ik*superlink._dx_ik).sum()
                                 + (superlink._A_SIk*superlink.h_Ik).sum()
                                 + (superlink.A_uk*superlink._dx_uk).sum()
                                 + (superlink.A_dk*superlink._dx_dk).sum()
                                 ) - volume_initial
                volume_over_time.append(volume_in_sys)
                true_volume_over_time.append(volume_in)
                # Record nodal contaminant concentrations
                c_j.append(WQ.c_j.copy())
        print("\nNo of links :", No_link, "dt = ", dt_)
        print("Elaplsed time (sec) = ", time.perf_counter() - start_t)
        print(f'Water quality continuity error (%) = {100* abs((mass_in_sys - mass_in) / mass_in) : 5.3f}')
        print(f'Water volume continuity error (%) = {100* abs((volume_in_sys - volume_in) / volume_in) : 5.3f}')