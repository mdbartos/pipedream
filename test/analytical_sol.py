import numpy as np
import pandas as pd
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.nquality import QualityBuilder
from pipedream_solver.simulation import Simulation
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn as sns

input_path = '../data/analytical_sol'
superjunctions = pd.read_csv(f'{input_path}/hillslope_superjunctions.csv')
superlinks = pd.read_csv(f'{input_path}/hillslope_superlinks.csv')
superlink_wq_params = pd.read_csv(f'{input_path}/hillslope_superlink_wq_params.csv')
superjunction_wq_params = pd.read_csv(f'{input_path}/hillslope_superjunction_wq_params.csv')

# input boundary 
H_bc = pd.read_csv(f'{input_path}/hillslope_boundary_stage.csv', index_col=0)

internal_links = 25
superlink = SuperLink(superlinks, superjunctions, internal_links=internal_links)
waterquality = QualityBuilder(superlink, 
                              superjunction_params=superjunction_wq_params,
                              superlink_params=superlink_wq_params)
dt = 1
T_max = 1800
C_start = 0
C_end = 1200
C_input = 10
Q_0Ik = 0*np.ones(superlink.NIk)
c_Ik = []
c_ik = []
c_dk = []
c_j = []
mass_in = 0.
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

with Simulation(superlink, dt=dt, H_bc=H_bc) as simulation:
    coeffs = simulation.h0321
    
    while simulation.t <= T_max:
        # setting the contaminant input duration
        if (superlink.t > C_start) and (superlink.t < C_end):
            # Apply contaminant to uphill superjunction
            Q_in = 0.5 * np.asarray([1.,0.])
            c_0j = C_input * np.asarray([1., 0.])
        else:
            # Otherwise, no contaminant input
            Q_in = 0.5 * np.asarray([1.,0.])
            c_0j = np.zeros(2)
            
        # Advance hydraulic model
        superlink.step(dt=dt, Q_in=Q_in, Q_0Ik=Q_0Ik)
        # Advance water quality model
        waterquality.step(dt=dt, c_0j=c_0j)
        mass_in += dt * (c_0j * Q_in).sum()
        mass_in_sys = ((superlink._dx_ik * superlink._A_ik * waterquality.c_ik).sum()
                 + (waterquality.c_Ik * superlink.h_Ik * superlink._A_SIk).sum()
                 + (waterquality.c_j * superlink.A_sj*(superlink.H_j - superlink.z_inv_j)).sum() 
                 + (waterquality.c_dk * waterquality._A_dk_next * waterquality._dx_dk).sum()
                 + (waterquality.c_uk * waterquality._A_uk_next * waterquality._dx_uk).sum())
        mass_over_time.append(mass_in_sys)
        true_mass_over_time.append(mass_in)
        volume_in += (dt * Q_in).sum()
        volume_in_sys = ( (superlink.A_sj*(superlink.H_j-superlink.z_inv_j)).sum()
                         + (superlink.A_ik*superlink._dx_ik).sum()
                         + (superlink._A_SIk*superlink.h_Ik).sum()
                         + (superlink.A_uk*superlink._dx_uk).sum()
                         + (superlink.A_dk*superlink._dx_dk).sum()
                         ) - volume_initial
        volume_over_time.append(volume_in_sys)
        true_volume_over_time.append(volume_in)
        # Record nodal contaminant concentrations
        c_Ik.append(waterquality.c_Ik.copy())
        c_ik.append(waterquality.c_ik.copy())
        c_dk.append(waterquality.c_dk.copy())
        c_j.append(waterquality.c_j.copy())
        simulation.print_progress()

# Plot the profile graph
plt.figure()
plt.figure(dpi = 150)
norm = matplotlib.colors.Normalize(vmin=0., vmax=6.5, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap='winter')
_ = superlink.plot_profile([0, 1], 
                           superlink_kwargs={'color' : 
                                         [mapper.to_rgba(c) for c in waterquality.c_ik]})

# Plot the results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
plt.rc('font', size=12)
sns.set_palette('viridis', internal_links + 1)
_ = ax1.plot(c_Ik)
ax1.set_ylabel('Concentration $(g/m^3)$')
ax1.set_xlabel('Time (s)')
ax1.set_ylim(0,20)
ax1.set_title("Internal junctions")
ax2.plot(c_ik)
ax2.set_ylabel('Concentration $(g/m^3)$')
ax2.set_xlabel('Time (s)')
ax2.set_ylim(0,20)
ax2.set_title("Internal Links")
ax3.plot(c_j)
ax3.set_ylabel('Concentration $(g/m^3)$')
ax3.set_xlabel('Time (s)')
ax3.set_ylim(0,20)
ax3.set_title("Superjunctions")
ax4.plot(true_mass_over_time, c='k', label='True mass input')
ax4.plot(mass_over_time, c='r', linestyle='--', label='Mass in system')
ax4.set_ylabel('Total mass')
ax4.set_xlabel('Time (s)')
ax4.set_title("Mass balance")
ax4.legend(loc=2)
fig.tight_layout()
plt.show()

#Analytical solution
from scipy import special
K = 0                                       # Reaction
D = 1.5                                  # Diffusion coefficient
Vx = superlink.Q_ik[5]/superlink.A_ik[5]    # velocity
t = T_max
Gamma = np.sqrt(1+(4*K*D/pow(Vx,2)))
tau = C_end - 0.0000001
C0 = C_input

X_location = np.zeros(internal_links)
X_location[0] = superlink._dx_ik[5]/2
C_Analytical = np.zeros(internal_links)

for i in range(1,internal_links):
    X_location[i] = X_location[i-1] + superlink._dx_ik[5]
for i in range(0,internal_links): 
    C_An_Temp = special.erfc((X_location[i]-Vx*t*Gamma)/(2*np.sqrt(D*t)))
    C_An_Temp += -special.erfc((X_location[i]-Vx*(t-tau)*Gamma)/(2*np.sqrt(D*(t-tau))))
    C_Analytical[i] = 0.5*C0*np.exp(Vx*X_location[i]*(1-Gamma)/(2*D))*C_An_Temp
    C_An_Temp = special.erfc((X_location[i]+Vx*t*Gamma)/(2*np.sqrt(D*t)))
    C_An_Temp += -special.erfc((X_location[i]+Vx*(t-tau)*Gamma)/(2*np.sqrt(D*(t-tau))))
    C_Analytical[i] += np.exp(Vx*X_location[i]*(1+Gamma)/(2*D)) * C_An_Temp

time = int(T_max/dt)

plt.figure(2)
plt.figure(dpi = 150)
plt.plot(X_location[0:internal_links], c_ik[time-1], color = 'blue',  linewidth=1.5, label ='Numerical solution')
plt.plot(X_location[0:internal_links], C_Analytical[0:internal_links], ':', color = 'black',  linewidth=1.5, label ='Analytical solution')
plt.legend(loc=3)
plt.xlabel('Location(m)')
plt.ylabel('Concentration(mg/L)')
plt.show()

plt.plot(true_volume_over_time, c='k', label='True volume input')
plt.plot(volume_over_time, c='r', linestyle='--', label='volume in system')
plt.ylabel('Total volume of water')
plt.xlabel('Time (s)')
plt.title("volume balance")
plt.legend(loc=2)
plt.show()

percent_mass = 100*(mass_in_sys/mass_in)
percent_volume = 100*(volume_in_sys/volume_in)
print("Mass(%, in system/input)  = " , percent_mass)
print("Volume(%, in system/input)  = " , percent_volume)