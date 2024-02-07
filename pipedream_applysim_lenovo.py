import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wntr
import scipy as sc
import networkx as nx
import networkx.drawing.nx_pylab as nxp
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation
from pipedream_solver.nutils import interpolate_sample
import random
import time
import pickle
import pipedream_utility as pdu
from pipedream_utility import *
import pipedream_simulation as pd_sim
from pipedream_simulation import *
import pipedream_simulation_sensor_results as pd_sim_sensor
from pipedream_simulation_sensor_results import *
from pipedream_kalman_filter import apply_EKF
import viswaternet as vis
import matplotlib as mpl

#Don't show future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
images_dir = 'Images/'
 

# In[] Run the greedy sensor placement problem and extract results
net_list = ['Networks/NetSimp_valve.inp']
net_list = ['Networks/PRV_test.inp'] 
net_list = ['Networks/Net2 active throughout.inp']
net_list = ['Networks/Net2 prv open.inp']
net_list = ['Networks/PRV_closed.inp']
#net_list = ['Networks/headpump ky6.inp']
#net_list = ['Networks/No controls/Net3_nocontrols.inp']           # a list of networks you want to run simulations for 
#net_list = ['Networks/No controls/reduced 11 headpump ky3_nocontrols.inp']   

metric = 'mean'
percent = False
Rcov = 1
source_node = False
max_sensors = 5
dt = 3600                                                        # time step duration in s
t_run = 12                                                      # total simulation duration in hr
banded = False
percentile = 90
cutoff_timestep = 5
sensors_list = []  
Rcov_case = 2
fs = 16

# In[] EPANET node and link comparison plots

cmap = 'coolwarm'
for inp in net_list:
    name = inp
    H_df, Q_df, Q_pump, Q_prv, model, Q_in_all_df, pumps, superjunctions, orifices, superlinks, prvs = run_pipedream_simulation(inp, t_run = t_run, dt = dt, banded = False)
    wn = wntr.network.WaterNetworkModel(inp)
    wn.options.time.report_timestep=dt    
    wn.options.time.duration=t_run*3600
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    abs_diff_node = abs(results.node['head'].loc[cutoff_timestep*dt:,]-H_df.loc[cutoff_timestep*dt:,]).dropna(axis=1,how='all').dropna(axis=0,how='all').mean()
    abs_diff_link = abs(results.link['flowrate'].loc[cutoff_timestep*dt:,]-Q_df.loc[cutoff_timestep*dt:,]).dropna(axis=1,how='all').dropna(axis=0,how='all').mean()
    print(name, 'node:', abs_diff_node.mean(), 'link', abs_diff_link.mean())
    

    fig, ax = plt.subplots(1, 2, figsize = (12,6))
    node_size = 200

    
    junction_names = list(abs_diff_node.index)
    link_names = list(abs_diff_link.index)
    junction_values = list(abs_diff_node)
    link_values = list(abs_diff_link)   
    
    vis_model = vis.VisWNModel(inp)
        
    ax[0].set_title(name, fontsize = fs+2)
    ax[0].set_frame_on(False) 
    vis_model.plot_unique_data(ax=ax[0], parameter = "custom_data", parameter_type = 'node', 
                                       custom_data_values = [junction_names, junction_values], data_type = 'continuous', 
                                       cmap = cmap, line_widths = 0, edge_colors = 'k', vmin = 0, draw_color_bar = True,  
                                       node_size = node_size, tank_color='k', draw_base_legend=False)
    
    ax[1].set_frame_on(False) 
    vis_model.plot_unique_data(ax=ax[1], parameter = "custom_data", parameter_type = 'link', 
                                       custom_data_values = [link_names, link_values], data_type = 'continuous', 
                                       cmap = cmap, line_widths = 1,  vmin = 0, draw_color_bar = True,  
                                       node_size = 0, tank_color='k', draw_base_legend=False, min_width=3, max_width = 3)
    
    start_ind_pd = 0
    start_ind_wn = start_ind_pd
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plot pipedream and wntr node heads 
    n_superjunctions = wn.num_junctions
    n_cols = 4
    n_rows = min(10,int(np.ceil(n_superjunctions // n_cols)))
    if n_rows==0:
        n_rows = 1
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 0.75 * 12 * n_rows / n_cols))
    
    wntr_results=results.node['head'].iloc[:-1,:]
    y_max = wntr_results.max().max() + 1
    y_min = wntr_results.min().min() - 1
    
    #junction_names = superjunctions.name.to_list()
    
    for i in range(n_rows * n_cols):
        if i < len(junction_names):
            ax.flat[i].plot(H_df.index[start_ind_pd:]/3600,H_df[junction_names[i]][start_ind_pd:], c='r', alpha=0.75, label = 'Pipedream')
            ax.flat[i].plot(wntr_results.index[start_ind_wn:]/3600,wntr_results[junction_names[i]][start_ind_wn:], c='0.3', linestyle = '--', alpha=0.75, label = 'EPANET')
            ax.flat[i].set_title(f'Node {junction_names[i]}')
            ax.flat[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.flat[i].set_ylim(bottom = y_min, top = y_max)
            ax.flat[i].set_ylabel('Head ($m$)')
            ax.flat[i].set_xlabel('Hour of Day')
        else:
            ax.flat[i].set_visible(False)
        
    ax.flat[0].legend()
    plt.suptitle('Node heads')
    plt.tight_layout()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plot pipedream and wntr link flows 
    n_superlinks = wn.num_pipes
    n_cols = 4
    n_rows = min(9,int(np.ceil(n_superlinks // n_cols)))
    if n_rows==0:
        n_rows = 1
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 0.75 * 12 * n_rows / n_cols))
    
    wntr_results=results.link['flowrate'].iloc[:-1,:]
    #junction_names = superjunctions.name.to_list()
    
    for i in range(n_rows * n_cols):
        if i < len(link_names):
            ax.flat[i].plot(Q_df.index[start_ind_pd:]/3600,3600*Q_df[link_names[i]][start_ind_pd:], c='r', alpha=0.75)
            ax.flat[i].plot(wntr_results.index[start_ind_wn:]/3600,3600*wntr_results[link_names[i]][start_ind_wn:], c='0.3', linestyle = '--', alpha=0.75)
            ax.flat[i].set_title(f'Pipe {link_names[i]}')
            ax.flat[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.flat[i].set_ylabel('Flow rate ($m^3/hr$)')
            ax.flat[i].set_xlabel('Hour of Day')
        else:
            ax.flat[i].set_visible(False)
        
    ax.flat[0].legend(['Pipedream','EPANET'])
    plt.suptitle('Link flows')
    plt.tight_layout()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plot pipedream and wntr tank heads 
    n_superjunctions = wn.num_tanks
    n_cols = 4
    n_rows = min(6,int(np.ceil(wn.num_tanks / n_cols)))
    if n_rows==0:
        n_rows = 1
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 0.75 * 12 * n_rows / n_cols))
    
    wntr_results=results.node['head'].iloc[:-1,:]
    #junction_names = superjunctions.name.to_list()
    
    for i in range(n_rows * n_cols):
        if i < len(wn.tank_name_list):
            ax.flat[i].plot(H_df.index[start_ind_pd:]/3600,H_df[wn.tank_name_list[i]][start_ind_pd:], c='r', alpha=0.75)
            ax.flat[i].plot(wntr_results.index[start_ind_wn:]/3600,wntr_results[wn.tank_name_list[i]][start_ind_wn:], c='0.3', linestyle = '--', alpha=0.75)
            ax.flat[i].set_title(f'Node {wn.tank_name_list[i]}')
            ax.flat[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
#            if (i % ax.shape[1]) == 0:
#                ax.flat[i].set_ylabel('Tank Head ($m$)')
#            if (i >= ax.shape[1] * (ax.shape[0] - 1)):
#                ax.flat[i].set_xlabel('Hour of Day')
        else:
            ax.flat[i].set_visible(False)
        
    ax.flat[0].legend(['Pipedream','EPANET'])
    plt.suptitle('Tank heads')
    plt.tight_layout()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plot pipedream and wntr pump flows 
    n_superlinks = wn.num_pumps
    n_cols = 4
    n_rows = min(6,int(np.ceil(wn.num_pumps / n_cols)))
    if n_rows==0:
        n_rows = 1
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 0.75 * 12 * n_rows / n_cols))
    
    wntr_results=results.link['flowrate'].iloc[:-1,:]
    #junction_names = superjunctions.name.to_list()
    
    for i in range(n_rows * n_cols):
        if i < len(wn.pump_name_list):
            ax.flat[i].plot(wntr_results.index[start_ind_wn:]/3600,3600*Q_pump[:,i:i+1][start_ind_pd:], c='r', alpha=0.75)
            ax.flat[i].plot(wntr_results.index[start_ind_wn:]/3600,3600*wntr_results[wn.pump_name_list[i]][start_ind_wn:], c='0.3', linestyle = '--', alpha=0.75)
            ax.flat[i].set_title(f'Pump {wn.pump_name_list[i]}')
            #ax.flat[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
#            if (i % ax.shape[1]) == 0:
#                ax.flat[i].set_ylabel('Flow rate ($m^3/s$)')
#            if (i >= ax.shape[1] * (ax.shape[0] - 1)):
#                ax.flat[i].set_xlabel('Hour of Day')
        else:
            ax.flat[i].set_visible(False)
        
    ax.flat[0].legend(['Pipedream','EPANET'])
    plt.suptitle('Pump flows')
    plt.tight_layout()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plot pipedream and wntr PRV flows 
    n_superlinks = wn.num_valves
    n_cols = 4
    n_rows = min(6,int(np.ceil(wn.num_valves / n_cols)))
    if n_rows==0:
        n_rows = 1
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(12, 0.75 * 12 * n_rows / n_cols))
    
    wntr_results=results.link['flowrate'].iloc[:-1,:]
    #junction_names = superjunctions.name.to_list()
    
    for i in range(n_rows * n_cols):
        if i < len(wn.valve_name_list):
            ax.flat[i].plot(wntr_results.index[start_ind_wn:]/3600,3600*Q_prv[:,i:i+1][start_ind_pd:], c='r', alpha=0.75, label = 'Pipedream')
            ax.flat[i].plot(wntr_results.index[start_ind_wn:]/3600,3600*wntr_results[wn.valve_name_list[i]][start_ind_wn:], c='0.3', linestyle = '--', alpha=0.75, label = 'EPANET')
            ax.flat[i].set_title(f'{wn.get_link(wn.valve_name_list[i]).valve_type} {wn.valve_name_list[i]}')
            #ax.flat[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
#            if (i % ax.shape[1]) == 0:
#                ax.flat[i].set_ylabel('Flow rate ($m^3/s$)')
#            if (i >= ax.shape[1] * (ax.shape[0] - 1)):
#                ax.flat[i].set_xlabel('Hour of Day')
        else:
            ax.flat[i].set_visible(False)
        
    ax.flat[0].legend()
    plt.suptitle('Valve flows')
    plt.tight_layout()