'''
This script extracts characteristics of the INP file
to be supplied to the pipedream_simulation script
'''
import numpy as np
import pandas as pd
import wntr
from scipy.optimize import curve_fit
import itertools

def pump_func(x, a, b):
    return a - b * x**2

def pattern_extrapolate(pattern, duration, pat_ts, dur_ts):
    # if simulation time step is larger than pattern time step
    duration = duration*3600/dur_ts
    pattern = list(pattern)
    if pat_ts > dur_ts:
        
        # match pattern time step to duration time step
        mult = int(pat_ts/dur_ts)
        #pattern = pattern * mult
        pattern = list(itertools.chain.from_iterable(itertools.repeat(x, mult) for x in pattern))
        #print(pattern)
        
        # match pattern duration to simulation duration
        if len(pattern) > duration:
            pattern = pattern[:duration]
        elif len(pattern) < duration:
            num_repetitions = int(duration/len(pattern))
            extras = duration % len(pattern)
            pattern_new = pattern.copy()
            for i in range(num_repetitions-1):
                pattern_new+=pattern
            if extras!=0:
                pattern_new.extend(pattern[:extras])
            pattern = pattern_new
            
    elif pat_ts < dur_ts:
        
        mult = int(dur_ts/pat_ts)
        pattern = pattern[::mult]
        
        if len(pattern) > duration:
            pattern = pattern[:duration]
        elif len(pattern) < duration:
            num_repetitions = int(duration/len(pattern))
            extras = duration % len(pattern)
            pattern_new = pattern.copy()
            for i in range(num_repetitions-1):
                pattern_new+=pattern
            if extras!=0:
                pattern_new.append(pattern[:extras])
            pattern = pattern_new
            
    return pattern

def wntr_2_pd(wn,t_run, dt):

    ######################################################################Create superjunctions dataframe from junctions
        
    superjunctions_m = pd.DataFrame(columns = ['name','id','z_inv','h_0','bc','storage','a','b','c',
                                               'max_depth','map_x','map_y','dem','tank','pat', 'tank_min', 'tank_max'], dtype=object)
    superjunctions_m['name'] = wn.node_name_list
    superjunctions_m['id'] = np.arange(0,len(wn.node_name_list),1)
    
    junction_names = wn.node_name_list
    
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
        
    #Extract values for 
    base_dem = np.zeros(len(junction_names))
    ele = np.zeros(len(junction_names))
    x = np.zeros(len(junction_names))
    y = np.zeros(len(junction_names))
    initial_head = np.zeros(len(junction_names))
    bc = [False]*len(junction_names)
    c = np.zeros(len(junction_names))
    dem = np.zeros(len(junction_names))
    tank = [False]*len(junction_names)
    pat = [None]*len(junction_names)
    has_tanks = 0
    num_tanks = 0
    tank_min = [-np.inf]*len(junction_names)
    tank_max = [np.inf]*len(junction_names)
    tank_dict = {}
    max_dep = np.zeros(len(junction_names))
    pumps= {}
    orifices = {}
    
    valve_info = {}
    for valve_name, valve in wn.valves():
        valve_info[valve_name] = {'start': valve.start_node_name, 
                                  'end': valve.end_node_name,
                                  'setting': valve.setting + wn.get_node(valve.end_node_name).elevation}
    
    for i in range(len(junction_names)):
        
        junction = wn.get_node(junction_names[i])
        j_type = junction.node_type
        
        if j_type == 'Junction':
            
            base_dem[i] = junction.base_demand
            ele[i] = junction.elevation
            # for valve_name in wn.valve_name_list:
            #     if junction_names[i] == valve_info[valve_name]['end']:
            #         ele[i] = valve_info[valve_name]['setting']-0.2
            if junction_names[i] == 'FP2_NU': # this node has negative pressures in the NWC model
                ele[i] = min(results.node['head'].loc[:,'FP2_NU'])
            initial_head[i] = 0
            bc[i] = False
            c[i] = 1e-5
            tank[i] = False
            dem[i] = junction.base_demand
            if junction.demand_timeseries_list[0].pattern != None:
                pat[i] = junction.demand_timeseries_list[0].pattern.name
            else:
                pat[i] = '-'
            max_dep[i] = np.inf
        
        if j_type == 'Reservoir':
            base_dem[i] = 0
#            if junction_names[i] == 'AM9001':
#                ele[i] = min(results.node['head'].loc[:,'AM9001'])  #319.1256 # to negate the effects of negative pressures in the model
#            else:
#                ele[i] = min(results.node['head'].loc[:,'FP9001']) # 331.9272
            ele[i] = min(results.node['head'].loc[:,junction_names[i]])
            initial_head[i] =0
            bc[i] = False
            c[i] =  10**(9) #1
            tank[i] = False
            dem[i] = 0
            pat[i] = '-' 
            max_dep[i] = np.inf
        
        if j_type == 'Tank':
            has_tanks = 1
            base_dem[i] = 0
            ele[i] = junction.elevation 
            initial_head[i] = junction.init_level 
            bc[i] = False
            c[i] = (junction.diameter/2)**2*np.pi
            tank[i] = True
            dem[i] = 0
            pat[i] = '-'
            tank_min[i] = junction.elevation + junction.min_level
            tank_max[i] = junction.elevation + junction.max_level
            num_tanks += 1
            max_dep[i] = junction.max_level #np.inf
            
        x[i],y[i]=junction.coordinates
    
    superjunctions_m['z_inv'] = ele
    superjunctions_m['h_0'] = initial_head
    superjunctions_m['bc'] = bc
    superjunctions_m['storage'] = 'functional'
    superjunctions_m['a'] = 0
    superjunctions_m['b'] = 0
    superjunctions_m['c'] = c
    superjunctions_m['max_depth'] = max_dep
    superjunctions_m['map_x'] = x
    superjunctions_m['map_y'] = y
    superjunctions_m['tank'] = tank
    superjunctions_m['dem'] = dem
    superjunctions_m['pat'] = pat
    superjunctions_m['tank_max'] = tank_max
    superjunctions_m['tank_min'] = tank_min
    if pat == None:
        superjunctions_m['pat'] = '-'
    
    ######################################################################Create superlinks dataframe from links
    
    links = wn.pipe_name_list 
    
    superlinks_m=pd.DataFrame(columns=['name','id',	'sj_0',	'sj_1',	'in_offset','out_offset',
                                       'dx','roughness','shape','g1',	'g2','g3','g4','Q_0',
                                       'h_0','ctrl','A_s','A_c','C', 'dx_uk', 'dx_dk', 'C_uk', 'C_dk', 
                                       'friction_method'], dtype=object)
    superlinks_m['name']=links
    superlinks_m['id']=np.arange(0,len(links),1)
    
    start_node=[None]*len(links)
    end_node=[None]*len(links)
    
    start_node_ind=[None]*len(links)
    end_node_ind=[None]*len(links)
    
    length=np.zeros(len(links))
    diam=np.zeros(len(links))
    rough=np.zeros(len(links))
    
    for i in range(len(links)):
        link=wn.get_link(links[i])
        start_node[i]=link.start_node.name
        end_node[i]=link.end_node.name
        diam[i]=link.diameter
        start_node_ind[i]=int(list(superjunctions_m['name']).index(start_node[i]))
        end_node_ind[i]=int(list(superjunctions_m['name']).index(end_node[i]))
        if links[i] in wn.valve_name_list:
            rough[i] = 100
            length[i] = 25.4
        else:
            rough[i] = link.roughness
            length[i] = link.length
    
    
    superlinks_m['dx'] = length
    superlinks_m['roughness'] = rough
    superlinks_m['shape'] = 'force_main'
    superlinks_m['g1'] = diam
    superlinks_m['g2'] = 0.00001
    superlinks_m['g3'] = 0
    superlinks_m['g4'] = 0
    superlinks_m['Q_0'] = 0
    superlinks_m['h_0'] = diam
    superlinks_m['ctrl'] = False
    superlinks_m['A_s'] = 0.001
    superlinks_m['A_c'] = 0
    superlinks_m['C'] = 0
    superlinks_m['in_offset'] = 0
    superlinks_m['out_offset'] = 0    
    superlinks_m['sj_0'] = start_node_ind
    superlinks_m['sj_1'] = end_node_ind
    superlinks_m['dx_uk'] = 0
    superlinks_m['dx_dk'] = 0
    superlinks_m['C_uk'] = 0
    superlinks_m['C_dk'] = 0
    superlinks_m['friction_method'] = 'hw'  
    
    # if there are tanks then do a bunch of stuff ##################################################################
    orifices_count = 0
    #If there is a tank, create the orifices variable
    if has_tanks==1:
        orifices=pd.DataFrame(columns=['name','id','sj_0','sj_1','A','orientation','z_o','y_max','C'], dtype=object)
        
        for j in range(num_tanks):
            superjunctions_new=pd.DataFrame(columns=['name','id','z_inv','h_0','bc','storage','a','b','c',
                                                     'max_depth','map_x','map_y','dem','tank','pat', 'tank_min', 'tank_max'],
                                            index=[superjunctions_m['id'].max()+1], dtype=object)
                
            
            #For each tank, create another node
            tank_junctions=superjunctions_m[superjunctions_m['tank']==True]
            
            #For each tank node, get the nodes on the other side of the links
            r,c=tank_junctions.shape
            
            ####################
            tank_row=tank_junctions.iloc[j,:]
            
            #Find which nodes are connected to that tank based on the superlinks DF
            
            tank_ind=tank_row['id']
            tank_dict[superjunctions_m['name'][tank_ind]] = {'id': tank_row['id'],
                                                             'min level': orifices_count}
            
            orifices_count +=1 
            
            #Find in the superlinks where that node exists
            con1=superlinks_m['sj_0']==tank_ind
            con2=superlinks_m['sj_1']==tank_ind
            tank_connected_links= superlinks_m[con1 | con2]
            
            link_index=tank_connected_links.index.to_list()
            
            #For each row of tank_connected_links create a new node with the invert of the 
            #not-tank node
            r_tc,c_tc=tank_connected_links.shape
    
            
            # another for loop for r_tc
            #Get the node that isn't the tank
            #Check the sj_0 node and if that is the same as the tank then get the other one
            other_node=tank_connected_links['sj_0'].to_numpy()[0]
            connect_col='sj_1'
            
            #If the tank is the sj_0 node then take the sj_1 node
            if other_node==tank_ind:
                other_node=tank_connected_links['sj_1'].to_numpy()[0]
                connect_col='sj_0'
                
            #Add a new node to the superjunctions in the same location as tank node and
            #and with invert of other node
            superjunctions_new['name']='i'+superjunctions_m['name'][tank_ind]
            superjunctions_new['id']=superjunctions_m['id'].max()+1
            superjunctions_new['z_inv']=superjunctions_m['z_inv'][superjunctions_m['id']==other_node].values[0]
            superjunctions_new['h_0']=0
            superjunctions_new['bc']=False
            superjunctions_new['storage']='functional'
            superjunctions_new['a']=0
            superjunctions_new['b']=0
            superjunctions_new['c']=1e-5
            superjunctions_new['max_depth']= wn.get_node(superjunctions_m['name'][tank_ind]).elevation + wn.get_node(superjunctions_m['name'][tank_ind]).max_level  #np.inf
            superjunctions_new['map_x']=superjunctions_m['map_x'][superjunctions_m['id']==tank_ind].values[0]
            superjunctions_new['map_y']=superjunctions_m['map_y'][superjunctions_m['id']==tank_ind].values[0]
            superjunctions_new['tank']=False
            superjunctions_new['dem']=0
            superjunctions_new['pat']='-'
            superjunctions_new['tank_min']=-np.inf
            superjunctions_new['tank_max']=np.inf
            
            # #Append superjunctions_new to the existing superjunctions dataframe
            superjunctions_m=superjunctions_m.append(superjunctions_new)
            
            #Connect the link to the new node instead of the tank
            superlinks_m.loc[link_index[0],connect_col]=superjunctions_new['id'].values[0]
            
            #Create an orifice connecting the existing tank to the new node    
            orifices_row=pd.DataFrame(columns=['name','id','sj_0','sj_1','A','orientation','z_o','y_max','C'],index=[j], dtype=object)
            orifices_row['name']='o'+superjunctions_m['name'][tank_ind]
            orifices_row['id']=j
            orifices_row['sj_0']=tank_ind
            orifices_row['sj_1']=superjunctions_m['id'].max()
            orifices_row['A']=0.164
            orifices_row['orientation']='side'
            orifices_row['z_o']=0
            orifices_row['y_max']=superlinks_m['g1'][link_index].values[0]
            orifices_row['C']=0.67
            
            orifices=orifices.append(orifices_row)
   
#        orifices = orifices.astype(orifices_row.dtypes.to_dict())
  
    
    #####################################################################Create PRVS dataframe 
       
    prvs=pd.DataFrame(columns=['name','id','sj_0','sj_1','A','orientation','z_o','y_max','C_active', 'C_open', 'Hset'], dtype=object)
    
    for valve_name, valve in wn.valves():
        
        if valve.valve_type == 'PRV':
        
            valve_index = wn.valve_name_list.index(valve_name)
            #Create an orifice connecting the existing tank to the new node    
            prv_row=pd.DataFrame(columns=['name','id','sj_0','sj_1','A','orientation','z_o','y_max','C_active', 'C_open', 'Hset'],index=[valve_index], dtype=object)
            prv_row['name']=valve_name
            prv_row['id']=valve_index
            prv_row['sj_0']=superjunctions_m['id'][superjunctions_m['name']==valve.start_node_name].values[0]
            prv_row['sj_1']=superjunctions_m['id'][superjunctions_m['name']==valve.end_node_name].values[0]
            prv_row['A']=0.164
            prv_row['orientation']='side'
            prv_row['z_o']=0
            prv_row['y_max']= 0.1 # superlinks_m['g1'][link_index].values[0] ## 
            prv_row['C_active']=0.67
            prv_row['C_open']=0.67
            if valve_name == '~@RV-1':
                prv_row['C_active'] = 0.3
            if valve_name == '33':
                prv_row['C_open'] = 0.55
            prv_row['Hset']= wn.get_node(valve.end_node_name).elevation + valve.setting
            
            prvs=prvs.append(prv_row)

    
    #If there is a pump, create the pumps variable
    if wn.num_pumps > 0:
        
        pumps=pd.DataFrame(columns=['name','id','sj_0','sj_1','z_p','dH_min','dH_max','a_p','b_p','c_p'], dtype=object)
        
        for j in range(wn.num_pumps):
            
            pump_name = wn.pump_name_list[j]
            pump = wn.get_link(pump_name)
            
            if len(pump.get_pump_curve().points) == 1:
                A1, B1 = pump.get_pump_curve().points[0][0], pump.get_pump_curve().points[0][1]
                A2, B2 = 0, 4*B1/3
                A3, B3 = 2*A1, 0
                xdata = [A1, A2, A3]
                ydata = [B1, B2, B3]

#            
            elif len(pump.get_pump_curve().points) == 3:
                A1, B1 = pump.get_pump_curve().points[0][0], pump.get_pump_curve().points[0][1]
                A2, B2 = pump.get_pump_curve().points[1][0], pump.get_pump_curve().points[1][1]
                A3, B3 = pump.get_pump_curve().points[2][0], pump.get_pump_curve().points[2][1]
                xdata = [A1, A2, A3]
                ydata = [B1, B2, B3]
            
            else:
                xdata = []
                ydata = []
                for k in range(len(pump.get_pump_curve().points)):
                    xdata.append(pump.get_pump_curve().points[k][0])
                    ydata.append(pump.get_pump_curve().points[k][1])

            popt, pcov = curve_fit(pump_func, xdata, ydata)
            A, B = popt[0], popt[1]
            
            pumps_row=pd.DataFrame(columns=['name','id','sj_0','sj_1','z_p','dH_min','dH_max','a_p','b_p','c_p'],index=[j], dtype=object)
            pumps_row['name']=pump_name
            pumps_row['id']=j
            pumps_row['sj_0']=superjunctions_m['id'][superjunctions_m['name']==pump.start_node_name].values[0]
            pumps_row['sj_1']=superjunctions_m['id'][superjunctions_m['name']==pump.end_node_name].values[0]
            pumps_row['z_p']=0.
            pumps_row['dH_min']=0
            pumps_row['dH_max']=A
            pumps_row['a_p']=A
            pumps_row['b_p']=B
            pumps_row['c_p']=2

            pumps=pumps.append(pumps_row)
        
        
        #Convert the data types of orifices/pumps to orifices_row/pumps_row to avoid problems later

        pumps = pumps.astype(pumps_row.dtypes.to_dict())
            #################
    
    superjunctions = superjunctions_m
    superlinks = superlinks_m
    
    #Create H_BC
    # Modify for PRVs
    H_bc=superjunctions['z_inv'].copy().to_numpy()
    bc_val=superjunctions['bc'].values
    flip_val=[not elem for elem in bc_val]
    H_bc[flip_val]=0
    
    
    #Give the base demands for each node
    # Constant demand input (cms)
    Q_in = -superjunctions['dem'].to_numpy()
    
    #Get the multipliers from the wntr model
    #Get the unique patterns from superjunctions
    pats=superjunctions['pat'].to_list()
    pats=list(np.unique(pats))
    pats.remove('-')
    
    mult=[]
    for i in range(len(pats)):
        
        pat=wn.get_pattern(pats[i])
        mult_orig=pat.multipliers

        mult_loop=[]
        if len(mult_orig)<t_run:
            while len(mult_loop)<t_run:
                mult_loop=np.hstack((mult_loop,mult_orig)) 
                
        if len(mult_orig)>=t_run:
            mult_loop=mult_orig #[:t_run]
        mult_loop=np.array(mult_loop).reshape(-1,1)
        if i==0:
            mult=mult_loop
        if i!=0:
            mult=np.hstack((mult,mult_loop))
    
    mult_df = pd.DataFrame(mult,columns=pats)
    
    # In[] Extract control rules and store in a dictionary
    
    time_controls_dict = {}
    event_controls_dict = {}
    time_controls_compiled = {}
    events_controls_pairs = {}
    time_controls_link_list = []
    event_controls_link_list = []
    
    for i in range(len(wn.control_name_list)):
        pattern = "'(.*?)'" 
        cont = wn.get_control(wn.control_name_list[i])
        cont_str = str(cont)
        contr_list = cont_str.split()
       
        # time-based controls
        if 'TIME' in cont_str:
            link_name  = contr_list[7]      # link name
            sim_time = contr_list[4]                                            # time
            stat = contr_list[8]                          # status? setting?
            stat_val = contr_list[10]
                       
            ctrl_name = 'Control {}'.format(i)
            time_controls_dict[ctrl_name] = {'Link': link_name, 
                                             'Time': sim_time, 
                                             'Stat': stat, 
                                             'Stat val': stat_val}
            if link_name not in time_controls_link_list:
                time_controls_link_list.append(link_name)
            
        # event-based controls
        else:
            node_name = contr_list[2]
            link_name = contr_list[8]
            node_level = contr_list[5]
            sense = contr_list[4]
            stat = contr_list[9]
            stat_val = contr_list[11]
            
            event_controls_dict['Control {}'.format(i)] = {'Link': link_name, 
                                                           'Node': node_name, 
                                                           'Sense': sense, 
                                                           'Level': node_level, 
                                                           'Stat': stat, 
                                                           'Stat val': stat_val}
            if link_name not in event_controls_link_list:
               event_controls_link_list.append(link_name)
     

    # store all time controls in one dict
    for link in time_controls_link_list:
        times_list_stat = [1- int(results.link['status'].loc[0,link])]
        times_list_time = [0]
    
        for key in list(time_controls_dict.keys()):
            if time_controls_dict[key]['Link'] == link:
                dt_list = time_controls_dict[key]['Time'].split(":")
                dt_time = int(dt_list[0])*3600 + int(dt_list[1])*60 + int(dt_list[2])
                time_cont = float(dt_time)/wn.options.time.report_timestep
                if time_cont <= t_run:
                    if time_controls_dict[key]['Stat'] == 'status' or time_controls_dict[key]['Stat'] == 'STATUS':
                        if time_controls_dict[key]['Stat val'] == 'Open' or time_controls_dict[key]['Stat val'] == 'OPEN':  # check if other stat vals are possible
                            times_list_stat.append(0)
                        else:
                            times_list_stat.append(1)
                    times_list_time.append(int(time_cont))  
                    
        stat_array = np.zeros((t_run))
                                                             
        for i in range(len(times_list_stat)-1):
            stat_array[times_list_time[i]:times_list_time[i+1]] = times_list_stat[i]
        stat_array[times_list_time[-1]:] = times_list_stat[-1]
        time_controls_compiled[link] = {'Status':stat_array}
    
    # store coupled event-based controls together 
    ev_pair_count = 0
    ev_pair_list = []
    
    above_list = ['>', 'ABOVE']
    below_list = ['<', 'BELOW']
    closed_list = ['Closed', 'CLOSED']
    open_list = ['OPEN', 'Opened', 'Open']
    
    for key in list(event_controls_dict.keys()):
        for key_2 in list(event_controls_dict.keys()):
            if key != key_2:
                
                dict1 = event_controls_dict[key]
                dict2 = event_controls_dict[key_2]
                
                link, node = dict1['Link'], dict1['Node']
                link_2, node_2 = dict2['Link'], dict2['Node']
            
                if link == link_2 and node == node_2:
                    if (link,node) not in ev_pair_list:
                        # add other possibilites? setting for each kind of valve
                        
                        # if status == link status
                        if dict1['Stat'] == 'status' or dict1['Stat'] == 'STATUS':
                            
                            val1 = wn.get_node(node).elevation + float(dict1['Level'])
                            val2 = wn.get_node(node).elevation + float(dict2['Level'])
                            init = str(wn.get_link(link).initial_status) # results.link['status'].loc[0, link]
                            if init == 'Closed' or init :
                                init_stat =  1
                            if init == 'Open':
                                init_stat = 0
                            
                            if dict1['Sense'] in above_list and dict1['Stat val'] in closed_list:
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val1,
                                                                        'Lower lim': val2,
                                                                        'Upper lim stat': 0, # this is the value that will go to y
                                                                        'Lower lim stat': 1,
                                                                        'Link initial status': init_stat} # this is the value that will go to y0
                                
                            if dict1['Sense'] in above_list and dict1['Stat val'] in open_list: #unlikely
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val1,
                                                                        'Lower lim': val2,
                                                                        'Upper lim stat': 1, # this is the value that will go to y
                                                                        'Lower lim stat': 0,
                                                                        'Link initial status': init_stat} # this is the value that will go to y0
                                
                            if dict1['Sense'] in below_list and dict1['Stat val'] in closed_list: #unlikely
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val2,
                                                                        'Lower lim': val1,
                                                                        'Upper lim stat': 1, # this is the value that will go to y
                                                                        'Lower lim stat': 0,
                                                                        'Link initial status': init_stat}
                                
                            if dict1['Sense'] in below_list and dict1['Stat val'] in open_list: 
                                events_controls_pairs[ev_pair_count] = {'Link': link,
                                                                        'Node': node,
                                                                        'Upper lim': val2,
                                                                        'Lower lim': val1,
                                                                        'Upper lim stat': 0, # this is the value that will go to y
                                                                        'Lower lim stat': 1,
                                                                        'Link initial status': init_stat}
                                
                        ev_pair_count += 1
                        ev_pair_list.append((link,node))
    
       
    superjunctions = superjunctions.rename(columns={'dem' : 'base_demand', 'pat' : 'demand_pattern'})
    
    if isinstance(pumps, dict):
        if not pumps:
            pumps = None
    if isinstance(orifices, dict):
        if not orifices:
            orifices = None
    if isinstance(prvs, dict):
        if not prvs:
            prvs = None
            
    return superjunctions, superlinks, orifices, pumps, prvs, H_bc, Q_in, pats, mult_df, tank_min, tank_max, tank_dict, time_controls_compiled, events_controls_pairs
