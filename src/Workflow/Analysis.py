#%% ------------------------------- ###
###            0. Import            ###
### ------------------------------- ###

import gams
import pandas as pd
import numpy as np
from datetime import datetime
import platform
OS = platform.platform().split('-')[0]

import shutil
import os
import sys
import pickle
import configparser
import plotly.express as px
import plotly.graph_objects as go
from Functions.Formatting import newplot, set_style, stacked_bar
from Functions.GeneralHelperFunctions import symbol_to_df, filter_low_max, AntaresOutput
from Functions.antaresViz import stacked_plot

### 0.0 Settings
if not('SC' in locals()):
    try:
        # Try to read something from the command line
        SC = sys.argv[1]
    except:
        # Otherwise, read config from top level
        print('Reading SC from Config.ini..') 
        Config = configparser.ConfigParser()
        Config.read('Config.ini')
        SC = Config.get('RunMetaData', 'SC')

Config = configparser.ConfigParser()
Config.read('Workflow/MetaResults/%s_meta.ini'%SC)
SC_folder = Config.get('RunMetaData', 'SC_Folder')
USE_CAPCRED   = Config.getboolean('PostProcessing', 'Capacitycredit')
USE_H2CAPCRED   = Config.getboolean('PostProcessing', 'H2Capacitycredit')

# Analysis Settings
plotprofiles = 'n' # Choose whether to plot profiles or not
plotantaresViz = 'n'
plotPFcomparison = False
style = Config.get('Analysis', 'plot_style')
plot_all = Config.getboolean('Analysis', 'plot_all')
zip_files = Config.getboolean('Analysis', 'zip_files')
del_files = Config.getboolean('Analysis', 'del_files')

fc, plotly_theme = set_style(style)


    
# Years
Y = np.array(Config.get('RunMetaData', 'Y').split(',')).astype(int)
Y.sort()
Y = Y.astype(str)
ref_year = Config.getint('RunMetaData', 'ref_year')
gams_system_directory = Config.get('RunMetaData', 'gams_system_directory')

### 0.1 Working Directory
wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


### 0.2 Antares region mapping
with open(wk_dir + '/Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
    A2B_regi = pickle.load(f)
with open(wk_dir + '/Pre-Processing/Output/A2B_regi_h2.pkl', 'rb') as f:
    A2B_regi_h2 = pickle.load(f)

# Full antares region list
with open(wk_dir + '/Pre-Processing/Output/antreglist.pkl', 'rb') as f:
    ANTREGLIST = pickle.load(f)
    
### 0.3 Checking if running this script by itself, otherwise, specify scenario
if np.all(pd.Series(sys.argv).str.find('Analysis.py') == -1):
    test_mode = 'Y' # Set to N if you're running iterations
    print('\n----------------------------\n\nTest mode ON\n\nScenario: %s\n\n----------------------------\n'%SC)
else:
    test_mode = 'N'

### 0.4 Which results to import?
ant_out = pd.Series(os.listdir(wk_dir + '/Antares/output'))
ant_out = ant_out[ant_out.str.find(('eco-' + SC + '_iter').lower().replace('+',' ')) != -1].sort_values(ascending=False)

# Find iterations
iters = list(ant_out.str.split('_iter', expand=True).iloc[:,1].str.split('_y-',expand=True).iloc[:,0].astype(int)) 
iters = pd.Series(iters).unique()
iters.sort()
print('\nIterations as read from Antares output: %d'%len(iters))


#%% ------------------------------- ###
###         1. Annual Values        ###
### ------------------------------- ###

### 1.0 Plot design
figsize = (10,5)
# back_color = (32/255, 31/255, 30/255)
# xticks = [i for i in np.arange(iters[0], iters[-1]+1)]
 

### 1.1 Placeholders and useful data
obj = pd.DataFrame({})
Antobj = pd.DataFrame({})
cap = pd.DataFrame({})
cap_F = pd.DataFrame({})
eltrans = pd.DataFrame({})
h2trans = pd.DataFrame({})
dem = pd.DataFrame({})
pro = pd.DataFrame({})
proH2 = pd.DataFrame({})
emi = pd.DataFrame({})

# uniq_fuels = np.array(['biogas', 'biooil', 'coal', 'electric', 'fueloil', 'heat',
#        'hydrogen', 'lightoil', 'lignite', 'muniwaste', 'natgas',
#        'nuclear', 'straw', 'sun', 'wasteheat', 'water', 'wind',
#        'woodchips', 'woodpellets', 'woodwaste'], dtype=object)
mc_choice = 'mc-all' # MC year in Antares for generation results
for i in iters:
    ### 1.2 Load Balmorel Results
    print('Reading results from Balmorel/%s/model/MainResults_%s_Iter%d.gdx..'%(SC_folder, SC, i))
    ws = gams.GamsWorkspace(system_directory=gams_system_directory)  
    db = ws.add_database_from_gdx(wk_dir + "/Balmorel/%s/model/MainResults_%s_Iter%d.gdx"%(SC_folder, SC, i))

    ## Get objective function
    temp = symbol_to_df(db, 'OBJ_YCR', ['Y', 'C', 'R', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = i
    temp = temp.groupby(['Y', 'Var', 'Iter']).aggregate({'Value' : np.sum})
    obj = pd.concat((obj, temp))

    ## Get Generation & Storage Capacities
    temp = symbol_to_df(db, 'G_CAP_YCRAF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = i
    temp_F = temp[(temp.Tech != 'H2-STORAGE') & (temp.Tech != 'INTRASEASONAL-ELECT-STORAGE') & (temp.Tech != 'INTRASEASONAL-HEAT-STORAGE')]
    temp_F = temp_F.groupby(['Y', 'F', 'Iter']).aggregate({'Value' : np.sum})
    temp = temp.groupby(['Y', 'Tech', 'Iter']).aggregate({'Value' : np.sum})
    cap = pd.concat((cap, temp))
    cap_F = pd.concat((cap_F, temp_F))
    
    ## Get Electricity Transmission Capacities
    temp = symbol_to_df(db, 'X_CAP_YCR', ['Y', 'C', 'From', 'To', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = i
    temp = temp.groupby(['Y', 'To', 'Iter', 'From']).aggregate({'Value' : np.sum})
    eltrans = pd.concat((eltrans, temp))

    ## Get H2 Transmission Capacities
    try:
        temp = symbol_to_df(db, 'XH2_CAP_YCR', ['Y', 'C', 'From', 'To', 'Var', 'Unit', 'Value'])
        temp.loc[:, 'Iter'] = i
        temp = temp.groupby(['Y', 'To', 'Iter', 'From']).aggregate({'Value' : np.sum})
        h2trans = pd.concat((h2trans, temp))
    except ValueError:
        print('No hydrogen transmission')

    ## Get Demand
    temp = symbol_to_df(db, 'EL_DEMAND_YCR', ['Y', 'C', 'R', 'Var', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = i
    temp = temp.groupby(['Y', 'Var', 'Iter']).aggregate({'Value' : np.sum})
    dem = pd.concat((dem, temp))

    ## Get Production
    temp = symbol_to_df(db, 'PRO_YCRAGF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Unit', 'Value'])
    temp.loc[:, 'Iter'] = i
    temp.loc[:, 'Model'] = 'Balmorel'
    curt = symbol_to_df(db, 'CURT_YCRAGF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Unit', 'Value'])
    curt.loc[:, 'Iter'] = i
    curt.loc[:, 'Model'] = 'Balmorel'
    curt = curt.pivot_table(index=['Y', 'Model', 'R', 'F', 'Tech', 'Iter'],
                            values='Value',
                            aggfunc='sum')
    
    # Filter away hydrogen and electrolyser 
    temp2 = temp[(temp.Commodity == 'ELECTRICITY')].copy()
    temp = temp[(temp.Commodity == 'HYDROGEN')]
    temp2 = temp2.groupby(['Y', 'Model', 'R', 'F', 'Tech', 'Iter']).aggregate({'Value' : np.sum})
    temp = temp.groupby(['Y', 'Model', 'R', 'F', 'Tech', 'Iter']).aggregate({'Value' : np.sum})
    pro = pd.concat((pro, temp2))
    pro = pd.concat((pro, curt))
    curt = curt.reset_index()
    curt['Tech'] = 'Spilled'
    curt['F'] = 'Spilled'
    curt = curt.pivot_table(index=['Y', 'Model', 'R', 'F', 'Tech', 'Iter'],
                            values='Value',
                            aggfunc='sum')
    curt.loc[:, 'Value'] = -curt.loc[:, 'Value']
    pro = pd.concat((pro, curt))
    proH2 = pd.concat((proH2, temp))

    ## Get Emissions
    temp = symbol_to_df(db, 'EMI_YCRAG', ['Y', 'C', 'R', 'A', 'G', 'F', 'Tech', 'Unit', 'Value'])
    if len(temp) == 0:
        temp = pd.DataFrame(columns=['Y', 'C', 'R', 'A', 'G', 'F', 'Tech', 'Unit', 'Value'],
                            index=[0])
    temp.loc[:, 'Iter'] = i
    temp.loc[:, 'Model'] = 'Balmorel'
    temp = temp.groupby(['Model', 'Iter', 'Y', 'R']).aggregate({'Value' : np.sum})
    emi = pd.concat((emi, temp))
    
    ### 1.3 Load Antares Results
    for year in Y:
        
        if not(year == str(ref_year) and i != 0):
            ant_output = ant_out[ant_out.str.find(('eco-' + SC + '_iter%d_y-%s'%(i, year)).lower().replace('+', ' ')) != -1].values[0]
            print('Reading results from %s..'%ant_output)
            
            # Load class
            ant_res = AntaresOutput(ant_output)
            
            # Load Antares Costs
            try:
                ant_cost = pd.read_table(os.path.join('Antares/output', ant_output, 'annualSystemCost.txt'),
                        sep=' : ', header=None, engine='python')
                ant_cost['SC'] = SC
                ant_cost['Year'] = year
                ant_cost['Iter'] = i 
                Antobj = pd.concat((Antobj, ant_cost), ignore_index=True) 
            except:
                # Just a safeguard if i made an error
                print('Couldnt store Antares cost output')
                
            ## Electricity
            for area in A2B_regi.keys(): 
                try:
                    f = ant_res.load_area_results(area, 'details', 'annual', mc_choice)
                    
                    ## Thermal Generation
                    for col in [column for column in f.columns if not(column in [area, 'annual'])]:
                        tech = col.split('_')[0]
                        fuel = col.split('_')[1]
                        # print(tech, fuel)
                        # Save annual production
                        pro.loc[year, 'Antares', area, fuel, tech, i] = f[col].values[0]/1e6
                except FileNotFoundError:
                    # print('No thermal generation in area %s'%area)
                    pass

                f = ant_res.load_area_results(area, 'values', 'annual', mc_choice)
                
                ## CO2
                emi.loc['Antares', i, year, area] = f['CO2 EMIS.'].sum() / 1e3 # kton
                    
                ## VRE Generation
                translation = {'WIND' : 'WIND ONSHORE',
                            'SOLAR' : 'SOLAR PV'}
                for ren in ['WIND', 'SOLAR']:
                    pro.loc[year, 'Antares', area, ren, ren, i] = f[translation[ren]].values[0] / 1e6

                ## Spilled Energy (Mainly curtailment of VRE, but in principle thermal must-runs as well)
                spilled = f['SPIL. ENRG'].values[0]             
                pro.loc[year, 'Antares', area, 'Spilled', 'Spilled', i] = -spilled / 1e6
                
                ## Hydro
                # In area itself
                pro.loc[year, 'Antares', area, 'WATER', 'HYDRO-RESERVOIRS', i] = f.loc[0, 'H. STOR'] / 1e6
                pro.loc[year, 'Antares', area, 'WATER', 'HYDRO-RUN-OF-RIVER', i] = f.loc[0, 'H. ROR'] / 1e6
                
                ## Modelled as virtual units (missing 0_pump_open for 2_*_hydro_open, but doesn't interact with BZ)
                for hydro_area in ['1_PUMP_closed', '1_TURB_closed', '2_*_HYDRO_OPEN', '3_*_HYDRO_RES', '4_*_HYDRO_SWELL']:
                    try:
                        f = ant_res.load_link_results([hydro_area.replace('*', area), area], temporal='hourly', mc_year=mc_choice)
                        flow = f.loc[:, 'FLOW LIN.']
                        pro.loc[year, 'Antares', area, 'WATER', 'HYDRO-RESERVOIRS', i] += flow[flow > 0].sum() / 1e6  
                        
                    except FileNotFoundError:
                        # print('No connection between %s and %s'%(hydro_area.replace('*', area), area))                
                        pass
                    
                # Battery here
                try:
                    f = ant_res.load_link_results(['0_battery_turb', area], temporal='annual', mc_year=mc_choice)
                    pro.loc[year, 'Antares', area, 'ELECTRIC', 'BATTERY', i] = f.loc[0, 'FLOW LIN.'] / 1e6
                except FileNotFoundError:
                    # No battery to ITCO, e.g.
                    pass
            
            ## Hydrogen
            # for area in A2B_regi_h2.keys():
                
            #     # File
            #     temp = ant_res.load_area_results(area, temporal='annual', mc_year=mc_choice)
                
            #     ## Emissions
            #     emi.loc['Antares', i, year, area] = temp['CO2 EMIS.'].sum() / 1e3 # kton
                
            #     ## H2 Storages
            #     proH2.loc[year, 'Antares', area, 'HYDROGEN', 'Large-scale Storage', i] = temp.loc[0, 'H. STOR'] / 1e6

            #     temp = ant_res.load_link_results(['0_h2tank_turb', area], temporal='annual', mc_year=mc_choice)
            #     proH2.loc[year, 'Antares', area, 'HYDROGEN', 'Tank Storage', i] = temp.loc[0, 'FLOW LIN.'] / 1e6
                
            #     # Electrolyser
            #     temp = ant_res.load_link_results(['x_c3', area], temporal='annual', mc_year=mc_choice)
            #     proH2.loc[year, 'Antares', area, 'ELECTRIC', 'ELECTROLYZER', i] = temp.loc[0, 'FLOW LIN.'] / 1e6

            #     # SMR
            #     try:
            #         temp = ant_res.load_area_results(area, 'details', 'annual', mc_choice)
            #         for col in [column for column in temp.columns if not(column in [area, 'annual'])]:
            #             tech = col.split('_')[0]
            #             fuel = col.split('_')[1]
                        
            #             # Save annual production
            #             proH2.loc[year, 'Antares', area, fuel.upper(), tech.upper(), i] = temp[col].values[0]/1e6
            #     except FileNotFoundError:
            #         # No thermal generation
            #         pass
       
#%% Reset index for plotly plots and store pickle file with all dataframes

with open('Workflow/OverallResults/%s_results.pkl'%SC, 'wb') as f:
    pickle.dump({'obj' : obj,
                 'Aobj' : Antobj,
                'capT' : cap,
                'capF' : cap_F,
                'eltrans' : eltrans,
                'h2trans' : h2trans,
                'dem' : dem,
                'pro' : pro,
                'proh2' : proH2,
                'emi' : emi}, f)

obj.reset_index(inplace=True)
cap.reset_index(inplace=True)
cap_F.reset_index(inplace=True)
pro.reset_index(inplace=True)
proH2.reset_index(inplace=True)
eltrans.reset_index(inplace=True)
                             
### 1.4 System Costs
# Filter iterations or not
idx = filter_low_max(obj, 'Iter', plot_all)
fig = px.bar(obj[idx], x='Y', y='Value', color='Var', barmode='stack', facet_col='Iter')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - System Costs (M€)'%SC)
fig.show()
fig.write_html('Workflow/OverallResults/%s_SystemCosts.html'%SC)

### 1.5 Generation Capacities wrt. Technology
# Filter iterations or not
idx = filter_low_max(cap, 'Iter', plot_all)
idx = idx & (cap.Tech != 'H2-STORAGE') &\
    (cap.Tech != 'INTERSEASONAL-HEAT-STORAGE') &\
    (cap.Tech != 'INTRASEASONAL-HEAT-STORAGE') &\
    (cap.Tech != 'INTRASEASONAL-ELECT-STORAGE') &\
    (cap.Value > 1e-6)
fig = px.bar(cap[idx], x='Y', y='Value', color='Tech', barmode='stack', facet_col='Iter')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - Generation Capacity wrt. Tech (GW)'%SC)
fig.show()
fig.write_html('Workflow/OverallResults/%s_GenerationTechCapacities.html'%SC)

### 1.6 Generation Capacities wrt. Fuel
# Filter iterations or not
idx = filter_low_max(cap_F, 'Iter', plot_all)
idx = idx & (cap_F.Value > 1e-6)
fig = px.bar(cap_F[idx], x='Y', y='Value', color='F', barmode='stack', facet_col='Iter')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - Generation Capacity wrt. Fuel (GW)'%SC)
fig.show()
fig.write_html('Workflow/OverallResults/%s_GenerationFuelCapacities.html'%SC)

### 1.7 Generation wrt. Fuel
# Electricity
for model in ['Balmorel', 'Antares']:
    temp = pro[pro.Model == model].groupby(['Y', 'F', 'Iter']).aggregate({'Value' : np.sum})
    temp.reset_index(inplace=True)
    # temp = temp[~(temp.F == 'Spilled')]
    idx = filter_low_max(temp, 'Iter', plot_all)
    idx = idx
    fig = px.bar(temp[idx], x='Y', y='Value', color='F', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - Generation wrt. Fuel in %s (TWh)'%(SC, model))
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_%sGenerationFuel.html'%(SC, model))

# Hydrogen
for model in ['Balmorel', 'Antares']:
    temp = proH2[proH2.Model == model].groupby(['Y', 'F', 'Iter']).aggregate({'Value' : np.sum})
    temp.reset_index(inplace=True)
    # temp = temp[~(temp.F == 'Spilled')]
    idx = filter_low_max(temp, 'Iter', plot_all)
    idx = idx
    fig = px.bar(temp[idx], x='Y', y='Value', color='F', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - H2 Generation wrt. Fuel in %s (TWh)'%(SC, model))
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_%sH2GenerationFuel.html'%(SC, model))

### 1.8 Storage Capacities
# Filter iterations or not
idx = filter_low_max(cap, 'Iter', plot_all)
idx = idx & (cap.Value > 1e-6) & ((cap.Tech == 'H2-STORAGE') |\
    (cap.Tech == 'INTERSEASONAL-HEAT-STORAGE') |\
    (cap.Tech == 'INTRASEASONAL-HEAT-STORAGE') |\
    (cap.Tech == 'INTRASEASONAL-ELECT-STORAGE'))
fig = px.bar(cap[idx], x='Y', y='Value', color='Tech', barmode='stack', facet_col='Iter')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - Storage Capacity (GWh)'%SC)
fig.show()
fig.write_html('Workflow/OverallResults/%s_StorageCapacities.html'%SC)

### 1.9 Electricity Transmission Capacities
# Filter iterations or not
temp = eltrans.groupby(['Y', 'Iter', 'To']).aggregate({'Value' : np.sum}) # Account for double counting
# temp = eltrans.groupby(['Y', 'From', 'Iter', 'To']).aggregate({'Value' : lambda x: sum(x)/2}) # Account for double counting
temp.reset_index(inplace=True)
idx = filter_low_max(temp, 'Iter', plot_all)
idx = idx & (temp.Value > 1e-6)
fig = px.bar(temp[idx], x='Y', y='Value', color='To', barmode='stack', facet_col='Iter')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - 2x El. Transmission Capacity (GW)'%SC)
fig.show()
fig.write_html('Workflow/OverallResults/%s_ElTransCapacities.html'%SC)
   
### 1.10 Hydrogen Transmission Capacities
try:
    # Filter iterations or not
    temp = h2trans.groupby(['Y', 'Iter', 'To']).aggregate({'Value' : np.sum}) # Account for double counting
    # temp = eltrans.groupby(['Y', 'From', 'Iter', 'To']).aggregate({'Value' : lambda x: sum(x)/2}) # Account for double counting
    temp.reset_index(inplace=True)
    idx = filter_low_max(temp, 'Iter', plot_all)
    idx = idx & (temp.Value > 1e-6)
    fig = px.bar(temp[idx], x='Y', y='Value', color='To', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - 2x H2 Transmission Capacity (GW)'%SC)
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_H2TransCapacities.html'%SC)
except:
    pass
 
### 1.11 Electricity Demand
# Filter iterations or not
temp = dem
temp.reset_index(inplace=True)
idx = filter_low_max(temp, 'Iter', plot_all)
idx = idx & (temp.Value > 1e-6)
fig = px.bar(temp[idx], x='Y', y='Value', color='Var', barmode='stack', facet_col='Iter')
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - Electricity Demand (GWh)'%SC)
fig.show()
fig.write_html('Workflow/OverallResults/%s_ElecDemand.html'%SC)
 
 
### 1.12 Emissiongs
# Filter iterations or not
temp = emi.reset_index()
for model in temp.Model.unique():
    temp2 = temp[temp.Model == model]
    # temp.reset_index(inplace=True)
    idx = filter_low_max(temp2, 'Iter', plot_all)
    idx = idx & (temp2.Value > 1e-6)
    fig = px.bar(temp2[idx], x='Y', y='Value', color='R', barmode='stack', facet_col='Iter')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template=plotly_theme,title='%s - %s Emissions (ktonCO2)'%(SC, model))
    fig.show()
    fig.write_html('Workflow/OverallResults/%s_%sEmissions.html'%(SC, model))
 

#%% 1.12 Unserved Energy
# for carrier in ['Elec', 'H2']:
#     f = pd.read_csv('OverallResults/%s_%sNotServedMWh.csv'%(SC, carrier))
#     f.columns = ['Iter'] + list(f.columns[1:])
#     f.iloc[:, 1:] = f.iloc[:, 1:] / 1e3 # GWh
#     f['Iter'] = f['Iter'].astype(int)

#     fig, ax = newplot(figsize=figsize, fc=fc)
#     f.plot(x=['Y', 'Iter'], ax=ax, stacked=True, kind='bar', zorder=5)
#     ax.legend(loc='center', bbox_to_anchor=(.5, 1.25), ncol=3)
#     ax.set_ylabel('Unsupplied %s (GWh)'%carrier)
#     ax.set_xlabel('Iteration')
#     ax.set_title(year)
#     # ax.set_xticks(xticks)
#     fig.savefig('OverallResults/%s_Unserved%s.png'%(SC, carrier), 
#                 bbox_inches='tight', transparent=True)


#%% ------------------------------- ###
###         2. Pareto Front         ###
### ------------------------------- ###

## Read LOLD
fLOLD = pd.read_csv('Workflow/OverallResults/%s_LOLD.csv'%SC, index_col=0)


### 2.1 Save pareto front data
# PF = pd.DataFrame({})
# fig, ax = newplot(figsize=figsize, fc=fc)
# for year in Y:
#     temp = fLOLD[(fLOLD.Year == int(year)) & (fLOLD.Carrier == 'Electricity')].groupby(['Iter', 'Year']).aggregate({'Value (h)' : np.sum})
    
#     PF = pd.concat((PF, pd.DataFrame({'Iter' : np.arange(len(temp)),
#                     'SC' : [SC]*len(temp),
#                     'Year' : [year]*len(temp),
#                     'ElecLOLD_h' : temp.values[:,0],
#                     'SystemCost_MEUR' : obj[obj.Y == year].groupby(by=['Iter', 'Y']).aggregate({'Value' : np.sum}).values[:,0]})))

#     ax.plot(PF.ElecLOLD_h,
#             PF.SystemCost_MEUR,
#             'o')
#     ax.set_ylabel('System Costs (MEUR)')
#     ax.set_xlabel('Loss of Load Duration (h)')
#     ax.set_xscale('log')
#     fig.savefig('Workflow/OverallResults/%s_ParetoFront.png'%SC, 
#                 bbox_inches='tight', transparent=True)
# PF.to_csv('Workflow/OverallResults/%s_ParetoFront.csv'%SC, index=False)


#%% 2.2 Comparison between other pareto fronts
if plotPFcomparison:
    colours = [(0.5, .85, 0.5), (.85, 0.5, 0.5), (0.5, 0.5, .85), (.85, .5, .85)]
    
    for year in Y:
        # PF = pd.DataFrame()
        pfdata = pd.Series(os.listdir('Workflow/OverallResults'))[pd.Series(os.listdir('Workflow/OverallResults')).str.find('ParetoFront.csv') != -1]
        fig, ax = newplot(figsize=(7,3), fc=fc)
        
        i = 0
        for pf in pfdata: 
            if (pf != 'FictDemMarketValue_ParetoFront.csv'):
                pf_name = pf.replace('_ParetoFront.csv', '')    
                # PF = pd.concat((PF, pd.read_csv('Workflow/OverallResults/%s'%pf)))
                PF = pd.read_csv('Workflow/OverallResults/%s'%pf)
                PF = PF[PF.Year == int(year)]
                ax.plot(PF.ElecLOLD_h, PF.SystemCost_MEUR, 'o', label=pf.replace('_ParetoFront.csv', ''),
                        markersize=2, color=colours[i])
                i += 1
            
        ax.set_title(year)
        ax.legend()
        # ax.legend(('Capacity Credit', 'Fictive Demand + Market Value', 'Fictive Demand'))
        ax.set_ylabel('System Cost (MEUR)')
        ax.set_xlabel('Loss of Load Duration (h)')
        ax.set_xscale('log')
        fig.savefig('Workflow/OverallResults/PFComparison_%s.png'%year, transparent=True,
                    bbox_inches='tight')

#%% ------------------------------- ###
###           2. Profiles           ###
### ------------------------------- ###

### 2.0 Plot choices and design
figsize = (10,5)
back_color = (32/255, 31/255, 30/255)
iters_to_plot = [0] # Iterations to plot
mc_year = 0 # If = 0, it will choose the aggregated results
if mc_year == 0:
    mc_choice = 'mc-all'
    mc_string = 'All MCY'
else:
    mc_choice = '0000%d'%mc_year
    mc_choice = mc_choice[len(mc_choice) - 5:] # Adjust so the amount of digits are correct
    mc_choice = 'mc-ind/' + mc_choice
    mc_string = 'MCY%d'%mc_year
list_of_balances = {}
list_of_plots = {}

### 2.1 Antares Power Operation
elbalance = {}
if plotprofiles == 'y':
    for i in iters_to_plot:
        elbalance[i] = {}
        for year in Y:
            elbalance[i][year] = {}
            ### Load Antares Results
            ant_output = ant_out[ant_out.str.find(('eco-' + SC + '_iter%d_y-%s'%(i, year)).lower().replace('+', ' ')) != -1].values[0]

            for area in A2B_regi.keys(): 
                if not(area in ['ITCO']):
                    balance = pd.DataFrame({})        
                    fig, ax = newplot(figsize=figsize, fc=fc) # Figure for balance profile plot
                    temp = np.zeros(8736) # Placeholder for positive profiles
                    
                    ## Electrolyser consumption
                    f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                    '/economy/%s/links/%s - x_c3/values-hourly.txt'%(mc_choice, area),
                                    skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                    
                    prod = f['FLOW LIN.'].values
                    ax.fill_between(np.arange(0, 8736, 1), -prod, temp, label='Electrolysis') 
                    temp += -prod
                    balance['Electrolyser'] = -prod
                    ax.plot(-prod, 'r--', linewidth=1, label='Elec Load')
                    
                    
                    ## Battery
                    f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                    '/economy/%s/links/0_battery_pmp - %s/values-hourly.txt'%(mc_choice, area),
                                    skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                    
                    prod = f['FLOW LIN.'].values
                    ax.fill_between(np.arange(0, 8736, 1), temp, temp + prod, label='Battery') 
                    temp += prod
                    balance['Bat Charge'] = prod
                    
                    f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                            '/economy/%s/links/0_battery_turb - %s/values-hourly.txt'%(mc_choice, area),
                            skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                    
                    prod = f['FLOW LIN.'].values
                    ax.fill_between(np.arange(0, 8736, 1), temp, prod) 
                    temp += prod
                    balance['Bat Discharge'] = prod
                    
                    
                    ## 'Hydraulic'
                    hydro_ror = np.zeros(8736)
                    
                    # The production in the area itself
                    f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                    '/economy/%s/areas/%s/values-hourly.txt'%(mc_choice, area),
                                    skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                    
                    try:
                        hydro_ror += f['H. ROR'].values      
                    except KeyError:
                        print('No H. ROR in %s'%area)
                    try:
                        hydro_ror += f['H. STOR'].values
                    except KeyError:
                        print('No H. STOR in %s'%area)
                    ax.fill_between(np.arange(0, 8736, 1), temp, temp + hydro_ror, label='Hydro ROR')
                    temp += hydro_ror
                    balance['Hydro ROR'] = hydro_ror             
                    
                    
                    ## Pumped Storage
                    hydro_sto = np.zeros(8736)
                    try:
                        hydro_sto += f['PSP'].values
                        balance['Hydro PSP'] = hydro_sto
                    except KeyError:
                        print('No Pumped Hydro in %s'%area)
                    
                    # The production from pumped storage areas
                    hydro_sto = np.zeros(8736)
                    for RG in ['2_*_hydro_open', '1_pump_closed', '1_turb_closed']:
                        try:
                            f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                            '/economy/%s/links/%s - %s/values-hourly.txt'%(mc_choice, RG.replace('*', area), area),
                                            skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2

                            # Assumed convention
                            hydro_sto += f['FLOW LIN.'].values
                        except FileNotFoundError:
                            print('No hydro connection between:', '\n%s - %s'%(RG.replace('*', area), area))
                                                    
                    ax.fill_between(np.arange(0, 8736, 1), temp, temp + hydro_sto, label='Hydro STO')
                    temp += hydro_sto 
                    balance['Hydro Nodes'] = hydro_sto
                    
                    
                    ## Import/Export
                    for area2 in A2B_regi.keys():
                        if area2 != area:
                            try:
                                f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                                '/economy/%s/links/%s - %s/values-hourly.txt'%(mc_choice, area2, area),
                                                skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                                
                                prod = f['FLOW LIN.'].values # Importing = positive convention
                                ax.fill_between(np.arange(0, 8736, 1), temp, temp + prod, label=area2) 
                                temp += prod
                                balance[area2] = prod
                                
                            except FileNotFoundError:
                                f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                                '/economy/%s/links/%s - %s/values-hourly.txt'%(mc_choice, area, area2),
                                                skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                                
                                prod = -f['FLOW LIN.'].values # Importing = positive convention
                                ax.fill_between(np.arange(0, 8736, 1), temp, temp + prod, label=area2) 
                                temp += prod
                                balance[area2] = prod
                    
                    
                    ## Thermal Generation
                    fd = pd.read_table(wk_dir + '/Antares/output/' + ant_output + '/economy/%s/areas/%s/details-hourly.txt'%(mc_choice, area.lower()),
                                        skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2 (BUT 0 IN HPC!!! Should find it with datestring instead)
                    
                    # Sort away other results than production  
                    fd = fd.iloc[:, 5:]      
                    fd = fd.iloc[:, :int(len(fd.columns)/3)]
                    
                    for col in fd.columns:
                        
                        # Plot temporal profiles
                        prod = np.array(fd[col]).astype(float)
                        ax.fill_between(np.arange(0, 8736, 1), temp, temp + prod, label=col)   
                        temp += prod
                        balance[col] = prod
                
                
                    ## Renewable Generation
                    # Iterate through Antares areas
                    ren_gen = pd.DataFrame({})
                    
                    for ren in ['wind', 'solar']:
                        ren_gen[ren] = np.zeros(8736)
                        
                        # The production in the area itself
                        f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                        '/economy/%s/areas/%s/values-hourly.txt'%(mc_choice, area),
                                        skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                        
                        spilled = f['SPIL. ENRG'].values # Total curtailment (also thermal, so beware of double counting)
                        LOLE = f['UNSP. ENRG'].values
                        ren_gen[ren] += f[ren.upper()].values
                    
                    # The production from SRES areas
                    for RG in ['5_*_sres', '6_*_sres', '8_*_sres']:
                        # try:
                        f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                                        '/economy/%s/links/%s - %s/values-hourly.txt'%(mc_choice, RG.replace('*', area), area),
                                        skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2

                        # Assumed convention
                        ren_gen[ren] += f['FLOW LIN.'].values
                            
                        # except FileNotFoundError:
                        #     f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                        #                     '/economy/%s/links/%s - %s/values-hourly.txt'%(mc_choice, area, RG.replace('*', area)),
                        #                     skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2

                        #     # Assumed convention
                        #     ren_gen[ren] += -f['FLOW LIN.'].values
                    
                    # Assuming spill on VRE
                    ax.fill_between(np.arange(0, 8736, 1), temp, temp + ren_gen.sum(axis=1) - spilled, label='VRE')
                    temp += ren_gen.sum(axis=1) - spilled 
                    
                    balance['VRE'] = ren_gen.sum(axis=1)
                    balance['Spill'] = -spilled
                    balance['LOLE'] = LOLE

                    ## Load
                    fd = pd.read_table(wk_dir + '/Antares/output/' + ant_output + '/economy/%s/areas/%s/values-hourly.txt'%(mc_choice, area.lower()),
                                        skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2 (BUT 0 IN HPC!!! Should find it with datestring instead)
                    
                    ax.plot(fd['LOAD'], 'r-', linewidth=1, label='Load')
        

                    ## DSR
                    # Find connections
                    l = pd.Series(os.listdir(wk_dir + '/Antares/output/'+ant_output+'/economy/mc-all/links'))
                    l = l[l.str.find(area + ' - z_dsr') != -1]
                    balance['DSR'] = np.zeros(8736)
                    for link in l:
                        f = pd.read_table(wk_dir + '/Antares/output/' + ant_output +\
                        '/economy/%s/links/%s/values-hourly.txt'%(mc_choice, link),
                        skiprows=[0,1,2,3,5,6]) # At the moment the recent one is -2
                        balance['DSR'] -= f['FLOW LIN.']
                    
                    ### Balance Profile Plot Settings
                    ax.legend(loc='center', bbox_to_anchor=(.5, 1.15), ncol=5)
                    ax.set_xlim([0, 8736])
                    ax.set_title('Iteration %d, %s, %s'%(i, area, mc_string))

                    ## Save yearly balance
                    elbalance[i][year][area] = balance.sum()


    ### 2.2 Antares Power Operation Plot
    for i in iters_to_plot:
        for year in Y:
            fig, ax = newplot(figsize=figsize, fc=fc)
            total_bal = pd.DataFrame()
            for area in A2B_regi.keys():
                for ind in elbalance[i][year][area].index:
                    # If there's already a column
                    try:
                        total_bal.loc[year, ind] += elbalance[i][year][area][ind] / 1e6
                    # If there isn't
                    except KeyError:
                        total_bal.loc[year, ind] = elbalance[i][year][area][ind] / 1e6
            
            # Combine Hydro nodes
            total_bal.loc[year, 'Hydro'] = total_bal.loc[year, 'Hydro ROR'] + total_bal.loc[year, 'Hydro PSP'] + total_bal.loc[year, 'Hydro Nodes']
            total_bal = total_bal.drop(columns=['Hydro ROR', 'Hydro PSP', 'Hydro Nodes'])
            
            # Drop zeros and links
            total_bal = total_bal[total_bal != 0.0].dropna(axis=1)
            total_bal = total_bal.drop(columns=A2B_regi.keys())
            
            # Plot
            total_bal.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('Iteration %d'%i)

    # stacked_bar(temp/2, ['Iter'], ['To'], ax, {'zorder' : 5})
    # ax.legend(loc='center', bbox_to_anchor=(.5, 1.25), ncol=2)
    # ax.set_ylabel('H$_2$ Transmission Capacity (GW)')
    # ax.set_xlabel('Iteration')
    # ax.set_title(year)
    # # ax.set_xticks(xticks)
    # fig.savefig('Workflow/OverallResults/%s_TransmissionCapacitiesH2.png'%SC, 
    #             bbox_inches='tight', transparent=True)

    ## How to use graph objects

    fig_p = go.Figure()
    for col in balance.columns[1:]:
        fig_p.add_trace(go.Scatter(
                        x=np.arange(8736), y=balance[col],
                        hoverinfo='x+y',
                        mode='lines',
                        # line=dict(width=0.5, color='rgb(131, 90, 241)'),
                        stackgroup='one', # define stack group
                        name=col))
    fig_p.add_trace(go.Scatter(
                        x=np.arange(8736), y=fd['LOAD'],
                        mode='lines',
                        name='Exogenous Load'
                    ))
    fig_p.add_trace(go.Scatter(
                        x=np.arange(8736), y=balance['Electrolyser'],
                        mode='lines',
                        name='PtH2'
                    ))
    fig_p.add_trace(go.Scatter(
                        x=np.arange(8736), y=fd['LOAD']-balance['Electrolyser'],
                        mode='lines',
                        name='Net Load'
                    ))

    fig_p.update_xaxes(range=[0, 168])


#%% ------------------------------- ###
###          3. AntaresViz          ###
### ------------------------------- ###

if plotantaresViz == 'y':
    stacked_plot()
    
#%% ------------------------------- ###
###        4. Collect Results       ###
### ------------------------------- ###

### 4.1 Collect LOLD .csv's
l = pd.Series(os.listdir('Workflow/OverallResults'))
lElec = l[l.str.find('_ElecLOLD.csv') != -1]

# Elec
df = pd.DataFrame()
for file in lElec:
    temp = pd.read_csv('Workflow/OverallResults/' + file)
    temp.columns = ['SC', 'Iter', 'Year', 'Region', 'Value']
    temp['SC'] = file.split('_ElecLOLD')[0]
    df = df._append(temp, ignore_index=True)
df.to_csv('Workflow/OverallResults/ElecLOLD_AllSC.csv', index=False)

# H2
df = pd.DataFrame()
lH2 = l[l.str.find('_H2LOLD.csv') != -1]
for file in lH2:
    temp = pd.read_csv('Workflow/OverallResults/' + file)
    temp.columns = ['SC', 'Iter', 'Year', 'Region', 'Value']
    temp['SC'] = file.split('_H2LOLD')[0]
    df = df._append(temp, ignore_index=True)
df.to_csv('Workflow/OverallResults/H2LOLD_AllSC.csv', index=False)


### 4.2 Collect Antares System Costs
l = pd.Series(os.listdir('Antares/output'))

df = pd.DataFrame()
for file in l:
    if '_iter' in file and '_y-' in file:    
        try:
            temp = pd.read_table('Antares/output/%s/annualSystemCost.txt'%file, header=None)
            temp = float(temp.loc[0,0].lstrip('EXP : '))
            
            SCENARIO = file.split('eco-')[1]
            year = int(SCENARIO.split('_y-')[1])
            SCENARIO = SCENARIO.split('_y-')[0]
            iter = int(SCENARIO.split('_iter')[1])
            SCENARIO = SCENARIO.split('_iter')[0]
            
            df = df._append(pd.DataFrame({'SC' : SCENARIO, 'Y' : year, 'Iter' : iter, 'ObjCost' : temp},
                                        index=[0]), ignore_index=True)
        except FileNotFoundError:
            pass

df.to_csv('Workflow/OverallResults/AntaresSystemCost.csv', index=False)


### 4.3 Zip everything (linux commands)
now = datetime.now()
dt_string = now.strftime("%Y%m%d-%H%M")
zip_filename = 'Workflow/OverallResults/' + dt_string + '_%s_Results.zip'%SC
errors = False
results = ['AntaresEmissions.html',
            'BalmorelEmissions.html',
            'ElecDemand.html',
            'H2TransCapacities.html',
            'ElTransCapacities.html',
            'AntaresGenerationFuel.html',
            'AntaresH2GenerationFuel.html',
            'StorageCapacities.html',
            'BalmorelGenerationFuel.html',
            'BalmorelH2GenerationFuel.html',
            'GenerationFuelCapacities.html',
            'GenerationTechCapacities.html',
            'SystemCosts.html',
            'results.pkl',
            'ProcessTime.csv',
            'ElecNotServedMWh.csv',
            'H2NotServedMWh.csv',
            'MV.csv',
            'LOLD.csv']

if USE_CAPCRED:
    results.append('CC.pkl')
    results.append('ResMar.csv')
    if USE_H2CAPCRED:
        results.append('CCH2.pkl')

# Make sure to close open GAMS database before zipping
del db

if zip_files:
    
    # Zip Overall Results
    print('Zipping overall results..')
    for result in results:
        
        out = os.system('zip -r -q "%s" "Workflow/OverallResults/%s_%s"'%(zip_filename, SC, result)) 
        
        if out != 0:
            errors = True
            break
        
        if del_files:
            print('\nDeleting..')
            os.remove(os.path.join(wk_dir, 'Workflow/OverallResults', SC + '_' + result))
            
    # Zip configfile
    out = os.system('zip -r -q "%s" "Workflow/MetaResults/%s_meta.ini"'%(zip_filename, SC)) 
    if del_files:
        os.remove(os.path.join(wk_dir, 'Workflow/MetaResults', SC + '_meta.ini'))
    
    # Zip Balmorel Results
    if not(errors):
        for i in iters:
            balm_res = "Balmorel/%s/model/MainResults_%s_Iter%d.gdx"%(SC_folder, SC, i)
            print('Zipping %s..'%balm_res)
            
            out = os.system('zip -r -q "%s" "%s"'%(zip_filename, balm_res)) 

            if out != 0:
                errors = True
                break
            
            if del_files:
                print('\nDeleting..')
                os.remove(os.path.join(wk_dir, balm_res))
            
    # Zip Antares Results
    if not(errors):
        for ant_file in ant_out:
            print('Zipping %s..'%ant_file)
            out = os.system('zip -r -q "%s" "Antares/output/%s"'%(zip_filename, ant_file)) 
    
            if out != 0:
                errors = True
                break
            
            if del_files:
                print('\nDeleting..')
                shutil.rmtree(os.path.join(wk_dir, 'Antares/output', ant_file))
            