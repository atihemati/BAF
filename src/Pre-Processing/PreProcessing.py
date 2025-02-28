"""
Created on 09-06-2023

@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###       0. Script Settings        ###
### ------------------------------- ###

import click
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('.')
from Workflow.Functions.Formatting import newplot, nested_dict_to_df
from Workflow.Functions.GeneralHelperFunctions import IncFile

import pickle
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
import configparser

# Load geojson's
try:
    import geopandas as gpd
    p = 'Pre-Processing/Data/Balmorelgeojson'
    
    if not('balmmap' in locals()):
        balmmap = gpd.read_file('Pre-Processing/Data/231206 AntBalmMap.geojson')
    plot_maps = True
        
except ModuleNotFoundError:
    plot_maps = False
    print('Geopandas not installed. Will not be able to plot maps')
    balmmap = pd.DataFrame({'id' : 'a'},index=[0])

@click.group()
@click.pass_context
def CLI(ctx):
    """The command line interface for pre-processing stuff"""

    ### 0.0 Load configuration file
    Config = configparser.ConfigParser()
    Config.read('Config.ini')
    UseAntaresData = Config.get('PeriProcessing', 'UseAntaresData').lower() == 'true' 
        
    style = 'report'

    if style == 'report':
        plt.style.use('default')
        fc = 'white'
    elif style == 'ppt':
        plt.style.use('dark_background')
        fc = 'none'

    ### 0.1 Assumptions
    balm_flh_type = 'FLH'   # Calculate FLH from XXX_VAR_T.inc or XXXFLH.inc file? Select 'VAR_T' or 'FLH'
    normalise = True        # Normalise when calculating FLH from BALM timeseries? 
    stoch_years = 35        # Amount of stochastic years
    elec_loss = 0.95        # See Murcia, Juan Pablo, Matti Juhani Koivisto, Graziela Luzia, Bjarke T. Olsen, Andrea N. Hahmann, Poul Ejnar Sørensen, and Magnus Als. “Validation of European-Scale Simulated Wind Speed and Wind Generation Time Series.” Applied Energy 305 (January 1, 2022): 117794. https://doi.org/10.1016/j.apenergy.2021.117794.
    year = 2050             # The year (analysing just one year for creating weights of electricity demand for different spatial resolutions)

#%% ------------------------------- ###
###      1. Hardcoded Mappings      ###
### ------------------------------- ###

@CLI.command()
@click.pass_context
def generate_mapping(ctx):
    """Generates spatial and technological mappings between Balmorel and Antares"""
    
    ### 1.1 Regions for A2B2A Mapping
    # Regions for VRE Mapping
    B2A_regi = {
                'CH':['CH'],
                'DE':['DE'],
                'FR':['FR']
                }

    A2B_regi = {
                'CH':['CH'],
                'DE':['DE'],
                'FR':['FR']
                }


    ## Save defined regions
    with open('Pre-Processing/Output/B2A_regi.pkl', 'wb') as f:
        pickle.dump(B2A_regi, f)
    with open('Pre-Processing/Output/A2B_regi.pkl', 'wb') as f:
        pickle.dump(A2B_regi, f)
        
        
    ### 1.2 Other useful dictionaries
    # Fuel
    B2A_fuel = {'SUN' : 'solar',
                'WIND' : 'wind',
                'BIOGAS' : 'biogas'}
    A2B_fuel = {B2A_fuel[k] : k for k in B2A_fuel.keys()}

    # Balmorel Technologies (rough combustion CO2 estimates hardcoded for now)
    kgGJ2tonMWh = 3.6 / 1e3 # Conversion from kg/GJ to ton/MWh
    BalmTechs = {'CHP-BACK-PRESSURE' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                        'WOODCHIPS' : {'CO2' : 0},  
                                        'WOODPELLETS' : {'CO2' : 0},  
                                        'STRAW' : {'CO2' : 0}, 
                                        'BIOGAS' : {'CO2' : 0}, 
                                        'MUNIWASTE' : {'CO2' : 0},
                                        'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                        'FUELOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                        'LIGHTOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                        'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh}},
                'CHP-EXTRACTION' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                    'WOODCHIPS' : {'CO2' : 0},  
                                    'WOODPELLETS' : {'CO2' : 0}, 
                                    'BIOGAS' : {'CO2' : 0}, 
                                    'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                    'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh},
                                    'MUNIWASTE' : {'CO2' : 0},
                                    'WOODWASTE' : {'CO2' : 0},
                                    'WOOD' : {'CO2' : 0},
                                    'PEAT' : {'CO2' : 0},
                                    'STRAW' : {'CO2' : 0}},
                'CONDENSING' : {'BIOGAS' : {'CO2' : 0}, 
                                'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                'FUELOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                'LIGHTOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh},
                                'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                'WOODCHIPS' : {'CO2' : 0}, 
                                'MUNIWASTE' : {'CO2' : 0},
                                'NUCLEAR' : {'CO2' : 0},
                                'STRAW' : {'CO2' : 0},
                                'WOOD' : {'CO2' : 0},
                                'PEAT' : {'CO2' : 0},
                                'STRAW' : {'CO2' : 0}},
                'CONDENSING-CCS' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh * 0.1}}, 
                'FUELCELL' : {'HYDROGEN' : {'CO2' : 0}}}
    {'BIOGAS' : 2.13,
                'WOODCHIPS' : 3.34,
                'MUNIWASTE' : 4,
                'WOOD' : 3.34,
                'WOODWASTE' : 3.34,
                'PEAT' : 3.34,
                'STRAW' : 3.34}

    with open('Pre-Processing/Output/BalmTechs.pkl', 'wb') as f:
        pickle.dump(BalmTechs, f)

#%% ------------------------------- ###
### 2. Normalise Antares Timeseries ###
### ------------------------------- ###

@CLI.command()
@click.pass_context
def old_preprocessing(ctx):
    """The old processing scripts"""
    
    ### 2.1 Solar and wind profiles
    #   Since VRE modelling is switched to VRE clusters, all 5_, 6_, 7_, 8_ etc. data
    #   was deleted! They are now stored in clusters of each region

    VRE_group = {'solar' : 'Solar PV',
                'wind' : 'Wind Onshore'}
    for VRE in ['solar', 'wind']:
        # Antares path to series from original model
        p = r'C:\Users\mberos\ElementsOutside1D\BZModel\input\%s\series'%VRE 
        l = pd.Series(os.listdir(p))
        l = list(l)
        
        balmmap[VRE + '_AveFLH'] = 0

        # for area in [area for area in A2B_regi.keys() if not(area in ['UKNI', 'GR03', 'ITCA', 'ITCS', 'ITS1', 'ITSA', 'ITSI'])]: # Delete Antares areas of higher resolution
        for area in A2B_regi.keys():
            
            vreseries = pd.DataFrame(np.zeros((8760, 31)))
            for subarea in ['VRE_*.txt', 'VRE_5_*_sres.txt', 'VRE_6_*_sres.txt',
                            'VRE_7_*_sres.txt', 'VRE_8_*_sres.txt']:
                
                # Format
                subarea = subarea.replace('VRE', VRE).replace('*', area.lower()).replace('lu00', 'lug1') # Remember Luxembourg was renamed from LUG1 to LU00

                # Read series, if any data is available
                try:
                    f = pd.read_csv(p + '/' + subarea, delimiter='\t', header=None)
                    f = f.loc[:, :stoch_years] # Filter stochastic years
                    # print(subarea, 'flh = %0.2f'%(f.sum() / f.max().max()))
                    vreseries += f
                except (EmptyDataError, FileNotFoundError):
                    pass
                    # print('No %s file'%subarea)
            
            # Normalisation
            max_prod = vreseries.max().max() / elec_loss
            print('%s %s FLH = %0.2f'%(area, VRE, (vreseries.sum() / vreseries.max().max()).mean() if vreseries.sum().sum() > 1e-6 else 0 ))
            # Save to balmmap
            idx = balmmap.id == A2B_regi[area][0]
            balmmap.loc[idx, VRE + '_AveFLH'] = (vreseries.sum() / vreseries.max().max()).mean() if vreseries.sum().sum() > 1e-6 else 0 

            # if len(reg_name.split('_')) > 1:
            #     print(reg_name)
            #     cluster_name = reg_name.split('_')[1].upper() + '_%s_'%VRE + reg_name.split('_')[0]
            #     reg_name = reg_name.split('_')[1]
            # else:
            #     cluster_name = reg_name.upper() + '_%s_0'%VRE
            # reg_name = reg_name.upper()
                
            if max_prod >= 1e-6: 
                # Create folder if it doesn't exist
                # (f / max_prod).to_csv(ant_region.replace('.txt', '_normalised-data.txt'), sep='\t', header=None, index=None)            
                (vreseries / max_prod).to_csv('Antares/input/renewables/series/%s/%s/series.txt'%(area.lower(), area.lower() + '_%s_0'%VRE), sep='\t', header=None, index=None)            
            else:
                pass
                # ('').to_csv('Antares/input/renewables/series/%s/%s/series.txt'%(area.lower(), area.lower() + '_%s_0'%VRE), sep='\t', header=None, index=None)                
                # (vreseries * 0).to_csv(ant_region.replace('.txt', '_normalised-data.txt'), sep='\t', header=None, index=None)
        
    ## Fix Germany and southern Norway for plotting the map
    try:
        for VRE in ['wind', 'solar']:
            # balmmap.loc[balmmap.id == 'DE4-N', '%s_AveFLH'%VRE] = balmmap.loc[balmmap.id == 'DE4-E', '%s_AveFLH'%VRE].values[0]
            # balmmap.loc[balmmap.id == 'DE4-W', '%s_AveFLH'%VRE] = balmmap.loc[balmmap.id == 'DE4-E', '%s_AveFLH'%VRE].values[0]
            # balmmap.loc[balmmap.id == 'DE4-S', '%s_AveFLH'%VRE] = balmmap.loc[balmmap.id == 'DE4-E', '%s_AveFLH'%VRE].values[0]
            # balmmap.loc[balmmap.id == 'NO2', '%s_AveFLH'%VRE]   = balmmap.loc[balmmap.id == 'NO1', '%s_AveFLH'%VRE].values[0]
            # balmmap.loc[balmmap.id == 'NO5', '%s_AveFLH'%VRE]   = balmmap.loc[balmmap.id == 'NO1', '%s_AveFLH'%VRE].values[0]

            

            ### Plot the FLH
            fig, ax = newplot(fc=fc, figsize=(10, 10))
            balmmap.plot(column='%s_AveFLH'%VRE, ax=ax, legend=True,
                        legend_kwds={'label' : '%s FLH (h)'%VRE.capitalize(),
                                    'orientation' : 'horizontal'})
            ax.set_xlim([-11, 30])
            ax.set_ylim([35, 68])
    except:
        plot_maps = False
        print("Geopandas probably not installed - can't plot maps")

    #%% Set unitcount to 0 in clusters with no profiles
    # NOTE: Obsolete after noticing that there is only a single profile pr. BZ
    # for area in A2B_regi.keys():
    #     for VRE in ['solar', 'wind']:
    #         cluster_config = configparser.ConfigParser()
    #         cluster_config.read('Antares/input/renewables/clusters/%s/list.ini'%(area.lower()))
    #         for subarea in ['_0', '_5', '_6', '_7', '_8']:
    #             # Antares path to series
    #             p = 'Antares/input/renewables/series/%s/%s_%s%s/series.txt'%(area.lower(), area.lower(), VRE, subarea)
                
    #             try:
    #                 f = pd.read_csv(p, delimiter='\t', header=None)
                
    #             except EmptyDataError:
    #                 print('No data in %s_%s%s'%(area, VRE, subarea))                
    #                 cluster_config.set('%s_%s%s'%(area, VRE, subarea), 'unitcount', '0')
        
    #             # Disable anything but 0
    #             if subarea != '_0':
    #                 cluster_config.set('%s_%s%s'%(area, VRE, subarea), 'enabled', 'false')
    #                 cluster_config.set('%s_%s%s'%(area, VRE, subarea), 'unitcount', '0')
            
    #         with open('Antares/input/renewables/clusters/%s/list.ini'%(area.lower()), 'w') as configfile:
    #             cluster_config.write(configfile)

    #%% Normalise Electricity and Hydrogen Demand Profiles

    ### NOTE: For new H2 regions in Antares it is assumed that, 
    #         TR = GR
    #          
    balmmap['El Dem'] = 0
    balmmap['H2 Dem'] = 0
    for area in list(A2B_regi.keys()) + list(A2B_regi_h2):
        # Load Antares demand
        try:
            ant_dem = pd.read_csv(r'C:\Users\mberos\ElementsOutside1D\BZModel\input' + '/load/series/load_%s.txt'%(area.lower()), sep='\t', header=None)
            normed = (ant_dem / ant_dem.sum().mean()) # Normalise wrt. annual energy
            (ant_dem / ant_dem.sum().mean()).to_csv('Antares' + '/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None, index=None)    
            # ax.plot(ant_dem[0] / ant_dem.sum().mean(), label=area)
            
            if 'z_h2' in area:
                balmmap.loc[balmmap.id == A2B_regi_h2[area][0], 'H2 Dem'] += ant_dem.sum().values[0] / 1e6
            else:
                balmmap.loc[balmmap.id == A2B_regi[area][0], 'El Dem'] += ant_dem.sum().values[0] / 1e6
        except (EmptyDataError, FileNotFoundError):
            print('No load data in %s'%area)

    ### Plot 
    if plot_maps:
        # El Demand (2040)
        fig, ax = newplot(fc=fc, figsize=(10, 10))
        balmmap.plot(column='El Dem', ax=ax, legend=True,
                    legend_kwds={'label' : 'Annual El Demand (TWh)',
                                'orientation' : 'horizontal'})
        ax.set_xlim([-11, 30])
        ax.set_ylim([35, 68])

        # H2 Demand (2040)
        fig, ax = newplot(fc=fc, figsize=(10, 10))
        balmmap.plot(column='H2 Dem', ax=ax, legend=True,
                    legend_kwds={'label' : 'Annual H2 Demand (TWh)',
                                'orientation' : 'horizontal'})
        ax.set_xlim([-11, 30])
        ax.set_ylim([35, 68])

    #%% Normalise Thermal Generation Profiles

    ### NOTE: For new H2 regions in Antares it is assumed that, 
    #         TR = GR
    #          
    # fig, ax = plt.subplots()
    for area in list(A2B_regi.keys()) + list(A2B_regi_h2):
        # Load Thermal generation profile
        try:
            l = os.listdir(r'C:\Users\mberos\ElementsOutside1D\BZModel\input' + '/thermal/series/%s'%area.lower())
            outageseries = 0
            for generator in l:
                gen = pd.read_csv('Antares/input/thermal/series/%s/%s/series.txt'%(area, generator), sep='\t', header=None)
                if gen[0].std() != 0:
                    (gen / gen.max()).to_csv('Antares/input/thermal/series/%s/%s.txt'%(area, 'outageseries_' + generator), sep='\t', header=None, index=None)            
                    # print('Outage modelled in %s for %s'%(area, generator))        
                    outageseries += 1
        except:
            print('No thermal generation in %s'%area)
        # try:
        #     ant_dem = pd.read_csv('Antares' + '/input/load/series/load_%s.txt'%(area), sep='\t', header=None)
        #     normed = (ant_dem / ant_dem.sum().mean()) # Normalise wrt. annual energy
        #     (ant_dem / ant_dem.sum().mean()).to_csv('Antares' + '/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None, index=None)    
        #     # ax.plot(ant_dem[0] / ant_dem.sum().mean(), label=area)
        # except EmptyDataError:
        #     print('No load data in %s'%area)
    # ax.legend()


    #%% ------------------------------- ###
    ###         3. VRE Analysis         ###
    ### ------------------------------- ###

    ### NOTE: This section is outdated! Need to change so that data is gathered from clusters
    #         ...and we won't be mapping actually (harmonising)

    ### 3.1 Get Antares Data 
    # ANT_FLH = pd.DataFrame(index=pd.MultiIndex(levels=[[],[]], codes=[[],[]]),
    #                    columns=['solar', 'wind'])

    # for area in A2B_regi.keys(): 
    # # for VRE in ['solar', 'wind']:
    #     # Antares path to series
    #     p = 'Antares/input/renewables/series/%s'%area
    #     l = pd.Series(os.listdir(p))

    #     for vre_region in l: 
    #     # for ant_region in [p + '%s_de00.txt'%VRE]:
    #         VRE = vre_region.split('_')[1]
            
    #         # Read series, if any data is available
    #         try:
    #             f = pd.read_csv(p + '/' + vre_region + '/series.txt', delimiter='\t', header=None)
    #             f = f.loc[:, :stoch_years] # Filter stochastic years
                
    #             cap = f.max().max()              # Estimates capacity as the highest production in all years
    #             flh = (f / cap).sum()            # Estimation of FLH for all stochastic years
                
    #             ANT_FLH.loc[(vre_region, 'Mean'), VRE] = round(flh.mean(), 2)
    #             # Std. Deviation of normalised production through a year, averaged by all stochastic years
    #             ANT_FLH.loc[(vre_region, 'StdDev'), VRE] = round((f / cap).std().mean(), 2)
    #             # Std. Deviation of FLH between stochastic years 
    #             ANT_FLH.loc[(vre_region, 'StochStdDev'), VRE] = round(flh.std(), 2) 
                                            
    #         except EmptyDataError:
    #             print('No data in %s'%vre_region)
                
            
    # #%% 3.2 Get Balmorel Data
    # BLM_FLH_WND = pd.read_excel('Pre-Processing/Data/BalmFLH.xlsx', sheet_name='WNDFLH', index_col=0)
    # BLM_FLH_SOL = pd.read_excel('Pre-Processing/Data/BalmFLH.xlsx', sheet_name='SOLEFLH', index_col=0)
    # BLM_FLH = pd.DataFrame({}, index=pd.Series(list(BLM_FLH_SOL.index) + list(BLM_FLH_WND.index)).unique())
    # BLM_FLH.loc[BLM_FLH_SOL.index, 'solar'] = BLM_FLH_SOL.FLH.values
    # BLM_FLH.loc[BLM_FLH_WND.index, 'wind'] = BLM_FLH_WND.FLH.values



    # #%% 3.3 Map wind & solar regions between Balmorel and Antares

    # ## This mapping will take a Balmorel area, and assign the Antares area
    # ## with the closest mean FLH to it.

    # # The container
    # VRE_MAPPING = pd.DataFrame(index=[], columns=['AntArea', 'BalmArea', 'Tech', 'FLH Difference [h]', 'Balm FLH Type'])

    # # Map Balmorel VRE areas to Antares Areas 
    # for VRE in ['solar', 'wind']:
    #     for ind in BLM_FLH[VRE].dropna().index:
    #         BLM_FLH0 = BLM_FLH.loc[ind, VRE] # Load Balmorel FLH
            
    #         # Balmorel Region
    #         R = ind.split('_')[0]
    #         # Search
    #         for ant_area in B2A_regi[R]:
    #             try:
                    
    #                 # Find Antares series
    #                 csearch = pd.Series(ANT_FLH.reset_index()['level_0'])
                    
    #                 idx = ant_area.lower() == csearch.str[:4]  # Assuming length 4 of region name
                    
                    
    #                 # print('Balmorel:')
    #                 # print(ind[0], VRE, round(BLM_FLH0), 'h')
    #                 # print('Antares:')
    #                 temp = ANT_FLH.loc[idx,'Mean',:][VRE]
    #                 # print(temp)
    #                 # Find lowest difference
    #                 dif_arr = temp-BLM_FLH0
    #                 dif0 = np.abs(dif_arr).min()
    #                 # Get sign
    #                 s = np.sign(dif_arr[dif0 == np.abs(dif_arr)].values[0])
    #                 ant_area = dif_arr[dif0 == np.abs(dif_arr)].index[0][0]
                    

    #                 VRE_MAPPING = pd.concat((VRE_MAPPING, pd.DataFrame({'BalmArea' : [ind], 'AntArea' : [ant_area.replace('_normalised-data', '')],
    #                                                                     'Tech' : [VRE], 'FLH Difference [h]' : [s*dif0],
    #                                                                     'Balm FLH Type' : ['FLH Input']})), ignore_index=True)
    #             except KeyError:
    #                 pass
    #                 # print('%s not found in B2A mapping'%R)

    # # Save it
    # VRE_MAPPING.to_csv('Pre-Processing/Output/AreaMapping.csv')

    # # # Illustrate difference
    # # for tech in ['solar', 'wind']:
    # #     idx = VRE_MAPPING['Tech'] == tech 
    # #     fig, ax = plt.subplots(figsize=(15, 5))
    # #     ax.plot(VRE_MAPPING.loc[idx, 'FLH Difference [h]'], 'o')
    # #     ax.set_title(tech)
    # #     ax.set_ylabel('FLH Difference [h]')
    # #     ax.set_xticks(VRE_MAPPING.loc[idx, :].index)
    # #     ax.set_xticklabels(VRE_MAPPING.loc[idx, 'BalmArea'], rotation=90)



    #%% ------------------------------- ###
    ###         4. A2B Weights          ###
    ### ------------------------------- ###

    ### 4.0 Weigths for assigning values from Balmorel to Antares
    ###     NOTE: It will use the current, absolute loads in Antares/input/load to distribute weights! 
    B2A_DH2_weights = {}
    B2A_DE_weights = {}

    ## Electricity
    for area in B2A_regi.keys():
        # If the spatial resolution is identical
        if len(B2A_regi[area]) == 1:
            B2A_DE_weights[area] = {B2A_regi[area][0] : 1}
            
        # If more Antares regions exist for that Balmorel region
        else:
            # Read data
            loads = []
            B2A_DE_weights[area] = {}
            for antarea in B2A_regi[area]:
                f = pd.read_csv('Antares/input/load/series/load_%s.txt'%antarea, sep='\t', header=None)
                loads.append(f.sum().mean()) # Mean annual load for all stochastic years
            
            # Assign load
            for n in range(len(B2A_regi[area])):
                B2A_DE_weights[area][B2A_regi[area][n]] = loads[n] / np.sum(loads)
                
    ## Hydrogen
    for area in B2A_regi_h2.keys():
        # If the spatial resolution is identical
        if len(B2A_regi_h2[area]) == 1:
            B2A_DH2_weights[area] = {B2A_regi_h2[area][0] : 1}
            
        # If more Antares regions exist for that Balmorel region
        else:
            # Read data
            loads = []
            B2A_DH2_weights[area] = {}
            for antarea in B2A_regi_h2[area]:
                f = pd.read_csv('Antares/input/load/series/load_%s.txt'%antarea, sep='\t', header=None)
                loads.append(f.sum().mean()) # Mean annual load for all stochastic years
            
            # Assign load
            for n in range(len(B2A_regi_h2[area])):
                B2A_DH2_weights[area][B2A_regi_h2[area][n]] = loads[n] / np.sum(loads)
                

    with open('Pre-Processing/Output/B2A_DE_weights.pkl', 'wb') as f:
        pickle.dump(B2A_DE_weights, f)
    with open('Pre-Processing/Output/B2A_DH2_weights.pkl', 'wb') as f:
        pickle.dump(B2A_DH2_weights, f)


    ### 4.1 Make Demand Ratios for Fictive Demand Approach
    f = pd.read_csv('Pre-Processing/Data/BalmDE.csv', sep=';')
    f['C'] = f.R.str[:2]

    A2B_DE_weights = {}

    for ant_area in A2B_regi.keys():
        
        A2B_DE_weights[ant_area] = {}
        total_dem = np.sum([f.loc[(f.R == balm_area) & (f.Y == year), 'Val'].sum() for balm_area in A2B_regi[ant_area]])

        for balm_area in A2B_regi[ant_area]:
            
            r_dem = f.loc[(f.R == balm_area) & (f.Y == year), 'Val'].sum()
            
            # Save share of electricity demand in dict
            A2B_DE_weights[ant_area][balm_area] = r_dem / total_dem
            
            # Print the share of demand in that region
            print(ant_area, balm_area, r_dem / total_dem*100, '%')


    ### 4.2 Save A2B DE weight dictionary
    with open('Pre-Processing/Output/A2B_DE_weights.pkl', 'wb') as f:
        pickle.dump(A2B_DE_weights, f) 


    ### 4.3 Fictive Demand from Antares to Balmorel
    f = pd.read_csv('Pre-Processing/Data/BalmDH2.csv', sep=';')
    f['C'] = f.R.str[:2]

    A2B_DH2_weights = {}

    for ant_area in A2B_regi_h2.keys():
        
        A2B_DH2_weights[ant_area] = {}
        total_dem = np.sum([f.loc[(f.R == balm_area) & (f.Y == year), 'Val'].sum() for balm_area in A2B_regi_h2[ant_area]])
        
        for balm_area in A2B_regi_h2[ant_area]:
            
            r_dem = f.loc[(f.R == balm_area) & (f.Y == year), 'Val'].sum()
            
            # Save share of electricity demand in dict
            val = r_dem / total_dem
            if pd.isna(val):
                print('No demand, setting to 1')
                A2B_DH2_weights[ant_area][balm_area] = 1.0
            else:
                A2B_DH2_weights[ant_area][balm_area] = val
                
            # Print the share of demand in that region
            print(ant_area, balm_area, r_dem / total_dem*100, '%')


    ### 3.4 Save A2B DE weight dictionary
    with open('Pre-Processing/Output/A2B_DH2_weights.pkl', 'wb') as f:
        pickle.dump(A2B_DH2_weights, f) 

    #%% ------------------------------- ###
    ###             5. Hydro            ###
    ### ------------------------------- ###

    ### 5.0 Choose parameters
    # 27 = 2009, the worst year (note we start counting from 0 here, compared to in Antares UI!). 
    # 30 = 2012, the average year in Balmorel (note we start counting from 0 here, compared to in Antares UI!)
    balm_year = 30

    # Whether or not to create FLH based on input data or Antares output
    compute_FLH_on_input = True
    ant_output = '20231219-2246eco-hydrotestquickfix_iter0_y-2050'
    mc_choice = '00001' # Should be aligned to the balm_year chosen..

    # Balmorel timestamps
    S = ['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)]
    T = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)]
    balmtime_index = ['%s . %s'%(S0, T0) for S0 in S for T0 in T]

    ### 5.1 Prepare placeholders
    hydro_res = configparser.ConfigParser()
    hydro_res.read('Antares/input/hydro/hydro.ini')

    hydro_AAA = '$ifi not %ADJUSTHYDRO%==yes $goto dont_adjust_hydro\n'
    # GNR_RES_WTR_NOPMP (100 % efficiency)
    # GNR_RES_WTR_PMP_MC-01 (100 % efficiency)
    hydro_GKFX = '$ifi not %ADJUSTHYDRO%==yes $goto dont_adjust_hydro_2\n'
    hydro_WTRRRFLH = ''
    hydro_WTRRSFLH = ''
    hydro_WTRRSVAR_S = pd.DataFrame(index=S)
    hydro_WTRRRVAR_T = pd.DataFrame(index=balmtime_index)
    hydro_HYRSMAXVOL_G = '$ifi not %ADJUSTHYDRO%==yes $goto dont_adjust_hydro\n'

    for area in A2B_regi.keys():
        
        ### 5.2 Reservoir power capacity and pumping eff in the area itself
        try:
            turb_cap = pd.read_table('Antares/input/hydro/common/capacity/maxpower_%s.txt'%area, header=None)[0].max() # MW
            pump_cap = pd.read_table('Antares/input/hydro/common/capacity/maxpower_%s.txt'%area, header=None)[2].max() # MW
            
            res_series = pd.read_table('Antares/input/hydro/series/%s/mod.txt'%area.lower(), header=None) # MW

            # Summing weekly flows for Balmorel
            hydro_WTRRSVAR_S['%s_hydro0'%area] = res_series.rolling(window=7).sum()[6::7][balm_year].values
            
            if compute_FLH_on_input:
                # FLH
                hydro_WTRRSFLH += "%s_hydro0\t\t%0.2f\n"%(area, 
                                                                    ((res_series[balm_year]).sum())/(8760*turb_cap)*8760)
            else:
                res_dispatch = pd.read_table('Antares/output/' + ant_output +\
                                        '/economy/mc-ind/%s/areas/%s/values-hourly.txt'%(mc_choice, area.lower()),
                                        skiprows=[0,1,2,3,5,6])['H. STOR'] 
                hydro_WTRRSFLH += "%s_hydro0\t\t%0.2f\n"%(area, 
                                                                    res_dispatch.sum()/(8760*turb_cap)*8760)

                        
            # Make .inc file commands
            hydro_AAA += '%s_hydro0\n'%area 
            if pump_cap == 0:
                G = 'GNR_RES_WTR_NOPMP'
                hydro_GKFX += "GKFX(YYY,'%s_hydro0','%s') = %0.2f;\n"%(area, G, turb_cap)
            else:
                G = 'GNR_RES_WTR_PMP_MC-01'
                hydro_GKFX += "GKFX(YYY,'%s_hydro0','%s') = %0.2f;\n"%(area, G, turb_cap)
            
            # See if there's a reservoir capacity
            try:
                res_cap = hydro_res.getfloat('reservoir capacity', area)
                hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_hydro0', '%s') = %0.2f;\n"%(area, G, res_cap/turb_cap)
            except configparser.NoOptionError:
                hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_hydro0', '%s') = %0.2f;\n"%(area, G, 0)
            
        except EmptyDataError:
            # No reservoir storage in area
            pass

        ### 5.3 Run of river in the area itself
        try:
            ror_series = pd.read_table('Antares/input/hydro/series/%s/ror.txt'%area.lower(), header=None) # MW
            ror_cap = ror_series.max().max()

            # Make .inc file commands        
            if not('%s_hydro0'%area in hydro_AAA):
                hydro_AAA += '%s_hydro0\n'%area 
                
            hydro_WTRRRVAR_T['%s_hydro0'%area] = ror_series.loc[:8735, balm_year].values
            
            if compute_FLH_on_input:
                # FLH
                hydro_WTRRRFLH += "%s_hydro0\t\t%0.2f\n"%(area,
                                                                    ((ror_series[balm_year]).sum())/(8760*ror_cap)*8760)
            else:
                ror_dispatch = pd.read_table('Antares/output/' + ant_output +\
                                        '/economy/mc-ind/%s/areas/%s/values-hourly.txt'%(mc_choice, area.lower()),
                                        skiprows=[0,1,2,3,5,6])['H. ROR'] 
                hydro_WTRRRFLH += "%s_hydro0\t\t%0.2f\n"%(area, 
                                                                    ror_dispatch.sum()/(8760*ror_cap)*8760)
                    
            
            hydro_GKFX += "GKFX(YYY, '%s_hydro0', 'GNR_ROR_WTR') = %0.2f;\n"%(area, ror_cap)
        except EmptyDataError:
            pass

        ### 5.4 Individual Hydro Areas
        for hydro_area in ['2_*_HYDRO_OPEN', '3_*_HYDRO_RES', '4_*_HYDRO_SWELL']:
            hydro_a0 = hydro_area.replace('*', area).lower()
            
            
            
            # Swell areas have run-of-river and reservoir, but no storage
            if 'SWELL' in hydro_area:
                try:
                    turb_cap = pd.read_table('Antares/input/links/%s/capacities/%s_direct.txt'%(hydro_a0, area.lower()), header=None).max()[0]
                                    
                    res_series = pd.read_table('Antares/input/hydro/series/%s/mod.txt'%hydro_a0, header=None) # MW

                    # Summing weekly flows for Balmorel
                    hydro_WTRRSVAR_S['%s_hydro%s'%(area, hydro_a0[0])] = res_series.rolling(window=7).sum()[6::7][balm_year].values
                    
                    if compute_FLH_on_input:
                        # FLH
                        hydro_WTRRSFLH += "%s_hydro%s\t\t%0.2f\n"%(area, hydro_a0[0],
                                                                            ((res_series[balm_year]).sum())/(8760*turb_cap)*8760)
                    else:
                        res_dispatch = pd.read_table('Antares/output/' + ant_output +\
                                    '/economy/mc-ind/%s/links/%s/values-hourly.txt'%(mc_choice, hydro_a0 + ' - %s'%area.lower()),
                                    skiprows=[0,1,2,3,5,6])['FLOW LIN.']
                        hydro_WTRRSFLH += "%s_hydro%s\t\t%0.2f\n"%(area, hydro_a0[0],
                                                                            ((res_dispatch).sum())/(8760*turb_cap)*8760)                    
                                            
                    ror_series = pd.read_table('Antares/input/hydro/series/%s/ror.txt'%hydro_a0, header=None) # MW 
                    hydro_WTRRRVAR_T['%s_hydro%s'%(area, hydro_a0[0])] = ror_series.loc[:8735, balm_year].values
                    
                    if compute_FLH_on_input:
                        # FLH
                        hydro_WTRRRFLH += "%s_hydro%s\t\t%0.2f\n"%(area, hydro_a0[0],
                                                                            ((ror_series[balm_year]).sum())/(8760*turb_cap)*8760)
                    else:
                        hydro_WTRRRFLH += "%s_hydro%s\t\t%0.2f\n"%(area, hydro_a0[0],
                                                                            ((res_dispatch).sum())/(8760*turb_cap)*8760)                    
                    # .inc file commands
                    hydro_AAA += '%s_hydro%s\n'%(area, hydro_a0[0]) 
                                            
                    # Balmorel accounts for a reservoir + ror area not using double capacity:
                    hydro_GKFX += "GKFX(YYY, '%s_hydro%s', 'GNR_RES_WTR_NOPMP') = %0.2f;\n"%(area, hydro_a0[0], turb_cap/2)
                    hydro_GKFX += "GKFX(YYY, '%s_hydro%s', 'GNR_ROR_WTR') = %0.2f;\n"%(area, hydro_a0[0], turb_cap/2)
                    # print(hydro_a0, 'worked')
                    # print(area, " had swell area")
                except FileNotFoundError:
                    pass
            else:
                try:
                    
                    if (hydro_a0 == '2_pt00_hydro_open'):
                        turb_cap = pd.read_table('Antares/input/thermal/series/w_hydro/maxturbopen_%s/series.txt'%(hydro_a0), header=None)[balm_year].max()
                        pump_cap = pd.read_table('Antares/input/thermal/series/w_hydro/maxpumpopen_%s/series.txt'%(hydro_a0), header=None)[balm_year].max()
                        # print(hydro_a0, '2_cap:', turb_cap, pump_cap)
                    elif ('RES' in hydro_area):
                        turb_cap = pd.read_table('Antares/input/thermal/series/w_hydro/pmaxreservoir_%s/series.txt'%(hydro_a0), header=None)[balm_year].max()
                        pump_cap = pd.read_table('Antares/input/thermal/series/w_hydro/pmaxreservoir_%s/series.txt'%(hydro_a0), header=None)[balm_year].max()
                        # print(hydro_a0, '3_cap:', turb_cap, pump_cap)
                        
                    else:
                        turb_cap = pd.read_table('Antares/input/links/%s/capacities/%s_direct.txt'%(hydro_a0, area.lower()), header=None).max()[0]
                        pump_cap = pd.read_table('Antares/input/links/%s/capacities/%s_indirect.txt'%(hydro_a0, area.lower()), header=None).max()[0]
                    
                    # Check if there's inflow, or if it's just a pumped storage
                    try:
                        res_series = pd.read_table('Antares/input/hydro/series/%s/mod.txt'%hydro_a0, header=None) # MW

                        # Summing weekly flows for Balmorel
                        hydro_WTRRSVAR_S['%s_hydro%s'%(area, hydro_a0[0])] = res_series.rolling(window=7).sum()[6::7][balm_year].values
                        
                        if compute_FLH_on_input:
                            # FLH
                            hydro_WTRRSFLH += "%s_hydro%s\t\t%0.2f\n"%(area, hydro_a0[0],
                                                                                ((res_series[balm_year]).sum())/(8760*turb_cap)*8760)
                        else:
                            res_dispatch = pd.read_table('Antares/output/' + ant_output +\
                                        '/economy/mc-ind/%s/links/%s/values-hourly.txt'%(mc_choice, hydro_a0 + ' - %s'%area.lower()),
                                        skiprows=[0,1,2,3,5,6])['FLOW LIN.']
                            hydro_WTRRSFLH += "%s_hydro%s\t\t%0.2f\n"%(area, hydro_a0[0],
                                                                                ((res_dispatch).sum())/(8760*turb_cap)*8760)                    
                                
                        # .inc file commands
                        hydro_AAA += '%s_hydro%s\n'%(area, hydro_a0[0]) 
                        if pump_cap == 0:
                            G = 'GNR_RES_WTR_NOPMP'
                            hydro_GKFX += "GKFX(YYY,'%s_hydro%s','%s') = %0.2f;\n"%(area, hydro_a0[0], G, turb_cap)
                        else:
                            G = 'GNR_RES_WTR_PMP_MC-01'
                            hydro_GKFX += "GKFX(YYY,'%s_hydro%s','%s') = %0.2f;\n"%(area, hydro_a0[0], G, turb_cap)
                        
                        # See if there's a reservoir capacity
                        try:
                            res_cap = hydro_res.getfloat('reservoir capacity', hydro_a0)
                            hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_hydro%s', '%s') = %0.2f;\n"%(area, hydro_a0[0], G, res_cap/turb_cap)
                        except configparser.NoOptionError:
                            hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_hydro%s', '%s') = %0.2f;\n"%(area, hydro_a0[0], G, 0)
                        # print(area, " had inflow")
                    except:
                        print(hydro_a0, 'is exclusively pumped storage')
                
                except FileNotFoundError:
                    pass


        ### 5.5 Pumped Hydro Storage (No inflow) 
        for hydro_area in ['1_TURB_closed']:
            
            try:
                turb_cap = pd.read_table('Antares/input/links/%s/capacities/%s_direct.txt'%(hydro_area, area.lower()), header=None).max()[0]
                
                # Found as exclusive pumped storage before, so add to this capacity
                if (area == 'FR00') | (area == 'MK00'):
                    turb_cap += pd.read_table('Antares/input/links/2_%s_hydro_open/capacities/%s_direct.txt'%(area.lower(), area.lower()), header=None).max()[0]
                
                # .inc file commands
                # Note that this capacity is defined in MWh! Thus factored the unloading parameter GDSTOHUNLD of  9.4 
                hydro_GKFX += "GKFX(YYY, '%s_A', 'GNR_ES_WTR_PMP') = %0.2f;\n"%(area, turb_cap*9.4)
                
            except FileNotFoundError:
                pass
            
            #'GNR_ES_WTR_PMP' changed to 75% efficiency in GKFX incfile prefix
            
            
        # pro = pd.DataFrame(columns=['Hour', 'Area', 'Tect', 'Value'])
        # pro.loc[hour, 'Antares', area, 'Water', 'HYDRO-RESERVOIRS'] = f.loc[0, 'H. STOR'] / 1e6
        # pro.loc[hour, 'Antares', area, 'Water', 'HYDRO-RUN-OF-RIVER'] = f.loc[0, 'H. ROR'] / 1e6
        
        
        
        
        ## Modelled as virtual units (missing 0_pump_open for 2_*_hydro_open, but doesn't interact with BZ's)
        # for hydro_area in ['1_PUMP_closed', '1_TURB_closed', '2_*_HYDRO_OPEN', '3_*_HYDRO_RES', '4_*_HYDRO_SWELL']:
        #     try:
        #         # This is the production, which is need for FLH estimation
        #         f = pd.read_table('Antares/output/' + ant_output +\
        #                 '/economy/%s/links/%s/values-hourly.txt'%(mc_choice, hydro_area.replace('*', area).lower() + ' - %s'%area.lower()),
        #                 skiprows=[0,1,2,3,5,6]) 
                
        #         pro = pd.concat(())  
                
        #         # Get link capacities in input/links/%s/capacities/%s_direct.txt or %s_indirect.txt
                
        #     except:
        #         print('No connection between %s and %s'%(hydro_area.replace('*', area), area))                


    # End with dont_adjust_hydro label, to make it an option
    hydro_GKFX += '$label dont_adjust_hydro_2\n'
    hydro_AAA += '$label dont_adjust_hydro\n'


    #%% ------------------------------- ###
    ###        6. Harmonisation         ###
    ### ------------------------------- ###

    ### 6.1 Choose stochastic year for balmorel input
    # 27 = 2009, the worst year (note we start counting from 0 here, compared to in Antares UI!). 
    # 30 = 2012, the average year in Balmorel (note we start counting from 0 here, compared to in Antares UI!)
    # balm_year = 30 # This is chosen already in hydro section



    ### 6.2 Hardcoded, first parts of the .inc files (easier to edit and define in GAMS)
    incfile_prefix_path = 'Pre-Processing/Data/IncFile Prefixes'
    def ReadIncFilePrefix(name):
        global balm_year
        global incfile_prefix_path
        
        if ('WND' in name) | ('SOLE' in name) | ('DE' in name) | ('WTR' in name):
            string = "* Weather year %d from Antares\n"%(balm_year+ 1) + ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())
        else:
            string = ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())
        
        return string

    incfiles = {incfile : IncFile(name=incfile, prefix=ReadIncFilePrefix(incfile)) for incfile in pd.Series(os.listdir('Pre-Processing/Data/IncFile Prefixes')).str.rstrip('.inc')}

    # Fill in Hydro from earlier
    incfiles['ANTBALM_WTRRRVAR_T'].body = hydro_WTRRRVAR_T.to_string()
    incfiles['ANTBALM_WTRRSVAR_S'].body = hydro_WTRRSVAR_S.to_string()
    incfiles['ANTBALM_WTRRRFLH'].body = hydro_WTRRRFLH
    incfiles['ANTBALM_WTRRSFLH'].body = hydro_WTRRSFLH
    incfiles['ANTBALM_HYRSMAXVOL_G'].body = hydro_HYRSMAXVOL_G
    incfiles['ANTBALM_CCCRRRAAA'].body = hydro_AAA
    incfiles['ANTBALM_AAA'].body = hydro_AAA
    for line in hydro_AAA.split('\n'):
        if (line != '') & (line != '$label dont_adjust_hydro') & (line != '$ifi not %ADJUSTHYDRO%==yes $goto dont_adjust_hydro'):
            incfiles['ANTBALM_RRRAAA'].body += "RRRAAA('%s', '%s') = YES;\n"%(A2B_regi[line.split('_')[0]][0], line)
            
    # Placeholders for 3 sections in ANTBALM_GKFX
    incfiles['ANTBALM_GKFX'].body1 = hydro_GKFX
    incfiles['ANTBALM_GKFX'].body2 = ''
    incfiles['ANTBALM_GKFX'].body3 = ''
    incfiles['ANTBALM_WTRRRVAR_T'].suffix   = "\n;\nWTRRRVAR_T(AAA,SSS,TTT) = WTRRRVAR_T1(SSS,TTT,AAA);\nWTRRRVAR_T1(SSS,TTT,AAA) = 0;\n$label dont_adjust_hydro"
    incfiles['ANTBALM_WTRRSVAR_S'].suffix   = "\n;\nWTRRSVAR_S(AAA,SSS) = WTRRSVAR_S1(SSS,AAA);\nWTRRSVAR_S1(SSS,AAA) = 0;\n$label dont_adjust_hydro"
    incfiles['ANTBALM_HYRSMAXVOL_G'].suffix = '$label dont_adjust_hydro'
    incfiles['ANTBALM_WTRRRFLH'].suffix = "/;\n$label dont_adjust_hydro"
    incfiles['ANTBALM_WTRRSFLH'].suffix = "/;\n$label dont_adjust_hydro"


    ### 6.3 Balmorel timestamps
    S = ['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)]
    T = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)]
    balmtime_index = ['%s . %s'%(S0, T0) for S0 in S for T0 in T]
    WND_VAR_T = pd.DataFrame([], index=balmtime_index)
    SOLE_VAR_T = pd.DataFrame([], index=balmtime_index)
    DE_VAR_T = pd.DataFrame([], index='RESE . ' + pd.Series(balmtime_index))


    ### 6.4 Automatic definition of .inc file body content
    for BalmArea in list(B2A_regi.keys()):

        for area in B2A_regi[BalmArea]:    
            if area != 'ITCO':        
                # if len(A2B_regi[area]) > 1:
                #     # If Balmorel is higher resolved:
                #     area += '_%s'%BalmArea

                ## Sets
                # ANTBALM_CCCRRRAAA, RRRAAA and AAA
                incfiles['ANTBALM_CCCRRRAAA'].body += "%s_A\n"%area
                incfiles['ANTBALM_RRRAAA'].body += "RRRAAA('%s', '%s_A') = YES;\n"%(BalmArea, area)
                incfiles['ANTBALM_AAA'].body += "%s_A\n"%area
                
                
                ## Moving Renewable Capacity to New Areas
                if area not in ['FR15', 'GR15', 'ITCA', 'ITCN', 
                                'ITCS', 'ITN1', 'ITS1', 'ITSA', 'ITSI', 'UKNI']:
                    incfiles['ANTBALM_GKFX'].body1 += "NOTNEWAREA('%s_A') = NO;\n"%area
                    incfiles['ANTBALM_GKFX'].body2 += "GKFX(YYY,'%s_A',GGG)$(GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE OR GDATA(GGG,'GDTECHGROUP') EQ SOLARPV) = SUM(AAA, GKFX(YYY,AAA,GGG)$(RRRAAA('%s',AAA) AND NOTNEWAREA(AAA) AND (GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE OR GDATA(GGG,'GDTECHGROUP') EQ SOLARPV)));\n"%(area, BalmArea)
                    incfiles['ANTBALM_GKFX'].body3 += "GKFX(YYY,AAA,GGG)$(RRRAAA('%s',AAA) AND NOTNEWAREA(AAA) AND (GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE OR GDATA(GGG,'GDTECHGROUP') EQ SOLARPV)) = 0;\n"%BalmArea
                
                ## VAR T and FLH
                try:
                    f = pd.read_csv('Antares/input/renewables/series/%s/%s/series.txt'%(area.replace('_%s'%BalmArea, '').lower(), area.replace('_%s'%BalmArea, '').lower() + '_wind_0'), sep='\t', header=None)            
                    WND_VAR_T[area + '_A'] = f.loc[:8735, balm_year].values / elec_loss # Electricity loss accounted for in Balmorel
                    incfiles['ANTBALM_WNDFLH'].body += "%s_A \t\t\t %0.2f\n"%(area, WND_VAR_T[area + '_A'].sum())
                except EmptyDataError:
                    print('No solar data in %s'%(area))    
                    
                try:
                    f = pd.read_csv('Antares/input/renewables/series/%s/%s/series.txt'%(area.replace('_%s'%BalmArea, '').lower(), area.replace('_%s'%BalmArea, '').lower() + '_solar_0'), sep='\t', header=None)            
                    SOLE_VAR_T[area + '_A'] = f.loc[:8735, balm_year].values / elec_loss # Electricity loss accounted for in Balmorel
                    incfiles['ANTBALM_SOLEFLH'].body += "%s_A \t\t\t %0.2f\n"%(area, SOLE_VAR_T[area + '_A'].sum())
                except EmptyDataError:
                    print('No wind data in %s'%(area))    
                    
                ## DE_VAR_T
                f = pd.read_csv('Antares/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None)            
                DE_VAR_T[BalmArea] = f.loc[:8735, balm_year].values        
                    
                ## AGKN, Investment options (Look at prefix for ANTBALM_AGKN, other investment options have been taken care of)
                for year0 in ['2020', '2030', '2040', '2050']:
                    incfiles['ANTBALM_AGKN'].body += "AGKN('%s_A','GNR_WT_ONS-%s') = YES;\n"%(area, year0)
                    incfiles['ANTBALM_AGKN'].body += "AGKN('%s_A','GNR_WT_OFF-%s') = YES;\n"%(area, year0)
                    
                    if area != 'NON1_A':
                        incfiles['ANTBALM_AGKN'].body += "AGKN('%s_A','GNR_PV-%s') = YES;\n"%(area, year0) 

    ### 6.5 The finishing lines and saving .inc files in Balmorel/base/data
    incfiles['ANTBALM_CCCRRRAAA'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_CCCRRR'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_RRR'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_AAA'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_WNDFLH'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_SOLEFLH'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_WND_VAR_T'].body = WND_VAR_T.to_string()
    incfiles['ANTBALM_WND_VAR_T'].suffix = '\n;\nWND_VAR_T(IA,SSS,TTT)$WND_VAR_T2(SSS,TTT,IA) = WND_VAR_T2(SSS,TTT,IA);\nWND_VAR_T2(SSS,TTT,AAA)=0;\n$label USEANTARESDATAEND'
    incfiles['ANTBALM_SOLE_VAR_T'].body = SOLE_VAR_T.to_string()
    incfiles['ANTBALM_SOLE_VAR_T'].suffix = '\n;\nSOLE_VAR_T(IA,SSS,TTT)$SOLE_VAR_T2(SSS,TTT,IA) = SOLE_VAR_T2(SSS,TTT,IA);\nSOLE_VAR_T2(SSS,TTT,AAA)=0;\n$label USEANTARESDATAEND'
    incfiles['ANTBALM_AGKN'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_GKFX'].body = incfiles['ANTBALM_GKFX'].body1 + incfiles['ANTBALM_GKFX'].body2 + incfiles['ANTBALM_GKFX'].body3
    incfiles['ANTBALM_GKFX'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_RRRAAA'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_DE_VAR_T'].body = DE_VAR_T.to_string()
    incfiles['ANTBALM_DE_VAR_T'].suffix = """\n;\nDE_VAR_T(RRR,'RESE',SSS,TTT) =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    DE_VAR_T(RRR,'OTHER',SSS,TTT) =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    DE_VAR_T(RRR,'DATACENTER',SSS,TTT)$SUM(YYY,DE(YYY,RRR,'DATACENTER'))  =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    DE_VAR_T(RRR,'PII',SSS,TTT) = ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);
    ANTBALM_DE_VAR_T(DEUSER,SSS,TTT,RRR)=0;\n$label USEANTARESDATAEND"""

    ## Save
    for key in incfiles.keys():
        incfiles[key].save()

    #%% ------------------------------- ###
    ###         7. Transmission         ###
    ### ------------------------------- ###

    # This analysis requires geopandas
    if plot_maps:
        
        # From DEA 2021, 111 'Main electricity distribution grid' (no data for transmission? these costs are for 50/60 kV)
        XE_cost = 3100 # €/MW/km high bound
        
        # From Kountouris, Ioannis. “A Unified European Hydrogen Infrastructure Planning to Support the Rapid Scale-up of Hydrogen Production,” August 1, 2023. https://doi.org/10.21203/rs.3.rs-3185467/v1.
        TransInvCost = {2030 : {'H2' : {'Onshore' : {'New' : 536.17,
                                                    'Rep' : 150},
                                        'Offshore': {'New' : 902.13,
                                                    'Rep' : 175}},
                                'El' : {'Onshore' : {'New' : XE_cost},
                                        'Offshore': {'New' : XE_cost}}}, 
                        2040 : {'H2' : {'Onshore' : {'New' : 263.08,
                                                    'Rep' : 86.15},
                                        'Offshore': {'New' : 450.77,
                                                    'Rep' : 120}},
                                'El': {'Onshore' : {'New' : XE_cost},
                                    'Offshore': {'New' : XE_cost}}}}
        TransInvCost = nested_dict_to_df(TransInvCost) # €/MW/km

        # Plot Settings
        scale = 0.75
        fig, ax = plt.subplots(dpi=75/scale, figsize=(20*scale,15*scale))
        balmmap.plot(ax=ax, facecolor=[.85, .85, .85])

        ### 7.1 Electricity Grid
        el_grid = gpd.read_file('Pre-Processing/Data/Infrastructure/EUPowerGrid.geojson')
        
        # Fix unknown voltages
        el_grid.loc[el_grid['voltage'] == '', 'voltage'] = 'unknown'
        # el_grid.voltage.astype('category')
        
        el_grid.plot(column='voltage', legend=True, ax=ax, linewidth=.5)
        # p1 = ax.get_legend().get_patches()
        
        ### 7.2 Gas Grid
        gas_grid = gpd.read_file('Pre-Processing/Data/Infrastructure/IGGIN_PipeSegments.geojson')
        gas_grid.plot(ax=ax, linewidth=.5, color='orange', label='Gas Grid')


        ax.set_xlim([-11, 35])
        ax.set_ylim([35, 68])
        # ax.legend(('132', '220', '300', '380', '500', '750', 'Unknown'))


        ### 7.3 Calculate Distances    
        balmmap_meters = balmmap.to_crs(4328)

        D = pd.DataFrame(
            distance_matrix(balmmap_meters.geometry.apply(lambda polygon: (polygon.centroid.x, polygon.centroid.y)).tolist(),
                            balmmap_meters.geometry.apply(lambda polygon: (polygon.centroid.x, polygon.centroid.y)).tolist()),
            index=balmmap_meters.id,
            columns=balmmap_meters.id
        ) / 1000 # in km

        # Links
        links = pd.read_csv('Pre-Processing/Data/Links.csv', sep=';')


        # Get distance matrix
        D = distance_matrix(x=balmmap.geometry.centroid.x,
                            y=balmmap.geometry.centroid.y)


### Main
if __name__ == '__main__':
    CLI()