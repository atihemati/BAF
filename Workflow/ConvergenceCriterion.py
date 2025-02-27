"""
Script for assessing convergence criterion(s)

Created on 18/08/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

print('\n|--------------------------------------------------|')   
print('              CONVERGENCE-CRITERION')
print('|--------------------------------------------------|\n')  

import pandas as pd
import numpy as np
import platform
OS = platform.platform().split('-')[0]

import os
if ('Workflow' in __file__) | ('Pre-Processing' in __file__):
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

import gams
import pickle
import configparser
import sys
from Workflow.Functions.GeneralHelperFunctions import symbol_to_df, AntaresOutput
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

### 0.0 Load configuration file
Config = configparser.ConfigParser()
Config.read('Workflow/MetaResults/%s_meta.ini'%SC)
SC_folder = Config.get('RunMetaData', 'SC_Folder')
UseFlexibleDemand = Config.getboolean('PeriProcessing', 'UseFlexibleDemand')

USE_CAPCRED   = Config.getboolean('PostProcessing', 'Capacitycredit')
USE_H2CAPCRED   = Config.getboolean('PostProcessing', 'H2Capacitycredit')

# Years
Y = np.array(Config.get('RunMetaData', 'Y').split(',')).astype(int)
Y.sort()
ref_year = Config.getint('RunMetaData', 'ref_year')

### 0.1 Other Variables
wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  

fLOLD = pd.read_csv('Workflow/OverallResults/%s_LOLD.csv'%SC, index_col=0)

i = Config.getint('RunMetaData', 'CurrentIter')

# Mappings from Antares to Balmorel Nodes
with open(wk_dir + '/Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
    A2B_regi = pickle.load(f)

with open(wk_dir + '/Pre-Processing/Output/A2B_regi_h2.pkl', 'rb') as f:
    A2B_regi_h2 = pickle.load(f)

# Weights on fictive electricity demand from A2B
with open(wk_dir + '/Pre-Processing/Output/A2B_DE_weights.pkl', 'rb') as f:
    A2B_DE_weights = pickle.load(f) 

ws = gams.GamsWorkspace()  
    

### ------------------------------- ###
###      1. Assess Convergence      ###
### ------------------------------- ###


### 1.0 Initialisations
l = np.array(os.listdir(wk_dir + '/Antares/output'))
l.sort()
l = pd.Series(l[l != 'maps'])

# Get reserve margin 
if USE_CAPCRED:
    old_resmar = ws.add_database_from_gdx(wk_dir + '/Balmorel/%s/model/all_endofmodel.gdx'%SC_folder)
    
    # Get start of investment year
    invest_year_start = symbol_to_df(old_resmar, 'GINVESTSTART')['Value'][0]
    
    # Get old reserve margin
    old_resmar = symbol_to_df(old_resmar, 'ANTBALM_RESMAR', ['Y', 'R', 'Value'])
    old_resmar['Iter'] = i
    old_resmar['Carrier'] = 'Electricity'
    
    # Load previous reserve margins
    resmar = pd.read_csv('Workflow/OverallResults/%s_ResMar.csv'%SC)
    
    # Store current reserve margin
    temp = pd.concat((resmar, old_resmar), ignore_index=True)
    
    if USE_H2CAPCRED:
        old_h2resmar = ws.add_database_from_gdx(wk_dir + '/Balmorel/%s/model/all_endofmodel.gdx'%SC_folder)
        h2resmar = symbol_to_df(old_h2resmar, 'ANTBALM_H2RESMAR', ['Y', 'R', 'Value'])
        h2resmar['Iter'] = i
        h2resmar['Carrier'] = 'Hydrogen'
        temp = pd.concat((temp, h2resmar), ignore_index=True)
        
    temp.to_csv('Workflow/OverallResults/%s_ResMar.csv'%SC, index=False)        
    
    old_resmar = old_resmar.groupby(by=['Y', 'R']).aggregate({'Value' : np.sum})
    
       

### 1.1 Assess Convergence
convergence = True
for year in Y:

    # Skip reference year
    if ref_year == year and i != 0:	
        continue
    
    idx = (l.str.find('eco-' + SC.lower() + '_iter%d_y-%d'%(i, year)) != -1)
    ant_res = l[idx].iloc[-1] # Should be the most recent
    print('Analysing Antares result %s'%ant_res)     
    # Get run
    AntOut = AntaresOutput(ant_res, wk_dir=wk_dir)
    
    if UseFlexibleDemand:
        flexdem_uns = AntOut.load_area_results('0_flexdem', 'details')
    
    # Elec LOLD
    for area in A2B_regi.keys(): 
        f = pd.read_table(wk_dir + '/Antares/output/' + ant_res + '/economy/mc-all/areas/%s/values-hourly.txt'%area.lower(),
                        skiprows=[0,1,2,3,5,6]) 
        
        ## Loss of load duration
        LOLD = f.loc[:, 'LOLD'].sum()
        if UseFlexibleDemand:
            # Add loss of load from flexible demand not fulfilled within a week
            LOLD += flexdem_uns['%s_flexloss'%area][flexdem_uns['%s_flexloss'%area] != 0].count()
        
        ## Store        
        fLOLD = pd.concat((fLOLD, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : area, 'Carrier' : 'Electricity', 'Value (h)' : LOLD}, index=[0])), ignore_index=True)
        
    # Hydrogen LOLD
    for area in A2B_regi_h2.keys(): 
        f = pd.read_table(wk_dir + '/Antares/output/' + ant_res + '/economy/mc-all/areas/%s/values-hourly.txt'%area.lower(),
                        skiprows=[0,1,2,3,5,6]) 
        
        ## Store        
        fLOLD = pd.concat((fLOLD, pd.DataFrame({'Iter' : i, 'Year' : year, 'Region' : area, 'Carrier' : 'Hydrogen', 'Value (h)' : f.loc[:, 'LOLD'].sum()}, index=[0])), ignore_index=True)
        
    ### LOLD has to be <= 3h for each year
    print('Total System LOLD in Year %d: %d\n'%(year, fLOLD.loc[(fLOLD.Iter == i) & (fLOLD.Year == year), 'Value (h)'].sum()))
    
    if USE_CAPCRED:
        # Get LOLD
        LOLD = fLOLD.loc[(fLOLD.Iter == i) & (fLOLD.Year == year) & (fLOLD.Carrier == 'Electricity')].groupby(by=['Region']).aggregate({'Value (h)' : np.sum})
        # If there's no reserve margin, the constraint is unbounding and is set to a converged region and year
        regions = LOLD.index
        for R in regions:
            for BalmArea in A2B_regi[R]:
                try:
                    temp = old_resmar.loc[str(year), R]
                except KeyError:
                    pass
                    # Necessary if you want capacity credit method to slack 
                    # if LOLD too low, but there's no reserve margin !
                    # (not important as long as convergence criterion isn't double-sided, e.g. 2 <= LOLD <= 3)
                    LOLD.loc[year, R] = 2.5 
        
        if year >= invest_year_start:           
            convergence = convergence & np.all(LOLD['Value (h)'] <= 3)
            print('Upper convergence until year %d: %s'%(year, convergence))
            convergence = convergence & np.all(LOLD['Value (h)'] >= 2)
            print('Lower convergence until year %d: %s'%(year, convergence))
        
        # Double-sided
        # convergence = convergence & np.all(LOLD['Value (h)'] <= 3) & np.all(LOLD['Value (h)'] >= 2) 
        
    else:
        convergence = convergence & np.all(fLOLD.loc[(fLOLD.Iter == i) & (fLOLD.Year == year) & (fLOLD.Carrier == 'Electricity'), 'Value (h)'] <= 3)
        print('Upper convergence until year %d: %s'%(year, convergence))
        convergence = convergence & np.all(fLOLD.loc[(fLOLD.Iter == i) & (fLOLD.Year == year) & (fLOLD.Carrier == 'Electricity'), 'Value (h)'] >= 2)
        print('Lower convergence until year %d: %s'%(year, convergence))

print('LOLD Results of Iteration %d:\n'%i)
print(fLOLD[fLOLD.Iter == i].to_string())
print('\nConvergence achieved: %s\n'%convergence)


# Save
fLOLD.to_csv('Workflow/OverallResults/%s_LOLD.csv'%SC)

print('\n|--------------------------------------------------|')   
print('              END OF CONVERGENCE-CRITERION')
print('|--------------------------------------------------|\n')   
