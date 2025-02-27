"""
Created on 25.01.2024

@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)

Script for doing timeseries aggregation.
NOTE - Requires the Profiles.gdx, which can be extracted from Balmorel by
placing the following command in Balmorelbb4.inc:
execute_unload  'Profiles.gdx' , WND_VAR_T, WNDFLH, SOLE_VAR_T, SOLEFLH, WTRRRVAR_T, WTRRSFLH, DE_VAR_T, DE;

It should be placed after:
* All remaining (cf. ... .inc above) data files af included from the following file
$ifi     exist "bb4datainc.inc" $include                   "bb4datainc.inc";
$ifi not exist "bb4datainc.inc" $include  "../../base/model/bb4datainc.inc";
..which would be line 291 in my Balmorelbb4.inc

NOTE - Requires the time series aggregation module:
pip install tsam

Docs: https://tsam.readthedocs.io/en/latest/gettingStartedDoc.html
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gams
import os
if 'Timeseries Aggregation' in __file__:
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
elif ('Workflow' in __file__) | ('Pre-Processing' in __file__):
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

from Workflow.Functions.GeneralHelperFunctions import symbol_to_df, doLDC, IncFile, ReadIncFilePrefix
try:
    import tsam.timeseriesaggregation as tsam
except ModuleNotFoundError:
    print('You need to install tsam to run this script:\npip install tsam')

style = 'report'

if style == 'report':
    plt.style.use('default')
    fc = 'white'
elif style == 'ppt':
    plt.style.use('dark_background')
    fc = 'none'
    

def format_and_save_profiles(typPeriods, agg_method, wy, Nperiods):
    ### Create All S and T index
    S = np.array(['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)])
    T = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)]

    # Make evenly distributed S and T for Balmorel input, based on Nperiods
    S = list(S[np.linspace(0, 51, Nperiods[0]).round().astype(int)])
    T = T[:Nperiods[1]]
    
    ### Scenario Name       
    if agg_method == "distributionAndMinMaxRepresentation":
        aggregation_scenario = 'W%dT%d_%s_WY%d'%(Nperiods[0], Nperiods[1], 'distminmax', wy) 
    else:
        aggregation_scenario = 'W%dT%d_%s_WY%d'%(Nperiods[0], Nperiods[1], agg_method[:4], wy) 
        
    try:
        os.mkdir('Balmorel/%s'%aggregation_scenario)
        os.mkdir('Balmorel/%s/data'%aggregation_scenario)
    except FileExistsError:
        pass


    # Formatting typical periods
    balmseries = typPeriods.copy()
    balmseries.index = pd.MultiIndex.from_product((S, T), names=['S', 'T'])

    s_series = balmseries.loc[(slice(None), 'T001'), 'Reservoir']
    s_series.index = s_series.index.get_level_values(0)
    s_series.index.name = ''

    balmseries.index = balmseries.index.get_level_values(0) + ' . ' + balmseries.index.get_level_values(1) 

    loadseries = balmseries['Load']
    loadseries.index = 'RESE . ' + loadseries.index


    ### 4.3 Save Profiles
    incfile_prefix_path = 'Pre-Processing/Data/IncFile Prefixes'
    incfiles = {'S' : IncFile(name='S', path='Balmorel/%s/data/'%aggregation_scenario,
                            prefix="SET S(SSS)  'Seasons in the simulation'\n/\n",
                            body=', '.join(S),
                            suffix="\n/;"),
                'T' : IncFile(name='T', path='Balmorel/%s/data/'%aggregation_scenario,
                            prefix="SET T(TTT)  'Time periods within a season in the simulation'\n/\n",
                            body=', '.join(T),
                            suffix="\n/;")}
    for incfile in ['ANTBALM_DE_VAR_T', 'ANTBALM_WTRRRVAR_T', 'ANTBALM_WTRRSVAR_S', 'ANTBALM_WND_VAR_T', 'ANTBALM_SOLE_VAR_T']:
        incfiles[incfile] = IncFile(name=incfile, prefix=ReadIncFilePrefix(incfile, incfile_prefix_path, wy),
                                    path='Balmorel/%s/data/'%aggregation_scenario)

    # Set bodies
    incfiles['ANTBALM_DE_VAR_T'].body = loadseries.to_string()
    incfiles['ANTBALM_WTRRRVAR_T'].body = balmseries['RoR'].to_string()
    incfiles['ANTBALM_WTRRSVAR_S'].body = s_series.to_string()
    incfiles['ANTBALM_WND_VAR_T'].body = balmseries['Wind'].to_string()
    incfiles['ANTBALM_SOLE_VAR_T'].body = balmseries['Solar'].to_string()

    # Suffix' (hardcoded! Also in Pre-Processing - make .inc files for the suffix as well for more systematic approach..)
    incfiles['ANTBALM_DE_VAR_T'].suffix = """\n;\nDE_VAR_T(RRR,'RESE',SSS,TTT) =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    DE_VAR_T(RRR,'OTHER',SSS,TTT) =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    DE_VAR_T(RRR,'DATACENTER',SSS,TTT)$SUM(YYY,DE(YYY,RRR,'DATACENTER'))  =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    DE_VAR_T(RRR,'PII',SSS,TTT) = ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);
    ANTBALM_DE_VAR_T(DEUSER,SSS,TTT,RRR)=0;\n$label USEANTARESDATAEND"""
    incfiles['ANTBALM_WTRRRVAR_T'].suffix   = "\n;\nWTRRRVAR_T(AAA,SSS,TTT) = WTRRRVAR_T1(SSS,TTT,AAA);\nWTRRRVAR_T1(SSS,TTT,AAA) = 0;\n$label dont_adjust_hydro"
    incfiles['ANTBALM_WND_VAR_T'].suffix = '\n;\nWND_VAR_T(IA,SSS,TTT)$WND_VAR_T2(SSS,TTT,IA) = WND_VAR_T2(SSS,TTT,IA);\nWND_VAR_T2(SSS,TTT,AAA)=0;\n$label USEANTARESDATAEND'
    incfiles['ANTBALM_SOLE_VAR_T'].suffix = '\n;\nSOLE_VAR_T(IA,SSS,TTT)$SOLE_VAR_T2(SSS,TTT,IA) = SOLE_VAR_T2(SSS,TTT,IA);\nSOLE_VAR_T2(SSS,TTT,AAA)=0;\n$label USEANTARESDATAEND'
    incfiles['ANTBALM_WTRRSVAR_S'].suffix   = "\n;\nWTRRSVAR_S(AAA,SSS) = WTRRSVAR_S1(SSS,AAA);\nWTRRSVAR_S1(SSS,AAA) = 0;\n$label dont_adjust_hydro"

    ## Save
    for key in incfiles.keys():
        incfiles[key].save()


#%% ------------------------------- ###
###         1. Reading File         ###
### ------------------------------- ###

### 1.0 Define used weather year (read .inc file description)
wy = 2009 # 1982 = 1, 2012 = 31

### 1.1 Read Profiles GDX
ws = gams.GamsWorkspace()
db = ws.add_database_from_gdx(os.path.abspath('Pre-Processing/Data/Profiles_WY%d.gdx'%wy))

### 1.2 Get S-T Timeseries
# Wind
df = symbol_to_df(db, 'WND_VAR_T', ['A', 'S', 'T', 'Wind']).pivot_table(index=['S', 'T'], columns=['A'], values=['Wind'], fill_value=0)

# Solar
df = df.join(symbol_to_df(db, 'SOLE_VAR_T', ['A', 'S', 'T', 'Solar']).pivot_table(index=['S', 'T'], columns=['A'], values=['Solar'], fill_value=0),
                how='outer').fillna(0)

# El. Load
df2 = symbol_to_df(db, 'DE_VAR_T', ['R', 'Type', 'S', 'T', 'Load']).pivot_table(index=['R', 'S', 'T'], columns=['Type'], values=['Load'])
df2 = df2['Load'].pivot_table(index=['S', 'T'], columns=['R'], values=['RESE'], fill_value=0)
# Assuming RESE, PII, OTHER and DATACENTER have same load profiles!
df2.columns = pd.MultiIndex.from_product([['Load'], df2.columns.get_level_values(1)])
df = df.join(df2, how='outer').fillna(0)

# Run-of-River
df = df.join(symbol_to_df(db, 'WTRRRVAR_T', ['A', 'S', 'T', 'RoR']).pivot_table(index=['S', 'T'], columns=['A'], values=['RoR'], fill_value=0),
                how='outer').fillna(0)

#%% 1.2 Get S Timeseries

## Reservoir Inflows
WTRRSVAR_S = symbol_to_df(db, 'WTRRSVAR_S', ['A', 'S', 'Reservoir'])
WTRRSVAR_S['T'] = 'T001' # Add T dimension
df2 = WTRRSVAR_S.pivot_table(index=['S', 'T'], columns=['A'], values=['Reservoir']).fillna(0)
df = df.join(df2, how='outer')
# Set all T values to T001
T = df.index.get_level_values(1).unique()
for T0 in T[1:]:
    df.loc[(slice(None), T0), 'Reservoir'] = df.loc[(slice(None), 'T001'), 'Reservoir'].values

## Fuel Potential (at the moment irrelevant, as they are constant for all S)
# GMAXFS = symbol_to_df(db, 'GMAXFS', ['Y', 'CRA', 'F', 'S', 'Potential'])
# GMAXFS['T'] = 'T001'
# GMAXFS = GMAXFS[GMAXFS.Y == '2050'] # Filter year in biomass potential (most important to be strict in 2050)
# df2 = GMAXFS.pivot_table(index=['S', 'T'], columns=['F', 'CRA'], values=['Potential']).fillna(0)


# Extracting a value:
df.loc[('S01', 'T001'), ('Wind', 'AL00_A')]
# Extracting many values
df.loc[('S26', slice(None)), ('Solar', 'AL00_A')]

# Create index
try:
    df.index = pd.date_range('%d-01-01 00:00'%wy, '%d-12-30 23:00'%wy, freq='h') # 8760 h
except ValueError:
    df.index = pd.date_range('%d-01-01 00:00'%wy, '%d-12-29 23:00'%wy, freq='h') # 8736 h

#%% ------------------------------- ###
###     2. Using a Random Choice    ###
### ------------------------------- ###

### 2.1 Make random time aggregation
# 26*21, 20*24, 52*8, 10*24, 5*24, 5*8 <- to compare with tsam aggregations
N_S = 5
N_T = 8
N_timeslices = N_S * N_T
N_hours = len(df)

# Make choices
agg_steps = []
for i in range(N_timeslices):
    agg_steps.append(np.random.randint(N_hours))

# Sort chronologically
agg_steps.sort()

format_and_save_profiles(df.iloc[agg_steps], 'random', wy, (N_S, N_T))

# Also save a small note with the chosen timesteps
with open('Balmorel/%s/picked_times.txt'%('W%dT%d_rand_WY%d'%(N_S, N_T, wy)), 'w') as f:
    f.write(pd.Series(df.iloc[agg_steps].index).to_string())

#%% ------------------------------- ###
###          3. Using tsam          ###
### ------------------------------- ###

### 3.0 Settings
typical_periods = 5
hours_per_period = 24
# agg_method = "distributionAndMinMaxRepresentation"
# agg_method = "meanRepresentation"
agg_method = "medoidRepresentation"

### 3.1 Create Aggregation Object
aggregation = tsam.TimeSeriesAggregation(df, 
                                         noTypicalPeriods=typical_periods,
                                         hoursPerPeriod=hours_per_period,
                                         segmentation=True,
                                         noSegments=hours_per_period,
                                         representationMethod=agg_method,
                                         distributionPeriodWise=False,
                                         clusterMethod='hierarchical',
                                         )

typPeriods = aggregation.createTypicalPeriods()


### 3.2 Inspect Full Profiles
agg_func = 'median'

for col in df.columns.get_level_values(0).unique():
    fig, ax = plt.subplots()
    ax.set_title(col)

    dur, cur = doLDC(eval('df[col].%s(axis=1)'%agg_func), n_bins=1000)

    ax.plot(np.cumsum(dur), cur, label='Original')

    dur, cur = doLDC(eval('typPeriods[col].%s(axis=1)'%agg_func), n_bins=1000)
    ax.plot(np.cumsum(dur)*8736/len(typPeriods), cur, label='Aggregated')

    ax.legend()

### 3.3 A Snapshot of the full Profile compared to the aggregated one
fig, ax = plt.subplots()
typPeriods.loc[:, ('Load', 'DK1')].plot()
fig, ax = plt.subplots()
df['Load', 'DK1'].plot()

fig, ax = plt.subplots()
typPeriods.loc[:, ('Reservoir', 'AL00_hydro0')].plot()
fig, ax = plt.subplots()
df['Reservoir', 'AL00_hydro0'].plot()

format_and_save_profiles(typPeriods, agg_method, wy, (typical_periods, hours_per_period))
