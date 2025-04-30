#%% -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:02:08 2022

@author: mathi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
sys.path.append(r"../../Workflow/Functions")
from Formatting import newplot
from scipy.optimize import curve_fit

style = 'ppt'

if style == 'report':
    plt.style.use('default')
    fc = 'white'
    fc2 = 'white'
elif style == 'ppt':
    plt.style.use('dark_background')
    fc = 'none'
    fc2 = 'black'

### ----------------------------- ###
###         1. Parameters         ###
### ----------------------------- ###

### 1.1 Pick timesteps
# S = ['S07', 'S23', 'S36', 'S49']
# S = ['S08', 'S22', 'S36', 'S50']
S = ['S0%d'%i for i in range(1, 10, 2)] + ['S%d'%i for i in range(10, 53, 2)] # The used resolution at the moment
S = ['S0%d'%i for i in range(1, 10, 4)] + ['S%d'%i for i in range(10, 53, 4)] # 42 h
S = ['S0%d'%i for i in range(1, 10, 4)] + ['S%d'%i for i in range(13, 53, 4)] # Every fourth
S = ['S0%d'%i for i in range(1, 10, 8)] + ['S%d'%i for i in range(17, 53, 8)] # Every 8th 
S = ['S01', 'S27']
# T = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)] # All terms
# T = ['T00%d'%i for i in range(1, 10, 3)] + ['T0%d'%i for i in range(10, 100, 3)] + ['T%d'%i for i in range(100, 169, 3)] # All terms
# T = ['T00%d'%i for i in range(1, 10, 3)] + ['T0%d'%i for i in range(10, 24, 3)] # The used resolution at the moment
T = ['T001', 'T008', 'T014']
timesteps = pd.DataFrame(index=pd.MultiIndex.from_product([S, T], names=['S', 'T'])).reset_index()
timesteps = pd.Series(timesteps['S']).str[:] + '.' + pd.Series(timesteps['T']).str[:]

print(str(S).replace('[','').replace(']','').replace("'",''))
print(str(T).replace('[','').replace(']','').replace("'",''))
print("Total hours: %d"%(len(S)*len(T)))

### 1.2 Pick regions
R = ['DK', 'DE']

read = 'n' # Read input files?

normalise = 'n' # Normalise graphs to max value in timeseries ?

### 1.3 LDC and Plot Function
def doLDC(file, cols, idx, title=''):
    ncols = int(len(cols)**(1/2))+1
    nrows = int(len(cols)**(1/2))
    
    fig, axes = newplot(nrows=nrows, ncols=ncols, figsize=(8,5))
    axes = np.asarray(axes).reshape(-1) # Convert axes to array
    i = 0
    for c in cols:
        # Extract profile
        data_ht = np.histogram(file[c], bins=50)
        data_lt = np.histogram(file[c][idx], bins=50)
        lt_hours = idx.sum() # Hours in low time res is the amount of "true" in idx
        
        # Plot
        if normalise == 'y':
            axes[i].plot((np.cumsum(data_ht[0][::-1])/8736*100), data_ht[1][:-1][::-1]/max(data_ht[1])*100, label='8736h')
            axes[i].plot((np.cumsum(data_lt[0][::-1])/lt_hours*100), data_lt[1][:-1][::-1]/max(data_lt[1])*100, label='%dh'%(lt_hours))
            axes[i].set_ylim([0, 100])
            axes[i].text(45, 80, c, 
                     bbox=dict(facecolor=fc2, alpha=.5, edgecolor='none'))
        else:
            axes[i].plot((np.cumsum(data_ht[0][::-1])/8736*100), data_ht[1][:-1][::-1], label='8736h')
            axes[i].plot((np.cumsum(data_lt[0][::-1])/lt_hours*100), data_lt[1][:-1][::-1], label='%dh'%(lt_hours))
            axes[i].text(45, max(data_ht[1])*0.8, c, 
                     bbox=dict(facecolor=fc2, alpha=.5, edgecolor='none'))
         
        #axes[i].set_title(c)
        # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        # verticalalignment='top', bbox=dict(boxstyle='square,pad=.6',facecolor='lightgrey', edgecolor='black', alpha=0.7))

        # Better if only one plot
        # if title != '':
        #     axes[i].set_ylabel(title)    
        if i % (nrows + 1) == 0:
            axes[i].set_ylabel(title)
        else:
            axes[i].set_yticklabels([])
        # axes[i].set_title(c)
        # axes[i].legend()
        if (i >= nrows * (ncols - 1) - 1):
            axes[i].set_xlabel('Duration [%]')
        else:
            axes[i].set_xticklabels([])
        i += 1
            
    # Other finesse
    if len(axes) > len(cols):
        fig.delaxes(axes[i])
        axes[i - 1].legend(loc='center', bbox_to_anchor=(1.7, .4))
    else:
        axes[round(nrows/2)].legend(loc='center', bbox_to_anchor=(.5, 1.2), ncol=2)
    
    
    return fig, axes

#%% ----------------------------- ###
###           2. Wind DC          ###
### ----------------------------- ###

if read == 'y':
    fW = pd.read_excel('DurationCurveData.xlsx', sheet_name='WIND')
    
    # NO1 is assumed to have same wind profile as NO2, while
    # NO5 is assumed to have same wind profile as NO5
    fW['NO1_A1'] = fW['NO2_A1']
    fW['NO5_A1'] = fW['NO4_A1']

## Area filter
col = pd.Series(fW.columns)
idx = np.ones(len(col)) == 0
for r in R:
    idx = idx | ((col.str.find(r) != -1) & (col.str.find('VRE') == -1)) 
cols = list(col[idx])
cols.sort()

## Time filter
t = fW['Time']
idxT = np.ones(len(t)) == 0
for i in range(len(timesteps)):
    idxT = (t.str.find(timesteps[i]) == 0) | idxT
    
fig1, axes1 = doLDC(fW, cols, idxT, 'Wind CF [%]')
fig1.savefig('WindDC.png', transparent=True, bbox_inches='tight')


#%% ----------------------------- ###
###          3. Solar DC          ###
### ----------------------------- ###

if read == 'y':
    fS = pd.read_excel('DurationCurveData.xlsx', sheet_name='SOL')

## Area filter
col = pd.Series(fS.columns)
idx = np.ones(len(col)) == 0
for r in R:
    idx = idx | ((col.str.find(r) != -1) & (col.str.find('VRE') == -1)) 
cols = list(col[idx])
cols.sort()

fig2, axes2 = doLDC(fS, cols, idxT, 'Solar CF')

fig2.savefig('SolarDC.png', transparent=True, bbox_inches='tight')



#%% ----------------------------- ###
###      4. District Heat LDC     ###
### ----------------------------- ###

if read == 'y':
    fDH = pd.read_excel('DurationCurveData.xlsx', sheet_name='DH')

cols = ['DK2_Large', 'DK1_Large', 'NO1_A3', 'NO2_A2',
        'NO3_A3', 'NO4_A2', 'NO5_A2', 'SE1_medium', 'SE2_medium', 
        'SE3_large', 'SE4_large']

fig3, axes3 = doLDC(fDH, cols, idxT, 'DH Capacity Req.')

fig3.savefig('DHLDC.png', transparent=True, bbox_inches='tight')



#%% ----------------------------- ###
###      5. El. Capacity Req.     ###
### ----------------------------- ###

if read == 'y':
    fEL = pd.read_excel('DurationCurveData.xlsx', sheet_name='EL')
    fEL = fEL.dropna()
    

cols = ['DK1', 'DK2', 'DE4-N', 'DE4-E', 'DE4-S', 'DE4-W']

fig4, axes4 = doLDC(fEL, cols, idxT, 'El. Capacity Req.')


fig4.savefig('ElLDC.png', transparent=True, bbox_inches='tight')




