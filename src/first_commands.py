
from pybalmorel import Balmorel
import matplotlib.pyplot as plt
import numpy as np
from pwlf import PiecewiseLinFit
import copy

m=Balmorel('Balmorel', gams_system_directory='/appl/gams/47.6.0')
m.collect_results()

scenario = 'base'
region = 'DE'
area = 'DE_A'
tech = 'GNR_BO_ELEC_E-80'
tech = 'GNR_BO_ELEC_E-99_LS-10-MW-FEED_Y-2050'
# tech = 'ENDO_H2'
# tech = 'ENDOGENOUS_ELECT2HEAT'

df1=m.results.get_result('PRO_YCRAGFST').query('Scenario==@scenario and Area==@area').pivot_table(index=['Season','Time'], columns='Generation', values='Value')
# df1=m.results.get_result('EL_DEMAND_YCRST').query('Scenario=="base"').pivot_table(index=['Season','Time'], columns='Category', values='Value')
df2=m.results.get_result('EL_PRICE_YCRST').query('Scenario==@scenario and Region==@region').pivot_table(index=['Season','Time'], values='Value')

temp=df1[[tech]].merge(df2[['Value']],left_index=True, right_index=True).fillna(0)

seasons = list(temp.index.get_level_values(0).unique())
colors = {season : [seasons.index(season)/len(seasons)*2, 0, 0, .7] for season in seasons[:int(round(len(seasons)/2))]} | {season : [1-(seasons.index(season)-len(seasons)/2)/len(seasons)*2, 0, 0, .7] for season in seasons[int(round(len(seasons)/2)):]}
fig, ax = plt.subplots()
for season in seasons:
# for season in seasons[1:2]:
    # Scatter plot
    temp.loc[season].plot(kind='scatter', x='Value', y=tech, ax=ax, 
                          label=season, color=colors[season])
    
# Piecewise linear fit
fitting = PiecewiseLinFit(temp.loc[:, 'Value'].values.flatten(),
                                temp.loc[:, tech].values.flatten())
                                #    disp_res=True)          
                                
# Amount of segments and fit
fitting.fit(2)

x = np.linspace(0, max(temp.loc[season, 'Value'])*1.1)
ax.plot(x, fitting.predict(x), linestyle='--', color=colors[season])
    
ax.set_ylabel(f'{tech} (MWh)')
ax.set_xlabel('Electricity Price (€/MWh)')
ax.set_title(region)
fig.savefig(f'eldempricecurve_{region}_{tech}.png', bbox_inches='tight')