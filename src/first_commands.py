
from pybalmorel import Balmorel
import matplotlib.pyplot as plt
import numpy as np
from Workflow.Functions.piecewise_linear_fits import get_mac_curve, combine_multiple_mac_curves

m=Balmorel('Balmorel', gams_system_directory='/appl/gams/47.6.0')
m.collect_results()

scenario = 'base'
area = 'DE_A'
tech = 'GNR_BO_ELEC_E-80'
tech = 'GNR_BO_ELEC_E-99_LS-10-MW-FEED_Y-2050'
commodity = 'HEAT'
# tech = 'ENDO_H2'
# tech = 'ENDOGENOUS_ELECT2HEAT'

mac_curves_x, mac_curves_y = [], []

df1_temp = m.results.get_result('PRO_YCRAGFST')
df2_temp = m.results.get_result('EL_PRICE_YCRST')
for area in df1_temp.Area.unique():
    df1=df1_temp.query('Scenario==@scenario and Area==@area and Fuel=="ELECTRIC" and Commodity==@commodity').pivot_table(index=['Season','Time'], columns='Generation', values='Value')

    if 'region' in locals() and region != area[:2]:
        print(mac_curves_x, mac_curves_y)
        fig, ax = plt.subplots()
        combined_x, combined_y = combine_multiple_mac_curves(mac_curves_x, mac_curves_y)
        print(combined_x, combined_y)
        ax.plot(combined_x, combined_y)
        ax.set_title('MAC Curve for %s in %s'%(commodity, region))
        ax.set_ylabel('MWh')
        ax.set_xlabel('€/MWh')
        fig.savefig('mac_curve_%s_%s.png'%(commodity, region))
        mac_curves_x, mac_curves_y = [], []
    
    region = area[:2]

    for tech in df1.columns:
        
        # Skip if very low max production
        if df1.loc[:, tech].max() < 1e-5:
            continue
        
        # df1=m.results.get_result('EL_DEMAND_YCRST').query('Scenario=="base"').pivot_table(index=['Season','Time'], columns='Category', values='Value')
        df2=df2_temp.query('Scenario==@scenario and Region==@region').pivot_table(index=['Season','Time'], values='Value')

        temp=df1[[tech]].merge(df2[['Value']],left_index=True, right_index=True).fillna(0)

        seasons = list(temp.index.get_level_values(0).unique())
        colors = {season : [seasons.index(season)/len(seasons)*2, 0, 0, .7] for season in seasons[:int(round(len(seasons)/2))]} | {season : [1-(seasons.index(season)-len(seasons)/2)/len(seasons)*2, 0, 0, .7] for season in seasons[int(round(len(seasons)/2)):]}
        fig, ax = plt.subplots()
        
        for season in seasons:
            temp.loc[season].plot(kind='scatter', x='Value', y=tech, ax=ax, 
                                label=season, color=colors[season])
           
        # Piecewise linear fit
        fit_x, fit_y = get_mac_curve(temp.loc[:, 'Value'].values.flatten(),
                                     temp.loc[:, tech].values.flatten())
        mac_curves_x.append(fit_x)
        mac_curves_y.append(fit_y)
            
        ax.plot(fit_x, fit_y)
            
        ax.set_ylabel(f'{tech} (MWh)')
        ax.set_xlabel('Electricity Price (€/MWh)')
        ax.set_title(area)
        fig.savefig(f'eldempricecurve_{area}_{tech}.png', bbox_inches='tight')