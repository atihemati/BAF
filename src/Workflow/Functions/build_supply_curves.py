"""
Piecewise Linear Fitting Functions

Monotonically decreasing piecewise linear fitting functions, one continuous and one discontinuous.

Created on 02.05.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling) and claude.ai 
"""

### ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pybalmorel import Balmorel

def get_supply_curve(x: np.array, y: np.array):
    
    df = pd.DataFrame({'x' : x,
                       'y' : y})

    # Sort values by descending x
    df=df.sort_values(by='x', ascending=False)
    df.index = np.arange(len(df))
    
    # The 'fitted' values
    fit_x, fit_y = [], []
    
    for i, row in df.iterrows():
        if i == 0:
            # Fill the first values
            if small_number_to_zero(row['y']) != 0:
                fit_x.append(small_number_to_zero(row['x'])+1e-3)
                fit_y.append(0)
            fit_x.append(small_number_to_zero(row['x']))
            fit_y.append(small_number_to_zero(row['y']))
        elif row['y'] > fit_y[-1]:
            # Add higher y at lower x as a stepwise linear function
            fit_x.append(small_number_to_zero(row['x'])+1e-3)
            fit_y.append(fit_y[-1])
            fit_x.append(small_number_to_zero(row['x']))
            fit_y.append(small_number_to_zero(row['y']))
        
    # Last point
    if fit_x[-1] != 0:
        fit_x.append(0)
        fit_y.append(fit_y[-1])

    return fit_x, fit_y

def small_number_to_zero(number: float):
    if number < 1e-6:
        number = 0
    return number

def find_closest_x(x0: float, xp: np.array, yp: np.array):
    try:
        x = list(xp).index(x0)
        y = yp[x]
    except ValueError:
        # x wasnt in xp
        diff = np.abs(xp - x0)
        closest_index = list(diff).index(diff.min())
        y = yp[closest_index]
    return y

def combine_step_curves(x1, y1, x2, y2):
    """
    Combine two step curves with different x-coordinates.
    
    Args:
        x1, y1: Coordinates of the first step curve
        x2, y2: Coordinates of the second step curve
        
    Returns:
        combined_x, combined_y: Coordinates of the combined step curve
    """
    # Combine and sort all unique x-coordinates
    all_x = np.unique(np.concatenate([x1, x2]))
    all_x[::-1].sort() # Sort descending
    
    # Initialize arrays for combined curve
    combined_x = []
    combined_y = []
    
    # Evaluate the first curve at all x points
    for i, x0 in enumerate(all_x):
        if i == 0:
            combined_x.append(x0)
            combined_y.append(0)
        combined_x.append(x0)
        combined_y.append(find_closest_x(x0, x1, y1) + find_closest_x(x0, x2, y2))
    
    return combined_x, combined_y

def combine_multiple_supply_curves(x_list, y_list):
    """
    Combine multiple step curves with different x-coordinates.
    
    Args:
        x_list: List of x-coordinate arrays for each curve
        y_list: List of y-coordinate arrays for each curve
        
    Returns:
        combined_x, combined_y: Coordinates of the combined step curve
    """
    # Verify input
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list must have the same length")
    
    if len(x_list) == 0:
        return np.array([]), np.array([])
    
    if len(x_list) == 1:
        return x_list[0], y_list[0]
    
    # Start with the first curve
    combined_x, combined_y = x_list[0], y_list[0]
    
    # Add each subsequent curve
    for i in range(1, len(x_list)):
        combined_x, combined_y = combine_step_curves(combined_x, combined_y, x_list[i], y_list[i])
    
    return combined_x, combined_y

# Example usage:
if __name__ == "__main__":
        
    m=Balmorel('Balmorel', gams_system_directory='/appl/gams/47.6.0')
    m.collect_results()

    scenario = 'base'
    area = 'DE_A'
    tech = 'GNR_BO_ELEC_E-80'
    tech = 'GNR_BO_ELEC_E-99_LS-10-MW-FEED_Y-2050'
    commodity = 'HEAT'
    commodity = 'HYDROGEN'
    # tech = 'ENDO_H2'
    # tech = 'ENDOGENOUS_ELECT2HEAT'

    supply_curves_x, supply_curves_y = [], []

    df1_temp = m.results.get_result('PRO_YCRAGFST')
    df2_temp = m.results.get_result('EL_PRICE_YCRST')
    for area in df1_temp.Area.unique():
        df1=df1_temp.query('Scenario==@scenario and Area==@area and Fuel=="ELECTRIC" and Commodity==@commodity').pivot_table(index=['Season','Time'], columns='Generation', values='Value')

        if 'region' in locals() and region != area[:2]:
            print(supply_curves_x, supply_curves_y)
            fig, ax = plt.subplots()
            combined_x, combined_y = combine_multiple_supply_curves(supply_curves_x, supply_curves_y)
            print(combined_x, combined_y)
            ax.plot(combined_x, combined_y)
            ax.set_title('Supply Curve for %s in %s'%(commodity, region))
            ax.set_ylabel('MWh')
            ax.set_xlabel('€/MWh')
            fig.savefig('supply_curve_%s_%s.png'%(commodity, region))
            supply_curves_x, supply_curves_y = [], []
        
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
            fit_x, fit_y = get_supply_curve(temp.loc[:, 'Value'].values.flatten(),
                                        temp.loc[:, tech].values.flatten())
            supply_curves_x.append(fit_x)
            supply_curves_y.append(fit_y)
                
            ax.plot(fit_x, fit_y)
                
            ax.set_ylabel(f'{tech} (MWh)')
            ax.set_xlabel('Electricity Price (€/MWh)')
            ax.set_title(area)
            fig.savefig(f'eldempricecurve_{area}_{tech}.png', bbox_inches='tight')