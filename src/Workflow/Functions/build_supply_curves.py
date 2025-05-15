"""
Building Supply Curves of Balmorel Results

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

if __name__ == "__main__":
    
    # Example usage for one scenario
    scenario = 'baf_test_Iter0'

    # Create a color dictionary for seasons
    colors = {}
    num_seasons = 52
    for i in range(1, num_seasons + 1):
        # Calculate how far we are from the midpoint (S26)
        # This gives a value between 0 (at S26) and 1 (at S01 or S52)
        distance_from_mid = abs(i - (num_seasons / 2 + 0.5)) / (num_seasons / 2)
        
        # Red component: max at midpoint, min at endpoints
        red = 1.0 - distance_from_mid
        # Black components: min at midpoint, max at endpoints
        black = distance_from_mid
        
        season_key = f'S{i:02d}'
        colors[season_key] = [red, 0, 0, 0.5]  # Red component varies, others fixed

    m=Balmorel('Balmorel')
    m.collect_results()
    df1_temp = m.results.get_result('PRO_YCRAGFST').query('Scenario == @scenario')
    df2_temp = m.results.get_result('EL_PRICE_YCRST').query('Scenario == @scenario')
    
    seasons = list(df1_temp.Season.unique())
    seasons.sort()
    
    for commodity in ['HEAT', 'HYDROGEN']:  
    
        for region in df1_temp.Region.unique():
                    
            fig_season, ax_season = plt.subplots()
            for season in seasons:
            
                supply_curves_x, supply_curves_y = [], []
                
                for area in df1_temp.query('Region == @region').Area.unique():
                    
                    df1=df1_temp.query('Area==@area and Fuel=="ELECTRIC" and Commodity==@commodity and Season == @season').pivot_table(index='Time', columns='Generation', values='Value')

                    for tech in df1.columns:
                        
                        # Skip if very low max production
                        if df1.loc[:, tech].max() < 1e-5:
                            continue
                        
                        # df1=m.results.get_result('EL_DEMAND_YCRST').query('Scenario=="base"').pivot_table(index=['Season','Time'], columns='Category', values='Value')
                        df2=df2_temp.query('Scenario==@scenario and Region==@region and Season == @season').pivot_table(index='Time', values='Value')

                        temp=df1[[tech]].merge(df2[['Value']],left_index=True, right_index=True).fillna(0)

                        
                        # fig, ax = plt.subplots()
                        # for season in seasons:
                        #     temp.loc[season].plot(kind='scatter', x='Value', y=tech, ax=ax, 
                        #                         label=season, color=colors[season])
                        
                        # Piecewise linear fit
                        fit_x, fit_y = get_supply_curve(temp.loc[:, 'Value'].values.flatten(),
                                                    temp.loc[:, tech].values.flatten())
                        supply_curves_x.append(fit_x)
                        supply_curves_y.append(fit_y)
                            
                        # ax.plot(fit_x, fit_y)
                        # ax.set_ylabel(f'{tech} (MWh)')
                        # ax.set_xlabel('Electricity Price (€/MWh)')
                        # ax.set_title(area)
                        # fig.savefig(f'eldempricecurve_{area}_{tech}.png', bbox_inches='tight')
                        
                if len(supply_curves_x) != 0:
                    combined_x, combined_y = combine_multiple_supply_curves(supply_curves_x, supply_curves_y)
                    ax_season.plot(combined_x, combined_y, color=colors[season], label=season)
            
            ax_season.set_title('Supply Curve for %s in %s'%(commodity, region))
            ax_season.set_ylabel('MWh')
            ax_season.set_xlabel('€/MWh')    
            ax_season.legend()    
            fig_season.savefig('Workflow/OverallResults/supply_curve_%s_%s.png'%(commodity, region))