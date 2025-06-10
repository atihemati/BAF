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
import matplotlib.colors as mcolors
import pandas as pd
from pybalmorel import Balmorel, MainResults
from .GeneralHelperFunctions import get_balmorel_time_and_hours

### ------------------------------- ###
###           1. Functions          ###
### ------------------------------- ###


def get_inverse_residual_load(result: MainResults, scenario: str, year: int, hour_index: list):
    """Calculate inverse residual load for the supply curve fitting functions

    Args:
        result (MainResults): The MainResults class
        scenario (str): The scenario
        year (int): The model year

    Returns:
        pd.DataFrame: Parameters in the format expected by get_supply_curves
    """
    
    # Load data
    
    ## Get all indices from electricity price, which is guaranteed to have a value for all indices
    balmorel_index, all_index = get_balmorel_time_and_hours(result)
    year = str(year)
    
    ## Actual Results
    production = result.get_result('PRO_YCRAGFST').query('Scenario == @scenario and Year == @year').query('Technology in ["WIND-ON", "WIND-OFF", "SOLAR-PV", "HYDRO-RESERVOIRS", "HYDRO-RUN-OF-RIVER"]').pivot_table(index=['Region', 'Season', 'Time'], values=['Value'], aggfunc='sum').reindex(index=balmorel_index, fill_value=0)
    curtailment = result.get_result('CURT_YCRAGFST').query('Scenario == @scenario and Y == @year').pivot_table(index=['RRR', 'SSS', 'TTT'], values=['Value'], aggfunc='sum').rename(index={'RRR' : 'Region', 'SSS' : 'Season', 'TTT' : 'Time'}).reindex(index=balmorel_index, fill_value=0)
    el_demand = result.get_result('EL_DEMAND_YCRST').query('Scenario == @scenario and Year == @year').query('Category == "EXOGENOUS"').pivot_table(index=['Region', 'Season', 'Time'], values=['Value'], aggfunc='sum').reindex(index=balmorel_index, fill_value=0)
    heat_demand = result.get_result('H_DEMAND_YCRAST').query('Scenario == @scenario and Year == @year').query('Category == "EXOGENOUS"').pivot_table(index=['Region', 'Season', 'Time'], values=['Value'], aggfunc='sum').reindex(index=balmorel_index, fill_value=0)
    
    ## Calculate inverse residual load
    inverse_residual_load = (production + curtailment - el_demand - heat_demand)
    
    return inverse_residual_load.reset_index()

def get_heat_demand(result: MainResults, scenario: str, year: int, hour_index: list):
    """Calculate inverse residual load for the supply curve fitting functions

    Args:
        result (MainResults): The MainResults class
        scenario (str): The scenario
        year (int): The model year
        hour_index (list): The chosen hours in a year chosen as Balmorel resolution

    Returns:
        pd.DataFrame: Parameters in the format expected by get_supply_curves
    """
    
    year = str(year)
    
    heat_demand = result.get_result('H_DEMAND_YCRAST').query('Scenario == @scenario and Year == @year').query('Category == "EXOGENOUS"').pivot_table(index=['Region', 'Season', 'Time'], values=['Value'], aggfunc='sum').reindex(index=balmorel_index, fill_value=0)
    
    return heat_demand.reset_index()

def get_parameters_for_supply_curve_fit(result: MainResults, scenario: str, year: int, commodity: str, hour_index: list):
    """Get parameters for supply curve fitting depending on the commodity

    Args:
        commodity (str): Either 'HEAT' or 'HYDROGEN'
    
    Raises:
        ValueError: If choice is not 'HEAT' or 'HYDROGEN'

    Returns:
        parameters (pd.DataFrame): The parameters to fit with columns [parameter_name, 'Region', 'Season', 'Time'] 
    """
    
    if commodity.upper() == 'HEAT':
        return get_heat_demand(result, scenario, year, hour_index)
    elif commodity.upper() == 'HYDROGEN':
        return get_inverse_residual_load(result, scenario, year, hour_index)
    else:
        raise ValueError(f"Commodity '{commodity}' is not yet a part of this framework. Please choose 'HEAT' or 'HYDROGEN'")


def get_supply_curve(x: np.array, y: np.array):
    """A function to construct a supply curve depending on x and y data

    Args:
        x (np.array): Typically electricity prices in €/MWh
        y (np.array): Typically electricity demands in MWh

    Returns:
        fit_x, fit_y (list, list): The 'fitted' curve in x and y coordinates
    """
    
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

def seasonal_colors(num_seasons: int, style: str):
    """Create colors for all S01-SNM seasons, that are more red in the middle

    Args:
        num_seasons (int): Amount of seasons
    """

    colors = {}
    for i in range(1, num_seasons + 1):
        # Calculate how far we are from the midpoint (S26)
        # This gives a value between 0 (at S26) and 1 (at S01 or S52)
        distance_from_mid = abs(i - (num_seasons / 2 + 0.5)) / (num_seasons / 2)
        
        # Red component: max at midpoint, min at endpoints
        color = 1.0 - distance_from_mid
        # Black components: min at midpoint, max at endpoints
        other_color = distance_from_mid
        
        season_key = f'S{i:02d}'
        if style == 'report':
            colors[season_key] = mcolors.to_hex([color, 0, 0, 0.5])  # Red component varies, others fixed. .to_hex used to avoid color map warning
        else:
            colors[season_key] = mcolors.to_hex([1, other_color, other_color, 0.5])  # Red component varies, others fixed. .to_hex used to avoid color map warning
    
    return colors

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

def get_supply_curves(scenario: str, 
                      year: int, 
                      commodity: str, 
                      parameters: pd.DataFrame,
                      fuel_consumption: pd.DataFrame, 
                      el_prices: pd.DataFrame,
                      precision: int = -4,
                      plot_overall_curves: bool = False,
                      plot_all_curves: bool = False,
                      style: str = 'report'):
    """Create seasonal curves for hydrogen and heat for every region in a scenario 

    Args:
        scenario (str): Scenario to analyse
        year (int): The model year
        commodity (str): The commodity
        parameters (pd.DataFrame): The dataframe containing the parameter values for all regions, seasons and time steps with columns ['Region', 'Season', 'Time']
        fuel_consumption (pd.DataFrame): Fuel consumption results. 
        el_prices (pd.DataFrame): Electricity prices.
        precision (int): The precision to fit parameter values to (NOTE: This is problematic when you start having smaller countries in the scope! Should be made dependent on the absolute magnitude of profiles)
        plot_overall_curves (bool, optional): Plot regional heat and hydrogen supply curves?
        plot_all_curves (bool, optional): Plot stepfunction fit of ALL technologies and regional heat and hydrogen supply curves?
        style (str, optional): Style of supply curve plot. Defaults to 'report'.

    Returns:
        resulting_curves (dict): Price and capacities for all region and parameters
    """

    year = str(year)
    commodity2technology = {'HEAT' : 'ELECT-TO-HEAT', 'HYDROGEN' : 'ELECTROLYZER'}
    technology = commodity2technology[commodity]
    df1_temp = fuel_consumption.query('Year == @year and Technology == @technology')
    df2_temp = el_prices.query('Year == @year')
    
    # Prepare parameters to iterate through and colors for plotting them
    regions = df1_temp.Region.unique()
    parameter_name = [col for col in parameters.columns if not(col in ['Region', 'Season', 'Time'])][0]
    parameters = parameters.round({parameter_name : precision}) # NOTE: This will be a problem when you have smaller countries in the model

    # Prepare fit result data
    resulting_curves = {region : dict() for region in regions}      

    for region in regions:

        fig_season, ax_parameter = plt.subplots(facecolor='none')
        
        region_parameters = parameters.query('Region == @region')
        unique_parameters = region_parameters[parameter_name].unique()
        
        for parameter in unique_parameters:
            
            supply_curves_x, supply_curves_y = [], []
            season, time, parameter_value = region_parameters.query(f'{parameter_name} == @parameter')[['Season', 'Time', parameter_name]].values[0]
            
            for area in df1_temp.query('Region == @region').Area.unique():
                
                df1=df1_temp.query('Area==@area and Fuel=="ELECTRIC" and Season == @season and Time == @time').pivot_table(index=['Season', 'Time'], columns='Generation', values='Value')

                for tech in df1.columns:
                    
                    # Skip if very low fuel consumption
                    if df1.loc[:, tech].max() < 1e-5:
                        continue
                    
                    df2=df2_temp.query('Scenario==@scenario and Region==@region and Season == @season and Time == @time').pivot_table(index=['Season', 'Time'], values='Value')

                    temp=df1[[tech]].merge(df2[['Value']], left_index=True, right_index=True).fillna(0)

                    # Piecewise linear fit
                    fit_x, fit_y = get_supply_curve(temp.loc[:, 'Value'].values.flatten(),
                                                temp.loc[:, tech].values.flatten())
                    
                    supply_curves_x.append(fit_x)
                    supply_curves_y.append(fit_y)
                        
                    # Plot fit to data points for specific technology
                    if plot_all_curves:
                        fig, ax = plt.subplots()
                        temp.plot(kind='scatter', x='Value', y=tech, ax=ax, 
                                    label=parameter)
                        ax.plot(fit_x, fit_y)
                        ax.set_ylabel(f'{tech} (MWh)')
                        ax.set_xlabel('Electricity Price (€/MWh)')
                        ax.set_title(area)
                        ax.legend(loc='center left', bbox_to_anchor=(1.05, .5))
                        fig.savefig(f'Workflow/OverallResults/eldempricecurve_{commodity}_{area}_{tech}_{parameter_name}{parameter:0.2f}.png', bbox_inches='tight')
                    
            if len(supply_curves_x) != 0:
                combined_x, combined_y = combine_multiple_supply_curves(supply_curves_x, supply_curves_y)
            else:
                combined_x, combined_y = [0, 0], [0, 0]
                
            # Plot overall curve    
            if plot_all_curves or plot_overall_curves:
                ax_parameter.plot(combined_x, combined_y, label=parameter_value)

            # Store seasonal curves   
            resulting_curves[region][parameter_value] = {'price' : np.round(combined_x),
                                                        'capacity' : np.round(combined_y)}


        # Plot overall curve
        if plot_all_curves or plot_overall_curves:
            ax_parameter.set_title('Supply Curve for %s in %s'%(commodity, region))
            ax_parameter.set_ylabel('MWh')
            ax_parameter.set_xlabel('€/MWh')
            ax_parameter.set_facecolor('none')
            ax_parameter.legend(loc='center left', bbox_to_anchor=(1.05, .5))
            fig_season.savefig('Workflow/OverallResults/supply_curve_%s_%s.png'%(commodity, region),
                                bbox_inches='tight')
            
    return resulting_curves

### ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###
if __name__ == "__main__":
    
    # Example usage for one scenario
    scenario = 'baf_test_Iter0'
    year = 2050
    commodities = ['HEAT', 'HYDROGEN']
    resulting_curves = {}
    
    ## Get Balmorel Results
    gams_system_directory = 'C:/GAMS/47'
    m=Balmorel('Balmorel')
    m.locate_results()
    res = MainResults('MainResults_' + scenario + '.gdx',
                      paths='Balmorel/' + m.scname_to_scfolder[scenario] + '/model', 
                      system_directory=gams_system_directory)
    production = res.get_result('F_CONS_YCRAST')
    el_prices= res.get_result('EL_PRICE_YCRST')
    
    ## Prepare parameters to fit
    parameters = el_prices.round({'Value' : 0}).pivot_table(index=['Region', 'Season', 'Time'], values='Value').reset_index().rename(columns={'Value' : 'Elprice'})
    
    ## Make curves
    for commodity in commodities:  
        resulting_curves[commodity] = get_supply_curves(scenario, year, commodity, parameters, production, el_prices, -1, plot_overall_curves=True)
    