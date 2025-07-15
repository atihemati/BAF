"""
TITLE

Description

Created on 15.07.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import click

#%% ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###

def load_and_format_supply_curves():
    
    with open('supply_curves.pkl', 'rb') as f:
        f = pkl.load(f)
        
    commodities = f.keys()
    
    # Collect data
    data = []
    for commodity in commodities:
        
        regions = f[commodity].keys()
        
        for region in regions:
            
            # Get parameters in the index
            temp_df = pd.DataFrame(f[commodity][region]).T
            
            # Apply indices
            temp_df['commodity'] = commodity
            temp_df['region'] = region
            temp_df.index.name = 'parameter'
            
            # Append
            data.append(temp_df.reset_index())

    # Get the total dataframe
    df = pd.concat(data)
    
    # Get maximum values (only one point, not actual supply curves, when we do it this way)
    df.loc[:, ['price']] = df['price'].apply(max)
    df.loc[:, ['capacity']] = df['capacity'].apply(max)

    return df

def gaussian_kernel_smooth(x_obs, y_obs, x_new, bandwidth): 
	y_smooth = np.zeros_like(x_new) 
	
	for i, x_target in enumerate(x_new): 
		weights = np.exp(
			-0.5 * ((x_obs - x_target) / bandwidth)**2
		) 
		weights = weights / np.sum(weights) 
		y_smooth[i] = np.sum(weights * y_obs) 
	
	return y_smooth

#%% ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
def main():

    df = load_and_format_supply_curves()
    
    fig, ax = plt.subplots()
    df.query('commodity == "HYDROGEN" and region == "DE"').plot(x='parameter', y='price', kind='scatter', ax=ax)
    df.query('commodity == "HYDROGEN" and region == "DE"').plot(x='parameter', y='capacity', kind='scatter', ax=ax, color='r')
    ax.legend(('Price', 'Capacity'))
    plt.show()

if __name__ == '__main__':
    main()
