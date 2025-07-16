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
        supply_curves = pkl.load(f)
        
    commodities = supply_curves.keys()
    
    # Collect data
    data = []
    for commodity in commodities:
        
        regions = supply_curves[commodity].keys()
        
        for region in regions:
            
            # Get parameters in the index
            temp_df = pd.DataFrame(supply_curves[commodity][region]).T

            # Apply indices
            temp_df['commodity'] = commodity
            temp_df['region'] = region
            
            # Append
            data.append(temp_df)

    # Get the total dataframe
    df = pd.concat(data)

    # Simply supply curves to max values, as they will be because this assumes using all timeslices
    df.loc[:, ['price']] = df['price'].apply(max)
    df.loc[:, ['capacity']] = df['capacity'].apply(max)

    return df

def gaussian_kernel_smooth_2d(x1_obs, x2_obs, y_obs, x1_new, x2_new, bandwidth1, bandwidth2):
    """
    2D Gaussian kernel smoothing
    
    Parameters:
    - x1_obs, x2_obs: 1D arrays of observed x1 and x2 coordinates
    - y_obs: 1D array of observed y values
    - x1_new, x2_new: 1D arrays of new x1 and x2 coordinates where you want predictions
    - bandwidth1, bandwidth2: bandwidths for x1 and x2 dimensions
    
    Returns:
    - y_smooth: 1D array of smoothed y values at (x1_new, x2_new) points
    """
    y_smooth = np.zeros(len(x1_new))
    
    for i, (x1_target, x2_target) in enumerate(zip(x1_new, x2_new)):
        # Calculate 2D Gaussian weights
        weights = np.exp(
            -0.5 * (
                ((x1_obs - x1_target) / bandwidth1)**2 + 
                ((x2_obs - x2_target) / bandwidth2)**2
            )
        )
        weights = weights / np.sum(weights)
        y_smooth[i] = np.sum(weights * y_obs)
    
    return y_smooth

#%% ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
def main():

    commodity = 'HEAT'
    region = 'FR'
    output = 'capacity'
    parameter_x = 'exo_demand'
    parameter_y = 'VRE_availability'
    bandwidths = [[5e5, 1e3], [5e5, 1e4], [1e6, 1e4]]
    bandwidths = [[5e5, 1e4]]

    df = (
        load_and_format_supply_curves()
        .query(f'commodity == "{commodity}" and region == "{region}"')
    )
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = [fig.add_subplot(1, len(bandwidths), i+1, projection='3d') for i in range(len(bandwidths))]
    
    for i,bandwidth in enumerate(bandwidths):
        
        x_obs = getattr(df, parameter_x).astype(float)
        y_obs = getattr(df, parameter_y).astype(float)
        z_obs = getattr(df, output).astype(float)
        
        ax[i].scatter(x_obs, y_obs, z_obs, label=bandwidth)
        ax[i].set_xlabel(parameter_x)
        ax[i].set_ylabel(parameter_y)
        ax[i].set_zlabel(output)
    
        # Create 2D grid for surface plot
        min_param_x = getattr(df, parameter_x).min()
        max_param_x = getattr(df, parameter_x).max()
        x_smooth = np.linspace(min_param_x, max_param_x, 100)  # Reduced from 1000 for performance
        
        min_param_y = getattr(df, parameter_y).min()
        max_param_y = getattr(df, parameter_y).max()
        y_smooth = np.linspace(min_param_y, max_param_y, 100)  # Reduced from 1000 for performance
        
        # Create meshgrid
        X_grid, Y_grid = np.meshgrid(x_smooth, y_smooth)
        
        # Flatten for the 2D smoothing function
        X_flat = X_grid.flatten()
        Y_flat = Y_grid.flatten()
        
        smoothed = gaussian_kernel_smooth_2d(x_obs, y_obs, z_obs, X_flat, Y_flat, bandwidth[0], bandwidth[1])
        Z_grid = smoothed.reshape(X_grid.shape) # Reshape back to grid
        
        ax[i].plot_surface(X_grid, Y_grid, Z_grid, alpha=0.7, cmap='viridis')
        ax[i].set_title(f'Bandwidth: {bandwidth}')
    
        if output == 'price':
            ax[i].set_zlim([0, 500])
    
    ax[-1].legend((output, 'smoothed'))
    fig.subplots_adjust(hspace=0.7)
    plt.show()
    # fig.savefig('test.png')

if __name__ == '__main__':
    main()
