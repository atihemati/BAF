"""
Piecewise Linear Fitting Functions

Monotonically decreasing piecewise linear fitting functions, one continuous and one discontinuous.

Created on 02.05.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling) and claude.ai 
Conversation link: 
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_mac_curve(x: np.array, y: np.array):
    
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
                fit_x.append(small_number_to_zero(row['x']))
                fit_y.append(0)
            fit_x.append(small_number_to_zero(row['x']))
            fit_y.append(small_number_to_zero(row['y']))
        elif row['y'] > fit_y[-1]:
            # Add higher y at lower x as a stepwise linear function
            fit_x.append(small_number_to_zero(row['x']))
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
    all_x = np.concatenate([x1, x2]).unique()
    all_x[::-1].sort() # Sort descending
    
    # Initialize arrays for combined curve
    combined_x = all_x
    combined_y = []
    
    # Evaluate the first curve at all x points
    for i, x0 in enumerate(combined_x):
        combined_y[i] += find_closest_x(x0, x1, y1) + find_closest_x(x0, x2, y2) 
    
    return combined_x, combined_y

def combine_multiple_mac_curves(x_list, y_list):
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
    pass