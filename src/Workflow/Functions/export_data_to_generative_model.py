"""
TITLE

Description

Created on 14.07.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pybalmorel import Balmorel
from pybalmorel.utils import symbol_to_df
from configparser import ConfigParser
import click

#%% ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###

def export_to_generative_model(scenario_folder: str, 
                               weather_year: int, 
                               filename: str,
                               parameters: list = [
                                    'DH_VAR_T',
                                    'DE_VAR_T',
                                    'DH',
                                    'WND_VAR_T', 
                                    'SOLE_VAR_T',
                                    'WTRRRVAR_T',
                                    'WTRRSVAR_S',
                                    'WNDFLH',
                                    'SOLEFLH', 
                                    'WTRRRFLH',
                                    'WTRRSFLH',
                               ],
                               ):
    
    model = Balmorel('Balmorel')
    model.load_incfiles(scenario_folder, overwrite=True)

    # Verify weather year
    with open(f'Balmorel/{scenario_folder}/data/DH_VAR_T.inc', 'r') as f:
        content = f.read()
        read_weather_year = int(content.split('Weather year ')[1].split('\n')[0])

    if read_weather_year != weather_year:
        raise ValueError(f"Read weather year was {read_weather_year}, but process expect {weather_year}!")

    print('Weather year', weather_year)

    for parameter in parameters:
        print(parameter)
        df = correct_format(symbol_to_df(model.input_data[scenario_folder], parameter), parameter, weather_year)
        
        if parameter == parameters[0]:
            df.T.to_csv(filename%weather_year, header=True)
        else:
            df.T.to_csv(filename%weather_year, mode='a', header=False)
    

def correct_format(df: pd.DataFrame, parameter: str, weather_year: int):
    
    # Extract unique dimensions
    unique_sets = [col for col in df.columns if not(col in ['SSS', 'TTT', 'Value'])]
    # Remove unnecessary model years
    if 'YYY' in df.columns:
        df = df.query('YYY in ["2030", "2040", "2050"]')
    
    # Add sets
    df['WY'] = weather_year
    df['Parameter'] = parameter + "|" + df[unique_sets].astype(str).agg("|".join, axis=1)
    
    # Add missing temporal sets
    if not('SSS' in df.columns):
        S_values = [f'S{i:02d}' for i in range(1, 53)]
        len_before = len(df)
        df = pd.concat([df]*52, ignore_index=True)
        df['SSS'] = np.tile(S_values, len_before)
    if not('TTT' in df.columns):
        T_values = [f'T{i:03d}' for i in range(1, 169)]
        len_before = len(df)
        df = pd.concat([df]*168, ignore_index=True)
        df['TTT'] = np.tile(T_values, len_before)
    
    sum_before = df.Value.sum()
    df = df.pivot_table(index=['WY', 'SSS', 'TTT'], columns='Parameter', values='Value', aggfunc='sum', fill_value=0)
    sum_after = df.sum().sum()

    if not(np.isclose(sum_before, sum_after, atol=0.1)):
        raise ValueError(f'Aggregation didnt work!\nSum before: {sum_before}\nSum after: {sum_after}')

    return df

def store_config(config: ConfigParser):
    
    with open('Config.ini', 'w') as f:
        config.write(f)

#%% ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
@click.argument('scenario_folder', type=str)
def main(scenario_folder: str):


    # Load configuration
    config = ConfigParser()
    config.read('Config.ini')
        
    weather_years = [1982, 1983, 1984, 1985, 1986, 
                    1987, 1988, 1989, 1990, 1991, 
                    1992, 1993, 1994, 1995, 1996, 
                    1997, 1998, 1999, 2000, 2001, 
                    2002, 2003, 2004, 2005, 2006, 
                    2007, 2008, 2009, 2010, 2011, 
                    2012, 2013, 2014, 2015, 2016]
    
    for weather_year in weather_years:
        
        # Change weather year before preprocessing
        config.set('PreProcessing', 'balmorel_weather_year', str(weather_year))
        store_config(config)
        
        # Preprocess data
        os.system('pixi run preprocessing -F --rerun-incomplete')
                
        # Get parameters
        export_to_generative_model(scenario_folder, weather_year, 'Pre-Processing/Output/genmodel_data_WY%d.csv')


if __name__ == '__main__':
    main()
