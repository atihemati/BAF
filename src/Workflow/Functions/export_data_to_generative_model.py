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
                               ]
                               ):
    
    model = Balmorel('Balmorel')
    model.load_incfiles(scenario_folder)

    # Verify weather year
    with open(f'Balmorel/{scenario_folder}/data/DH_VAR_T.inc', 'r') as f:
        content = f.read()
        read_weather_year = int(content.split('Weather year ')[1].split('\n')[0])

    if read_weather_year != weather_year:
        raise ValueError(f"Read weather year was {read_weather_year}, but process expect {weather_year}!")

    input_data = {parameter : correct_format(symbol_to_df(model.input_data[scenario_folder], parameter), weather_year) for parameter in parameters}
    
    return input_data

def correct_format(df: pd.DataFrame, weather_year: int):
    
    # Extract unique dimensions
    unique_sets = [col for col in df.columns if not(col in ['SSS', 'S', 'TTT', 'T'])]
    
    # Add sets
    df['WY'] = weather_year
    df['Parameter'] = df[unique_sets].astype(str).agg("|".join, axis=1)
    
    sum_before = df.Value.sum()
    df = df.pivot_table(index=['WY', 'SSS', 'TTT'], columns='Parameter', values='Value', aggfunc='sum')
    sum_after = df.sum()

    if sum_before != sum_after:
        raise ValueError('Aggregation didnt work!')

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
    
    full_dataset = {}
    for weather_year in [1982]:
        
        # Change weather year before preprocessing
        config.set('PreProcessing', 'balmorel_weather_year', str(weather_year))
        store_config(config)
        
        # Preprocess data
        # os.system('pixi run preprocessing -F')
                
        # Get parameters
        full_dataset[weather_year] = export_to_generative_model(scenario_folder, weather_year)


if __name__ == '__main__':
    main()
