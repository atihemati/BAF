"""
Created on 09-06-2023

@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###          Script Settings        ###
### ------------------------------- ###

import click
import pandas as pd
from pandas.errors import EmptyDataError
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('.')
from Functions.Formatting import newplot, nested_dict_to_df
from Functions.GeneralHelperFunctions import load_OSMOSE_data, data_context
from pybalmorel import IncFile
import pickle
from scipy.spatial import distance_matrix
import configparser

# Load geojson's
try:
    import geopandas as gpd
    p = 'Pre-Processing/Data/Balmorelgeojson'
    
    if not('balmmap' in locals()):
        balmmap = gpd.read_file('Pre-Processing/Data/231206 AntBalmMap.geojson')
    plot_maps = True
        
except ModuleNotFoundError:
    plot_maps = False
    print('Geopandas not installed. Will not be able to plot maps')
    balmmap = pd.DataFrame({'id' : 'a'},index=[0])

@click.group()
@click.pass_context
def CLI(ctx):
    """The command line interface for pre-processing stuff"""

    ### 0.0 Load configuration file
    Config = configparser.ConfigParser()
    Config.read('Config.ini')
    UseAntaresData = Config.getboolean('PeriProcessing', 'UseAntaresData') 
    balmorel_weather_year = Config.getint('PreProcessing', 'balmorel_weather_year')
    
    style = 'report'

    if style == 'report':
        plt.style.use('default')
        fc = 'white'
    elif style == 'ppt':
        plt.style.use('dark_background')
        fc = 'none'

    ### 0.1 Assumptions
    balm_flh_type = 'FLH'   # Calculate FLH from XXX_VAR_T.inc or XXXFLH.inc file? Select 'VAR_T' or 'FLH'
    normalise = True        # Normalise when calculating FLH from BALM timeseries? 
    stoch_years = 35        # Amount of stochastic years
    elec_loss = 0.95        # See Murcia, Juan Pablo, Matti Juhani Koivisto, Graziela Luzia, Bjarke T. Olsen, Andrea N. Hahmann, Poul Ejnar Sørensen, and Magnus Als. “Validation of European-Scale Simulated Wind Speed and Wind Generation Time Series.” Applied Energy 305 (January 1, 2022): 117794. https://doi.org/10.1016/j.apenergy.2021.117794.
    model_year = 2050       # The model year (analysing just one year for creating weights of electricity demand for different spatial resolutions)

    # Set geographical scope
    ctx.ensure_object(dict)
    ctx.obj['geographical_scope'] = Config.get('PreProcessing', 'geographical_scope').replace(' ', '').split(',')
    
    # Set Balmorel weather year choice
    ctx.obj['balmorel_weather_year'] = balmorel_weather_year
    
    # Detect which command has been passed
    command = ctx.invoked_subcommand
    
    ## Define Balmorel time index
    if command in ['generate-antares-vre', 'generate-balmorel-hydro', 'generate-balmorel-timeseries', 'generate-balmorel-heat-series']:
        # Create S and T timeseries
        ctx.obj['S'] = ['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)]
        ctx.obj['T'] = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)]
        ctx.obj['ST'] = [S + ' . ' + T for S in ctx.obj['S'] for T in ctx.obj['T']]
        
    ## Set paths for data
    if command in ['generate-antares-vre', 'generate-balmorel-timeseries', 'generate-balmorel-heat-series', 'generate-balmorel-annual-heat-adjustment']:
        data_context()

    ## Set weather years
    if command in ['generate-balmorel-timeseries', 'generate-balmorel-heat-series']:
        ctx.obj['weather_years'] = [balmorel_weather_year]
    elif command in ['generate-antares-vre', 'generate-balmorel-annual-heat-adjustment']:
        ctx.obj['weather_years'] = [1982, 1983, 1984, 1985, 1986, 
                                    1987, 1988, 1989, 1990, 1991, 
                                    1992, 1993, 1994, 1995, 1996, 
                                    1997, 1998, 1999, 2000, 2001, 
                                    2002, 2003, 2004, 2005, 2006, 
                                    2007, 2008, 2009, 2010, 2011, 
                                    2012, 2013, 2014, 2015, 2016]


### ------------------------------- ###
###            Utilities            ###
### ------------------------------- ###
def read_incfile_presuf(name: str, 
                      weather_year: int,
                      incfile_prefix_path: str = 'Pre-Processing/Data/IncFile PreSuffixes',
                      presuf: str = 'pre'):
    """Reads an .inc file that should just contain the prefix of the incfile

    Args:
        name (str): Name of the incfile
        incfile_prefix_path (str): Path to where the related prefix .inc file is
        weather_year (int): The weather year, which is relevant for the description of some .inc files.

    Returns:
        _type_: _description_
    """
    # Add information about weather year if weather dependent parameter
    if ('WND' in name) | ('SOLE' in name) | ('WTR' in name) | ('DH' in name and not ('INDUSTRY' in name or 'HYDROGEN' in name)):
        string = "* Weather year %d\n"%(weather_year) + ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())
    else:
        string = ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())

    # Get prefix or suffix
    if presuf == 'pre':
        string = string.split('*PRESUFSPLIT*')[0]
    else:
        try:
            string = string.split('*PRESUFSPLIT*')[1]
        except IndexError:
            raise IndexError("%s didn't include a *PRESUFSPLIT* line, making it impossible to figure out what the suffix is!"%name)    
    
    return string

def create_incfiles(names: list, 
                    weather_year: int,
                    bodies: dict = None, 
                    suffixes: dict = None):
    """A convenient way to create many incfiles and fill in predefined prefixes, bodies and suffixes

    Args:
        names (list): Names of the incfiles created (without .inc)
        weather_year (int): The weather year for weather dependent parameters.
        bodies (dict, optional): bodies of the incfiles. Defaults to None.
        suffixes (dict, optional): suffixes of the incfiles. Defaults to None.
        incfile_prefix_path (str, optional): path to the prefixes of the incfiles. Defaults to 'Pre-Processing/Data/IncFile Prefixes'.
        
    Returns:
        incfiles (dict): Dictionary of incfiles
    """
    
    incfiles = {
        incfile : IncFile(name=incfile, prefix=read_incfile_presuf(incfile, weather_year), suffix=read_incfile_presuf(incfile, weather_year, presuf='suf')) \
        for incfile in names
    }
    
    if bodies != None:
        for incfile in bodies.keys():
            incfiles[incfile].body = bodies[incfile]
            
    return incfiles

def append_neighbouring_years(filename: str, year: int, values: str,
                              index: str = 'timestamp', columns: str = 'country'):
    """Include year before and after for year-separated .csv files"""
    df = (
        pd.read_csv(filename)
        .pivot_table(index=index, columns=columns, values=values, aggfunc='sum',
                     fill_value=0)   
    )
    
    for neighbour_year in [year-1, year+1]:
        neighbour_filename = filename.replace(str(year), str(neighbour_year))
        # Check if file for neighbouring year exists
        try:
            temp = (
            pd.read_csv(neighbour_filename)
            .pivot_table(index=index, columns=columns, 
                         values=values, aggfunc='sum',
                        fill_value=0)   
            )
            
            # Remove most of the year before and after, except one week
            if neighbour_year == year - 1:
                temp = temp.iloc[-168:]
                df = pd.concat((temp, df))
            elif neighbour_year == year + 1:
                temp = temp.iloc[:168]
                df = pd.concat((df, temp))
            else:
                raise ValueError('Something is wrong in finding neighbouring years!')

        
        except FileNotFoundError:
            print("Couldn't find year %d of file %s"%(neighbour_year, neighbour_filename))

    return df
    
@click.pass_context
def convert_to_52weeks(ctx, year: int, filename: str, values: str):
    """Convert timeseries to start on first hour of first monday and be 52 weeks long"""
    
    filename = filename%year
    df = pd.read_csv(filename).pivot_table(index='time_id', columns='country', values=values)
    
    # Just take first 52 weeks
    df = df.iloc[:8736]
    
    # Apply S and T index
    df.index = ctx.obj['ST']
    
    return df

def list_of_str(value):
    return value.replace(' ', '').split(',')


#%% ------------------------------- ###
###        Hardcoded Mappings       ###
### ------------------------------- ###

@CLI.command()
@click.pass_context
def generate_mappings(ctx):
    """Generates spatial and technological mappings between Balmorel and Antares"""
    
    ### 1.1 Regions for A2B2A Mapping
    geographical_scope = ctx.obj['geographical_scope']
    
    # Regions for VRE Mapping (currently uniform mapping)
    B2A_regi = {region : [region] for region in geographical_scope}

    A2B_regi = {region : [region] for region in geographical_scope}


    ## Save defined regions
    with open('Pre-Processing/Output/B2A_regi.pkl', 'wb') as f:
        pickle.dump(B2A_regi, f)
    with open('Pre-Processing/Output/A2B_regi.pkl', 'wb') as f:
        pickle.dump(A2B_regi, f)
        
        
    ### 1.2 Other useful dictionaries
    # Fuel
    B2A_fuel = {'SUN' : 'solar',
                'WIND' : 'wind',
                'BIOGAS' : 'biogas'}
    A2B_fuel = {B2A_fuel[k] : k for k in B2A_fuel.keys()}

    # Balmorel Technologies (rough combustion CO2 estimates hardcoded for now)
    kgGJ2tonMWh = 3.6 / 1e3 # Conversion from kg/GJ to ton/MWh
    BalmTechs = {'CHP-BACK-PRESSURE' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                        'WOODCHIPS' : {'CO2' : 0},  
                                        'WOODPELLETS' : {'CO2' : 0},  
                                        'STRAW' : {'CO2' : 0}, 
                                        'BIOGAS' : {'CO2' : 0}, 
                                        'MUNIWASTE' : {'CO2' : 0},
                                        'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                        'FUELOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                        'LIGHTOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                        'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh}},
                'CHP-EXTRACTION' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                    'WOODCHIPS' : {'CO2' : 0},  
                                    'WOODPELLETS' : {'CO2' : 0}, 
                                    'BIOGAS' : {'CO2' : 0}, 
                                    'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                    'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh},
                                    'MUNIWASTE' : {'CO2' : 0},
                                    'WOODWASTE' : {'CO2' : 0},
                                    'WOOD' : {'CO2' : 0},
                                    'PEAT' : {'CO2' : 0},
                                    'STRAW' : {'CO2' : 0}},
                'CONDENSING' : {'BIOGAS' : {'CO2' : 0}, 
                                'COAL' : {'CO2' : 94.6*kgGJ2tonMWh},
                                'FUELOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                'LIGHTOIL' : {'CO2' : 74*kgGJ2tonMWh},
                                'LIGNITE' : {'CO2' : 111.1*kgGJ2tonMWh},
                                'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh}, 
                                'WOODCHIPS' : {'CO2' : 0}, 
                                'MUNIWASTE' : {'CO2' : 0},
                                'NUCLEAR' : {'CO2' : 0},
                                'STRAW' : {'CO2' : 0},
                                'WOOD' : {'CO2' : 0},
                                'PEAT' : {'CO2' : 0},
                                'STRAW' : {'CO2' : 0}},
                'CONDENSING-CCS' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh * 0.1}}, 
                'CHP-BACK-PRESSURE-CCS' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh * 0.1}}, 
                'CHP-EXTRACTION-CCS' : {'NATGAS' : {'CO2' : 56.1 * kgGJ2tonMWh * 0.1}}, 
                'FUELCELL' : {'HYDROGEN' : {'CO2' : 0}}}
    {'BIOGAS' : 2.13,
                'WOODCHIPS' : 3.34,
                'MUNIWASTE' : 4,
                'WOOD' : 3.34,
                'WOODWASTE' : 3.34,
                'PEAT' : 3.34,
                'STRAW' : 3.34}

    with open('Pre-Processing/Output/BalmTechs.pkl', 'wb') as f:
        pickle.dump(BalmTechs, f)

#%% ------------------------------- ###
###            Timeseries           ###
### ------------------------------- ###


@CLI.command()
@click.pass_context
@load_OSMOSE_data(files=['offshore_wind', 'onshore_wind', 'solar_pv'])
def generate_antares_vre(ctx, data: str, stoch_year_data: dict, antares_input_paths = {
        'offshore_wind' : 'Antares/input/renewables/series/%s/offshore/series.txt',
        'onshore_wind' : 'Antares/input/renewables/series/%s/onshore/series.txt',
        'solar_pv' : 'Antares/input/renewables/series/%s/photovoltaics/series.txt',
    }):
    """Generate production factor timeseries for Antares VRE"""
    
    # Create matrix of input that Antares expects
    for region in ctx.obj['geographical_scope']:
        
        try:   
            data_to_antares_input = pd.DataFrame({year : stoch_year_data[year][region] for year in stoch_year_data.keys()})
            data_to_antares_input.to_csv(antares_input_paths[data]%(region.lower()),
                        index=False, header=False, sep='\t')
        except KeyError:
            print('No %s for %s'%(data, region))
    
    
@CLI.command()
@click.pass_context   
@load_OSMOSE_data(files=['offshore_wind', 'onshore_wind', 'solar_pv', 'load']) 
def generate_balmorel_timeseries(ctx, data: str, stoch_year_data: dict):
    """Generate Balmorel timeseries input data for VRE (except hydro) and exogenous electricity demand"""
    
    # Format data to Balmorel input
    balmorel_names = {
        'offshore_wind' : {'incfile' : 'WND_VAR_T_OFF',
                           'area_suffix' : '_OFF'}, 
        'onshore_wind'  : {'incfile' : 'WND_VAR_T',
                           'area_suffix' : '_A'},
        'solar_pv'      : {'incfile' : 'SOLE_VAR_T',
                           'area_suffix' : '_A'},
        'load'          : {'incfile' : 'DE_VAR_T',
                           'area_suffix' : ''}
    }
    
    # Format data
    ## Only the chosen weather year for Balmorel is loaded here 
    year_data_column = list(stoch_year_data.keys())[0] 
    df = stoch_year_data[year_data_column]
    FLH = df.sum()
    df = df.loc[:8736, :] # Just leave out the last (two last) day(s)
    df.index = ctx.obj['ST']  
    df.columns = pd.Series(df.columns) + balmorel_names[data].get('area_suffix')
    df.columns.name = ''
            
    # Create .inc files for Balmorel
    f = IncFile(name=balmorel_names[data].get('incfile'),
                prefix=read_incfile_presuf(balmorel_names[data].get('incfile'), ctx.obj['weather_years'][0]),
                suffix=read_incfile_presuf(balmorel_names[data].get('incfile'), ctx.obj['weather_years'][0], presuf='suf'))
    f.body = df
    f.save()

    if data != 'load':
        FLH_name = balmorel_names[data].get('incfile').replace('_VAR_T', 'FLH')
        f = IncFile(name=FLH_name,
                    prefix=read_incfile_presuf(FLH_name, ctx.obj['weather_years'][0]),
                    suffix=read_incfile_presuf(FLH_name, ctx.obj['weather_years'][0], presuf='suf'),
                    body="\n".join(['%s%s %0.2f'%(col, balmorel_names[data]['area_suffix'], FLH[col]) for col in FLH.index]))
        f.save()


@CLI.command()
@click.pass_context
@load_OSMOSE_data(files=['heat'])
def generate_balmorel_heat_series(ctx, data: str, stoch_year_data: dict):
    # Format data
    weather_year = ctx.obj['balmorel_weather_year']
    df = stoch_year_data[weather_year]
    df = df.loc[:8736, :] # Just leave out the last (two last) day(s)
    df.index = ctx.obj['ST']  
    df.columns.name = ''

    # Individual heat 
    df.columns = pd.Series(df.columns) + '_IDVU-SPACEHEAT'
     
    # Make constant, 20% heat for hot water assumption in district heat
    df_dh = 0.2/8736 + df*0.8
    
    # District Heat
    df_dh.columns = pd.Series(df_dh.columns).str.replace('_IDVU-SPACEHEAT', '_A')

    incfiles = create_incfiles(names=['DH_VAR_T', 'INDIVUSERS_DH_VAR_T'],
                               weather_year=weather_year,
                               bodies={'DH_VAR_T' : df_dh,
                                       'INDIVUSERS_DH_VAR_T' : df})

    for incfile in incfiles.values():
        incfile.save()

@CLI.command()
@click.pass_context
@load_OSMOSE_data(files=['heat'])
def generate_balmorel_annual_heat_adjustment(ctx, data: str, stoch_year_data: dict):
    
    # Get factors on annual heat demand
    weather_year = ctx.obj['balmorel_weather_year']
    factors, normal_year = get_annual_heat_demand_factor(stoch_year_data, weather_year)
    
    # Add the factor commands to DH.inc and INDIVUSERS_DH.inc 
    DH_COMMANDS = f'* Weather year {weather_year} factors relative to normal weather year {normal_year}, based on sum of heat demand throughout regions\n'
    INDIVDH_COMMANDS = f'* Weather year {weather_year} factors relative to normal weather year {normal_year}, based on sum of heat demand throughout regions\n'
    
    for region in factors.index:
        # 80% of district heating demand is space heating
        DH_COMMANDS += f"DH(YYY, AAA, DHUSER)$RRRAAA('{region}', AAA) = 0.8*{factors[region]}*DH(YYY, AAA, DHUSER)$RRRAAA('{region}', AAA) + 0.2*DH(YYY, AAA, DHUSER)$RRRAAA('{region}', AAA);\n"
        
        # Choosing only space heat individual heating areas
        INDIVDH_COMMANDS += f"DH(YYY, '{region}_IDVU-SPACEHEAT', 'TERTIARY'   ) = {factors[region]}*DH(YYY, '{region}_IDVU-SPACEHEAT', 'TERTIARY'   );\n"
        INDIVDH_COMMANDS += f"DH(YYY, '{region}_IDVU-SPACEHEAT', 'RESIDENTIAL') = {factors[region]}*DH(YYY, '{region}_IDVU-SPACEHEAT', 'RESIDENTIAL');\n"

    replace_text_in_file('Balmorel/base/data/DH.inc', DH_COMMANDS)
    replace_text_in_file('Balmorel/base/data/INDIVUSERS_DH.inc', INDIVDH_COMMANDS)

def replace_text_in_file(file: str, text: str):
    """Will add or replace content of a text file within specific modification warning labels

    Args:
        file (str): The file to append or replace content within warning labels
        text (str): The text to put within warning labels
    """
    
    # Factor district heat ares by 0.8*factors
    with open(file, 'r') as f:
        content = f.read()
    
    ## Remove previous insertions
    before_label = '* ------ DO NOT MODIFY THIS LINE OR BELOW ------'
    after_label = '* ------ DO NOT MODIFY THIS LINE OR ABOVE ------'
    if before_label in content and after_label in content:
        content_cleaned = content.split(before_label)[0] + content.split(after_label)[1]
    else:
        content_cleaned = content
    
    ## Write new file with content appended to bottom
    with open(file, 'w') as f:
        f.write(content_cleaned)
        f.write('\n')
        f.write(before_label)
        f.write('\n')
        f.write(text)
        f.write('\n')
        f.write(after_label)
        f.write('\n')
        

def get_annual_heat_demand_factor(stoch_year_data: dict, weather_year: int):
    """Finds median annual heat demand year and returns the factor
    calculated relative to the median year, that should be 
    multiplied to annual heat demand per region

    Args:
        stoch_year_data (dict): The heat data
        weather_year (int): The chosen weather year

    Returns:
        factors (pd.Series): Factors on chosen weather year per region
        most_normal_year (int): The normal weather year
    """
    
    data = []
    for year in stoch_year_data.keys():
        data.append(
            stoch_year_data[year].sum().to_dict() | {'Weather Year' : year}
        )
        
    df = pd.DataFrame(data).pivot_table(index='Weather Year')
    idx_of_most_normal_year = df.sum(axis=1).median() == df.sum(axis=1)
    most_normal_year = df[idx_of_most_normal_year].index[0] # Just take the first one, if there were identical years

    factors = df.loc[weather_year] / df.loc[most_normal_year]
         
    return factors, most_normal_year

### ------------------------------- ###
###            Hydropower           ###
### ------------------------------- ###
@CLI.command()
@click.pass_context
def generate_balmorel_hydro(ctx, weather_year: int = 2000):
    """Generate Balmorel input data for hydropower"""

    # Read parameters
    ## E.g., 30 = 2012, note we start counting from 0 here, compared to in Antares UI!
    weather_year = weather_year - 1982

    ## Balmorel timestamps
    S = ctx.obj['S']
    T = ctx.obj['T']
    ST = ctx.obj['ST']

    # Prepare placeholders
    hydro_res = configparser.ConfigParser()
    hydro_res.read('Antares/input/hydro/hydro.ini')
    GKFX_hydro_split_string_start = '* ---- START OF HYDRO HARMONISATION - DONT REMOVE OR MODIFY THIS LINE ----'
    GKFX_hydro_split_string_end = '* ---- END OF HYDRO HARMONISATION - DONT REMOVE OR MODIFY THIS LINE ----'
    with open('Balmorel/base/data/GKFX.inc', 'r') as f:
        GKFX = f.read()
        GKFX_start = GKFX.split(GKFX_hydro_split_string_start)[0] + GKFX_hydro_split_string_start + '\n'
        GKFX_end = GKFX.split(GKFX_hydro_split_string_end)[1] + GKFX_hydro_split_string_end + '\n'
        
    hydro_AAA = '\n'
    # GNR_RES_WTR_NOPMP (100 % efficiency)
    # GNR_RES_WTR_PMP_MC-01 (100 % efficiency)
    hydro_GKFX = '\n'
    hydro_WTRRRFLH = ''
    hydro_WTRRSFLH = ''
    hydro_WTRRSVAR_S = pd.DataFrame(index=S)
    hydro_WTRRRVAR_T = pd.DataFrame(index=ST)
    hydro_HYRSMAXVOL_G = '\n'

    for area in ctx.obj['geographical_scope']:
        print('Generating hydro data for %s'%area)
        ### Reservoir power capacity and pumping eff in the area itself
        try:
            turb_cap = pd.read_table('Antares/input/hydro/common/capacity/maxpower_%s.txt'%area, header=None)[0].max() # MW
            pump_cap = pd.read_table('Antares/input/hydro/common/capacity/maxpower_%s.txt'%area, header=None)[2].max() # MW
            
            res_series = pd.read_table('Antares/input/hydro/series/%s/mod.txt'%area.lower(), header=None) # MW

            # Summing weekly flows for Balmorel
            hydro_WTRRSVAR_S['%s_A'%area] = res_series.rolling(window=7).sum()[6::7][weather_year].values
            
            # FLH
            hydro_WTRRSFLH += "%s_A\t\t%0.2f\n"%(area, ((res_series[weather_year]).sum())/(8760*turb_cap)*8760)
              
            # Make .inc file commands
            hydro_AAA += '%s_A\n'%area 
            if pump_cap == 0:
                G = 'GNR_RES_WTR_NOPMP'
                hydro_GKFX += "GKFX(YYY,'%s_A','%s') = %0.2f;\n"%(area, G, turb_cap)
            else:
                G = 'GNR_RES_WTR_PMP_MC-01'
                hydro_GKFX += "GKFX(YYY,'%s_A','%s') = %0.2f;\n"%(area, G, turb_cap)
            
            print('Found reservoir data..')
            
            # See if there's a reservoir capacity
            try:
                res_cap = hydro_res.getfloat('reservoir capacity', area)
                hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_A', '%s') = %0.2f;\n"%(area, G, res_cap/turb_cap)
                print('..with pump')
            except configparser.NoOptionError:
                hydro_HYRSMAXVOL_G += "HYRSMAXVOL_G('%s_A', '%s') = %0.2f;\n"%(area, G, 0)
            
        except EmptyDataError:
            # No reservoir storage in area
            pass

        ### Run of river in the area itself
        try:
            ror_series = pd.read_table('Antares/input/hydro/series/%s/ror.txt'%area.lower(), header=None) # MW
            ror_cap = ror_series.max().max()

            # Make .inc file commands        
            if not('%s_A'%area in hydro_AAA):
                hydro_AAA += '%s_A\n'%area 
                
            hydro_WTRRRVAR_T['%s_A'%area] = ror_series.loc[:8735, weather_year].values
            
            hydro_WTRRRFLH += "%s_A\t\t%0.2f\n"%(area, ((ror_series[weather_year]).sum())/(8760*ror_cap)*8760)

            
            hydro_GKFX += "GKFX(YYY, '%s_A', 'GNR_ROR_WTR') = %0.2f;\n"%(area, ror_cap)
            print('Found run-of-river data..')
        except EmptyDataError:
            pass

        ### Pumped Hydro Storage (No inflow) 
        for hydro_area in ['00_PSP_STO']:
            
            try:
                turb_cap = pd.read_table('Antares/input/links/%s/capacities/%s_direct.txt'%(hydro_area, area.lower()), header=None).max()[0]
                
                # .inc file commands
                # Note that this capacity is defined in MWh! Thus factored the unloading parameter GDSTOHUNLD of  9.4 
                hydro_GKFX += "GKFX(YYY, '%s_A', 'GNR_ES_WTR_PMP') = %0.2f;\n"%(area, turb_cap*9.4)
                print('Found pumped hydro data..')

            except FileNotFoundError:
                pass
            
            #'GNR_ES_WTR_PMP' changed to 75% efficiency in GKFX incfile prefix

    # Create 
    incfiles = create_incfiles(['WTRRSVAR_S', 'WTRRSFLH', 'WTRRRVAR_T', 'WTRRRFLH', 'HYRSMAXVOL_G'], weather_year+1982,
                               bodies={'WTRRSVAR_S' : hydro_WTRRSVAR_S.to_string(),
                                       'WTRRRVAR_T' : hydro_WTRRRVAR_T.to_string(),
                                       'WTRRRFLH' : hydro_WTRRRFLH,
                                       'WTRRSFLH' : hydro_WTRRSFLH,
                                       'HYRSMAXVOL_G' : hydro_HYRSMAXVOL_G})
    
    # Save
    for key in incfiles.keys():
        incfiles[key].save()
        
    with open('Balmorel/base/data/GKFX.inc', 'w') as f:
        f.write(GKFX_start)
        f.write(hydro_GKFX)
        f.write(GKFX_end)



@CLI.command()
@click.pass_context
def old_preprocessing(ctx):
    """The old processing scripts"""
    # Normalise Electricity and Hydrogen Demand Profiles

    ### NOTE: For new H2 regions in Antares it is assumed that, 
    #         TR = GR
    #          
    balmmap['El Dem'] = 0
    balmmap['H2 Dem'] = 0
    for area in list(A2B_regi.keys()) + list(A2B_regi_h2):
        # Load Antares demand
        try:
            ant_dem = pd.read_csv(r'C:\Users\mberos\ElementsOutside1D\BZModel\input' + '/load/series/load_%s.txt'%(area.lower()), sep='\t', header=None)
            normed = (ant_dem / ant_dem.sum().mean()) # Normalise wrt. annual energy
            (ant_dem / ant_dem.sum().mean()).to_csv('Antares' + '/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None, index=None)    
            # ax.plot(ant_dem[0] / ant_dem.sum().mean(), label=area)
            
            if 'z_h2' in area:
                balmmap.loc[balmmap.id == A2B_regi_h2[area][0], 'H2 Dem'] += ant_dem.sum().values[0] / 1e6
            else:
                balmmap.loc[balmmap.id == A2B_regi[area][0], 'El Dem'] += ant_dem.sum().values[0] / 1e6
        except (EmptyDataError, FileNotFoundError):
            print('No load data in %s'%area)

    ### Plot 
    if plot_maps:
        # El Demand (2040)
        fig, ax = newplot(fc=fc, figsize=(10, 10))
        balmmap.plot(column='El Dem', ax=ax, legend=True,
                    legend_kwds={'label' : 'Annual El Demand (TWh)',
                                'orientation' : 'horizontal'})
        ax.set_xlim([-11, 30])
        ax.set_ylim([35, 68])

        # H2 Demand (2040)
        fig, ax = newplot(fc=fc, figsize=(10, 10))
        balmmap.plot(column='H2 Dem', ax=ax, legend=True,
                    legend_kwds={'label' : 'Annual H2 Demand (TWh)',
                                'orientation' : 'horizontal'})
        ax.set_xlim([-11, 30])
        ax.set_ylim([35, 68])

    # Normalise Thermal Generation Profiles

    ### NOTE: For new H2 regions in Antares it is assumed that, 
    #         TR = GR
    #          
    # fig, ax = plt.subplots()
    for area in list(A2B_regi.keys()) + list(A2B_regi_h2):
        # Load Thermal generation profile
        try:
            l = os.listdir(r'C:\Users\mberos\ElementsOutside1D\BZModel\input' + '/thermal/series/%s'%area.lower())
            outageseries = 0
            for generator in l:
                gen = pd.read_csv('Antares/input/thermal/series/%s/%s/series.txt'%(area, generator), sep='\t', header=None)
                if gen[0].std() != 0:
                    (gen / gen.max()).to_csv('Antares/input/thermal/series/%s/%s.txt'%(area, 'outageseries_' + generator), sep='\t', header=None, index=None)            
                    # print('Outage modelled in %s for %s'%(area, generator))        
                    outageseries += 1
        except:
            print('No thermal generation in %s'%area)
        # try:
        #     ant_dem = pd.read_csv('Antares' + '/input/load/series/load_%s.txt'%(area), sep='\t', header=None)
        #     normed = (ant_dem / ant_dem.sum().mean()) # Normalise wrt. annual energy
        #     (ant_dem / ant_dem.sum().mean()).to_csv('Antares' + '/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None, index=None)    
        #     # ax.plot(ant_dem[0] / ant_dem.sum().mean(), label=area)
        # except EmptyDataError:
        #     print('No load data in %s'%area)
    # ax.legend()




    #%% ------------------------------- ###
    ###         4. A2B Weights          ###
    ### ------------------------------- ###

    ### 4.0 Weigths for assigning values from Balmorel to Antares
    ###     NOTE: It will use the current, absolute loads in Antares/input/load to distribute weights! 
    B2A_DH2_weights = {}
    B2A_DE_weights = {}

    ## Electricity
    for area in B2A_regi.keys():
        # If the spatial resolution is identical
        if len(B2A_regi[area]) == 1:
            B2A_DE_weights[area] = {B2A_regi[area][0] : 1}
            
        # If more Antares regions exist for that Balmorel region
        else:
            # Read data
            loads = []
            B2A_DE_weights[area] = {}
            for antarea in B2A_regi[area]:
                f = pd.read_csv('Antares/input/load/series/load_%s.txt'%antarea, sep='\t', header=None)
                loads.append(f.sum().mean()) # Mean annual load for all stochastic years
            
            # Assign load
            for n in range(len(B2A_regi[area])):
                B2A_DE_weights[area][B2A_regi[area][n]] = loads[n] / np.sum(loads)
                
    ## Hydrogen
    for area in B2A_regi_h2.keys():
        # If the spatial resolution is identical
        if len(B2A_regi_h2[area]) == 1:
            B2A_DH2_weights[area] = {B2A_regi_h2[area][0] : 1}
            
        # If more Antares regions exist for that Balmorel region
        else:
            # Read data
            loads = []
            B2A_DH2_weights[area] = {}
            for antarea in B2A_regi_h2[area]:
                f = pd.read_csv('Antares/input/load/series/load_%s.txt'%antarea, sep='\t', header=None)
                loads.append(f.sum().mean()) # Mean annual load for all stochastic years
            
            # Assign load
            for n in range(len(B2A_regi_h2[area])):
                B2A_DH2_weights[area][B2A_regi_h2[area][n]] = loads[n] / np.sum(loads)
                

    with open('Pre-Processing/Output/B2A_DE_weights.pkl', 'wb') as f:
        pickle.dump(B2A_DE_weights, f)
    with open('Pre-Processing/Output/B2A_DH2_weights.pkl', 'wb') as f:
        pickle.dump(B2A_DH2_weights, f)


    ### 4.1 Make Demand Ratios for Fictive Demand Approach
    f = pd.read_csv('Pre-Processing/Data/BalmDE.csv', sep=';')
    f['C'] = f.R.str[:2]

    A2B_DE_weights = {}

    for ant_area in A2B_regi.keys():
        
        A2B_DE_weights[ant_area] = {}
        total_dem = np.sum([f.loc[(f.R == balm_area) & (f.Y == year), 'Val'].sum() for balm_area in A2B_regi[ant_area]])

        for balm_area in A2B_regi[ant_area]:
            
            r_dem = f.loc[(f.R == balm_area) & (f.Y == year), 'Val'].sum()
            
            # Save share of electricity demand in dict
            A2B_DE_weights[ant_area][balm_area] = r_dem / total_dem
            
            # Print the share of demand in that region
            print(ant_area, balm_area, r_dem / total_dem*100, '%')


    ### 4.2 Save A2B DE weight dictionary
    with open('Pre-Processing/Output/A2B_DE_weights.pkl', 'wb') as f:
        pickle.dump(A2B_DE_weights, f) 


    ### 4.3 Fictive Demand from Antares to Balmorel
    f = pd.read_csv('Pre-Processing/Data/BalmDH2.csv', sep=';')
    f['C'] = f.R.str[:2]

    A2B_DH2_weights = {}

    

    ### 6.3 Balmorel timestamps
    S = ['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)]
    T = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)]
    balmtime_index = ['%s . %s'%(S0, T0) for S0 in S for T0 in T]
    WND_VAR_T = pd.DataFrame([], index=balmtime_index)
    SOLE_VAR_T = pd.DataFrame([], index=balmtime_index)
    DE_VAR_T = pd.DataFrame([], index='RESE . ' + pd.Series(balmtime_index))


    ### 6.4 Automatic definition of .inc file body content
    for BalmArea in list(B2A_regi.keys()):

        for area in B2A_regi[BalmArea]:    
            if area != 'ITCO':        
                # if len(A2B_regi[area]) > 1:
                #     # If Balmorel is higher resolved:
                #     area += '_%s'%BalmArea

                ## Sets
                # ANTBALM_CCCRRRAAA, RRRAAA and AAA
                incfiles['ANTBALM_CCCRRRAAA'].body += "%s_A\n"%area
                incfiles['ANTBALM_RRRAAA'].body += "RRRAAA('%s', '%s_A') = YES;\n"%(BalmArea, area)
                incfiles['ANTBALM_AAA'].body += "%s_A\n"%area
                
                
                ## Moving Renewable Capacity to New Areas
                if area not in ['FR15', 'GR15', 'ITCA', 'ITCN', 
                                'ITCS', 'ITN1', 'ITS1', 'ITSA', 'ITSI', 'UKNI']:
                    incfiles['ANTBALM_GKFX'].body1 += "NOTNEWAREA('%s_A') = NO;\n"%area
                    incfiles['ANTBALM_GKFX'].body2 += "GKFX(YYY,'%s_A',GGG)$(GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE OR GDATA(GGG,'GDTECHGROUP') EQ SOLARPV) = SUM(AAA, GKFX(YYY,AAA,GGG)$(RRRAAA('%s',AAA) AND NOTNEWAREA(AAA) AND (GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE OR GDATA(GGG,'GDTECHGROUP') EQ SOLARPV)));\n"%(area, BalmArea)
                    incfiles['ANTBALM_GKFX'].body3 += "GKFX(YYY,AAA,GGG)$(RRRAAA('%s',AAA) AND NOTNEWAREA(AAA) AND (GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_OFFSHORE OR GDATA(GGG,'GDTECHGROUP') EQ WINDTURBINE_ONSHORE OR GDATA(GGG,'GDTECHGROUP') EQ SOLARPV)) = 0;\n"%BalmArea
                
                ## VAR T and FLH
                try:
                    f = pd.read_csv('Antares/input/renewables/series/%s/%s/series.txt'%(area.replace('_%s'%BalmArea, '').lower(), area.replace('_%s'%BalmArea, '').lower() + '_wind_0'), sep='\t', header=None)            
                    WND_VAR_T[area + '_A'] = f.loc[:8735, balm_year].values / elec_loss # Electricity loss accounted for in Balmorel
                    incfiles['ANTBALM_WNDFLH'].body += "%s_A \t\t\t %0.2f\n"%(area, WND_VAR_T[area + '_A'].sum())
                except EmptyDataError:
                    print('No solar data in %s'%(area))    
                    
                try:
                    f = pd.read_csv('Antares/input/renewables/series/%s/%s/series.txt'%(area.replace('_%s'%BalmArea, '').lower(), area.replace('_%s'%BalmArea, '').lower() + '_solar_0'), sep='\t', header=None)            
                    SOLE_VAR_T[area + '_A'] = f.loc[:8735, balm_year].values / elec_loss # Electricity loss accounted for in Balmorel
                    incfiles['ANTBALM_SOLEFLH'].body += "%s_A \t\t\t %0.2f\n"%(area, SOLE_VAR_T[area + '_A'].sum())
                except EmptyDataError:
                    print('No wind data in %s'%(area))    
                    
                ## DE_VAR_T
                f = pd.read_csv('Antares/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None)            
                DE_VAR_T[BalmArea] = f.loc[:8735, balm_year].values        
                    
                ## AGKN, Investment options (Look at prefix for ANTBALM_AGKN, other investment options have been taken care of)
                for year0 in ['2020', '2030', '2040', '2050']:
                    incfiles['ANTBALM_AGKN'].body += "AGKN('%s_A','GNR_WT_ONS-%s') = YES;\n"%(area, year0)
                    incfiles['ANTBALM_AGKN'].body += "AGKN('%s_A','GNR_WT_OFF-%s') = YES;\n"%(area, year0)
                    
                    if area != 'NON1_A':
                        incfiles['ANTBALM_AGKN'].body += "AGKN('%s_A','GNR_PV-%s') = YES;\n"%(area, year0) 

    ### 6.5 The finishing lines and saving .inc files in Balmorel/base/data
    incfiles['ANTBALM_CCCRRRAAA'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_CCCRRR'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_RRR'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_AAA'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_WNDFLH'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_SOLEFLH'].suffix = "/;\n$label USEANTARESDATAEND"
    incfiles['ANTBALM_WND_VAR_T'].body = WND_VAR_T.to_string()
    incfiles['ANTBALM_WND_VAR_T'].suffix = '\n;\nWND_VAR_T(IA,SSS,TTT)$WND_VAR_T2(SSS,TTT,IA) = WND_VAR_T2(SSS,TTT,IA);\nWND_VAR_T2(SSS,TTT,AAA)=0;\n$label USEANTARESDATAEND'
    incfiles['ANTBALM_SOLE_VAR_T'].body = SOLE_VAR_T.to_string()
    incfiles['ANTBALM_SOLE_VAR_T'].suffix = '\n;\nSOLE_VAR_T(IA,SSS,TTT)$SOLE_VAR_T2(SSS,TTT,IA) = SOLE_VAR_T2(SSS,TTT,IA);\nSOLE_VAR_T2(SSS,TTT,AAA)=0;\n$label USEANTARESDATAEND'
    incfiles['ANTBALM_AGKN'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_GKFX'].body = incfiles['ANTBALM_GKFX'].body1 + incfiles['ANTBALM_GKFX'].body2 + incfiles['ANTBALM_GKFX'].body3
    incfiles['ANTBALM_GKFX'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_RRRAAA'].suffix = "$label USEANTARESDATAEND"
    incfiles['ANTBALM_DE_VAR_T'].body = DE_VAR_T.to_string()
    incfiles['ANTBALM_DE_VAR_T'].suffix = """\n;\nDE_VAR_T(RRR,'RESE',SSS,TTT) =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    DE_VAR_T(RRR,'OTHER',SSS,TTT) =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    DE_VAR_T(RRR,'DATACENTER',SSS,TTT)$SUM(YYY,DE(YYY,RRR,'DATACENTER'))  =  ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    DE_VAR_T(RRR,'PII',SSS,TTT) = ANTBALM_DE_VAR_T('RESE',SSS,TTT,RRR);
    ANTBALM_DE_VAR_T(DEUSER,SSS,TTT,RRR)=0;\n$label USEANTARESDATAEND"""

    ## Save
    for key in incfiles.keys():
        incfiles[key].save()

    #%% ------------------------------- ###
    ###           Transmission          ###
    ### ------------------------------- ###

    # This analysis requires geopandas
    if plot_maps:
        
        # From DEA 2021, 111 'Main electricity distribution grid' (no data for transmission? these costs are for 50/60 kV)
        XE_cost = 3100 # €/MW/km high bound
        
        # From Kountouris, Ioannis. “A Unified European Hydrogen Infrastructure Planning to Support the Rapid Scale-up of Hydrogen Production,” August 1, 2023. https://doi.org/10.21203/rs.3.rs-3185467/v1.
        TransInvCost = {2030 : {'H2' : {'Onshore' : {'New' : 536.17,
                                                    'Rep' : 150},
                                        'Offshore': {'New' : 902.13,
                                                    'Rep' : 175}},
                                'El' : {'Onshore' : {'New' : XE_cost},
                                        'Offshore': {'New' : XE_cost}}}, 
                        2040 : {'H2' : {'Onshore' : {'New' : 263.08,
                                                    'Rep' : 86.15},
                                        'Offshore': {'New' : 450.77,
                                                    'Rep' : 120}},
                                'El': {'Onshore' : {'New' : XE_cost},
                                    'Offshore': {'New' : XE_cost}}}}
        TransInvCost = nested_dict_to_df(TransInvCost) # €/MW/km

        # Plot Settings
        scale = 0.75
        fig, ax = plt.subplots(dpi=75/scale, figsize=(20*scale,15*scale))
        balmmap.plot(ax=ax, facecolor=[.85, .85, .85])

        ### 7.1 Electricity Grid
        el_grid = gpd.read_file('Pre-Processing/Data/Infrastructure/EUPowerGrid.geojson')
        
        # Fix unknown voltages
        el_grid.loc[el_grid['voltage'] == '', 'voltage'] = 'unknown'
        # el_grid.voltage.astype('category')
        
        el_grid.plot(column='voltage', legend=True, ax=ax, linewidth=.5)
        # p1 = ax.get_legend().get_patches()
        
        ### 7.2 Gas Grid
        gas_grid = gpd.read_file('Pre-Processing/Data/Infrastructure/IGGIN_PipeSegments.geojson')
        gas_grid.plot(ax=ax, linewidth=.5, color='orange', label='Gas Grid')


        ax.set_xlim([-11, 35])
        ax.set_ylim([35, 68])
        # ax.legend(('132', '220', '300', '380', '500', '750', 'Unknown'))


        ### 7.3 Calculate Distances    
        balmmap_meters = balmmap.to_crs(4328)

        D = pd.DataFrame(
            distance_matrix(balmmap_meters.geometry.apply(lambda polygon: (polygon.centroid.x, polygon.centroid.y)).tolist(),
                            balmmap_meters.geometry.apply(lambda polygon: (polygon.centroid.x, polygon.centroid.y)).tolist()),
            index=balmmap_meters.id,
            columns=balmmap_meters.id
        ) / 1000 # in km

        # Links
        links = pd.read_csv('Pre-Processing/Data/Links.csv', sep=';')


        # Get distance matrix
        D = distance_matrix(x=balmmap.geometry.centroid.x,
                            y=balmmap.geometry.centroid.y)

### Main
if __name__ == '__main__':
    CLI()