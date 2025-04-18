
"""
General Helper Functions

Created on 18/08/2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
from typing import Union
import time
import sys
import configparser

#%% ------------------------------- ###
###          1. Logging etc.        ###
### ------------------------------- ###

### 1.0 Logging error
def ErrorLog(succes, N_errors, error_log, error_message):
    """
    If a script had an error, log it

    Args:
        succes (CompletedProcess): The subprocess status
        N_errors (Integer): Amount of errors so far
        error_log (String): The error log to append
        error_message (String): The error message to append, if there's an error

    Returns:
        Nerrors: The amount of errors after subprocess
        error_log: The error log after subprocess
    """
    if succes.returncode != 0:
        N_errors += 1
        error_log = '\n'.join([error_log, error_message])
         
    return N_errors, error_log

### 1.1 Log Process Time
def log_process_time(file_path, iteration, process_name, delta_time):
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write('%d,%s,%d\n'%(iteration, process_name, delta_time))
    else:
        with open(file_path, 'w') as file:
            file.write('Iteration,Process,Time\n')
            file.write('%d,%s,%d\n'%(iteration, process_name, delta_time))

#%% ------------------------------- ###
###           2. Dataframes         ###
### ------------------------------- ###

### 1.1 GDX to DataFrames
def symbol_to_df(db, symbol, cols='None'):
    """
    Loads a symbol from a GDX database into a pandas dataframe

    Args:
        db (GamsDatabase): The loaded gdx file
        symbol (string): The wanted symbol in the gdx file
        cols (list): The columns
    """   
    df = dict( (tuple(rec.keys), rec.value) for rec in db[symbol] )
    df = pd.DataFrame(df, index=['Value']).T.reset_index() # Convert to dataframe
    if cols != 'None':
        try:
            df.columns = cols
        except:
            pass
    return df 

### 1.2 Create a filter for either all values or only the highest and lowest value in a column
def filter_low_max(df, col='none', plot_all=True):
    """_summary_

    Args:
        df (_type_): _description_
        col (_type_): _description_
        filter_type (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if plot_all:
        print('All values included')
        idx = df.iloc[:, 0] == df.iloc[:, 0]
    else:
        if col != 'none':
            idx = (df[col] == df[col].max()) | (df[col] == df[col].min())
        else:
            print('Wrong column specified where max and min values should be filtered!')
    return idx

def store_capcred(CC, i, year, BalmArea, tech, tech_cap, val):
    """Takes the capacity credit dataframe and stores a new capacity credit for iteration i 

    Args:
        CC (dataframe): Dataframe comprising all capacity credits
        i (int): The current iteration
        BalmArea (str): The Balmorel Region
        tech (str): The column (Technology or receiving region)
        tech_cap (float): The capacity of the technology or link
        val (float): Capacity credit (float between 0 and 1)

    Returns:
        _type_: _description_
    """
    # Save capacity credit
    if tech_cap > 1e-6:
        CC.loc[(i, year, BalmArea), tech] = val
    # If there is no capacity and it is the first iteration, select a capcred of 1
    elif (tech_cap < 1e-6) & (i == 0):
        CC.loc[(i, year, BalmArea), tech] = 1
    # Otherwise - capacity credit from last iteration will be used
    else:
        # print('using last iteration CC')
        CC.loc[(i, year, BalmArea), tech] = CC.loc[(i-1, year, BalmArea), tech]

    return CC

#%% ------------------------------- ###
###            3. Analysis          ###
### ------------------------------- ###

### 3.1 LDC Curve and Plot Function
def doLDC(array, n_bins, plot=False, fig = None, ax = None):
    """Make load duration curve from timeseries

    Args:
        array (array): A timeseries of load, wind-, solar profiles or other.
        n-bins (int): Amount of bins in histogram

    Returns:
        duration (array): ordered hours
        curve (array): frequency
    """
    # Extract profile
    data = np.histogram(array, bins=n_bins)
    duration = data[0][::-1]
    curve = data[1][:-1][::-1]
    
    if plot:
        if (fig == None) | (ax == None): 
            fig, ax = plt.subplots()
            ax.plot(np.cumsum(duration), curve)
        else:
            ax.plot(np.cumsum(duration), curve)
    
        return duration, curve, fig, ax
    else:
        return duration, curve

#%% ------------------------------- ###
###        4. Antares Input         ###
### ------------------------------- ###

def set_cluster_attribute(name: str, attribute: str, value: any, 
                          node: str, cluster_type: str = 'thermal'):
    """Set the attribute of a 

    Args:
        name (str): The name of the cluster element, i.e. the clusterfile section
        attribute (str): The attribute of the cluster element to set, i.e. the clusterfile option
        value (any): The value to set
        node (str): The node containing the cluster
        cluster_type (str, optional): The type of cluster, e.g. 'thermal', 'renewables'... Defaults to 'thermal'.
    """
    discharge_config = configparser.ConfigParser()
    discharge_config.read('Antares/input/%s/clusters/%s/list.ini'%(cluster_type, node.lower()))
    discharge_config.set(name, attribute, str(value))
    with open('Antares/input/%s/clusters/%s/list.ini'%(cluster_type, node.lower()), 'w') as f:
        discharge_config.write(f)
    discharge_config.clear()

def create_transmission_input(wk_dir, ant_study, area_from, area_to, trans_cap, hurdle_costs):
    try: 
        f = open(wk_dir+ant_study + '/input/links/%s/%s_parameters.txt'%(area_from, area_to))
        p = wk_dir+ant_study + '/input/links/%s/%s_parameters.txt'%(area_from, area_to)
        pcap_dir = wk_dir+ant_study + '/input/links/%s/capacities/%s_direct.txt'%(area_from, area_to)
        pcap_ind = wk_dir+ant_study + '/input/links/%s/capacities/%s_indirect.txt'%(area_from, area_to)
    except FileNotFoundError:
        f = open(wk_dir+ant_study + '/input/links/%s/%s_parameters.txt'%(area_to, area_from)) 
        p = wk_dir+ant_study + '/input/links/%s/%s_parameters.txt'%(area_to, area_from)
        pcap_dir = wk_dir+ant_study + '/input/links/%s/capacities/%s_direct.txt'%(area_to, area_from)
        pcap_ind = wk_dir+ant_study + '/input/links/%s/capacities/%s_indirect.txt'%(area_to, area_from)
    
    # Write Parameters
    with open(p, 'w') as f:
        for k in range(8760):
            f.write('%0.2f\t%0.2f\t0\t0\t0\t0\n'%(hurdle_costs, hurdle_costs))
    
    # Write Capacities
    if type(trans_cap) != list:
        with open(pcap_dir, 'w') as f:
            for k in range(8760):
                f.write(str(int(trans_cap)) + '\n')
        with open(pcap_ind, 'w') as f:
            for k in range(8760):
                f.write(str(int(trans_cap)) + '\n')  
    else:
        with open(pcap_dir, 'w') as f:
            for k in range(8760):
                f.write(str(int(trans_cap[0])) + '\n')
        with open(pcap_ind, 'w') as f:
            for k in range(8760):
                f.write(str(int(trans_cap[1])) + '\n')  
        

def get_marginal_costs(year, cap, idx_cap, fuel, GDATA, FPRICE, FDATA, EMI_POL):
    """Gets average marginal cost of generators in cap[idx_cap], provided VOM, fuel and emission policy data 

    Args:
        year (_type_): _description_
        cap (_type_): _description_
        idx_cap (_type_): _description_
        totalcap (_type_): _description_
        GDATA (_type_): _description_
        FPRICE (_type_): _description_
        FDATA (_type_): _description_
        EMI_POL (_type_): _description_
    """
    ## Weighting capacities on highest data level (G)
    totalcap = cap.loc[idx_cap, 'Value'].sum() * 1e3 # MW
        
    mc_cost_temp = 0
    for G in cap[idx_cap].G.unique():
        Gcap = cap.loc[idx_cap & (cap.G == G), 'Value'].sum() * 1e3 # MW
        
        # VOM defined on output
        try:
            mc_cost_temp += GDATA.loc[(G, 'GDOMVCOST0'), 'Value'] * Gcap / totalcap
            # print('Added VOMO cost: ', mc_cost_temp)
        except:
            # print('No VOM-Out defined cost')
            pass
        
        # VOM defined on input
        try:
            # print('VOMI Cost: ', GDATA.loc[G, 'GDOMVCOSTIN'])
            mc_cost_temp += GDATA.loc[(G, 'GDOMVCOSTIN'), 'Value'] / GDATA.loc[(G, 'GDFE'), 'Value'] * Gcap / totalcap
            # print('Added VOMI cost: ', mc_cost_temp)
        except:
            # print('No VOM-In defined cost')
            pass         
        
        # Fuel cost
        try:
            mc_cost_temp += FPRICE.loc[(year, 'DK', fuel), 'Value'] / GDATA.loc[(G, 'GDFE'), 'Value'] * Gcap / totalcap # Same prices everywhere as in DK
            # print('Added fuel cost: ', mc_cost_temp)
        except:
            pass
        
        # Carbon cost
        try:
            country = cap[idx_cap].C.values[0]
            fuelemi = FDATA.loc[(fuel, 'FDCO2'), 'Value'] * 3.6 / 1000 # in t/MWh
            tax = EMI_POL.loc[(year, country, 'ALL_SECTORS', 'TAX_CO2'), 'Value'] # €/tCO2
            mc_cost_temp += tax * fuelemi / GDATA.loc[(G, 'GDFE'), 'Value'] * Gcap / totalcap
            # print('Added CO2 cost: ', mc_cost_temp)
        except:
            pass
        
    return mc_cost_temp


def get_efficiency(cap: pd.DataFrame, idx_cap: pd.Index, GDATA: pd.DataFrame):
    """Gets average marginal cost of generators in cap[idx_cap], provided VOM, fuel and emission policy data 

    Args:
        cap (_type_): _description_
        idx_cap (_type_): _description_
        GDATA (_type_): _description_
    """
    ## Weighting capacities on highest data level (G)
    totalcap = cap.loc[idx_cap, 'Value'].sum() * 1e3 # MW
    
    eff_temp = 0
    for G in cap[idx_cap].G.unique():
        Gcap = cap.loc[idx_cap & (cap.G == G), 'Value'].sum() * 1e3 # MW
        
        # Fuel cost
        try:
            eff_temp += GDATA.loc[(G, 'GDFE'), 'Value'] * Gcap / totalcap # Same prices everywhere as in DK
            # print('Added fuel cost: ', mc_cost_temp)
        except:
            pass
        
        
    return eff_temp

def get_capex(cap: pd.DataFrame, idx_cap: pd.Index, GDATA: pd.DataFrame, ANNUITYCG: pd.DataFrame):
    """Gets average marginal cost of generators in cap[idx_cap], provided VOM, fuel and emission policy data 

    Args:
        cap (_type_): _description_
        idx_cap (_type_): _description_
        GDATA (_type_): _description_
    """
    
    capex_temp = 0
    for G in cap[idx_cap].G.unique():
        Gcap = cap.loc[idx_cap & (cap.G == G), 'Value'].sum() * 1e3 # MW
        
        # Country
        country = cap.loc[idx_cap & (cap.G == G), 'C'].unique()[0] # Will just be one country as region is filtered
        
        try:
            capex_temp += GDATA.loc[(G, 'GDINVCOST0'), 'Value'] * ANNUITYCG.loc[(country, G), 'Value'] * Gcap * 1e6 # €
            # print('Added fuel cost: ', mc_cost_temp)
        except:
            pass
        
    return capex_temp

#%% ------------------------------- ###
###            5. Classes           ###
### ------------------------------- ###

class BC:
    """A class for handling the binding constraints of Antares"""

    def __init__(self, path_to_antares_study: str = './Antares'):
        self.study_path = path_to_antares_study
        cf = configparser.ConfigParser()
        cf.read(os.path.join(self.study_path, 'input/bindingconstraints/bindingconstraints.ini'))
        
        bc_name_to_idx = {}
        section_names = []
        for section in cf.sections():
            name = cf.get(section, 'name')
            # The operator is part of the filename in Antares 8.7
            operator = cf.get(section, 'operator').replace('greater', 'gt').replace('less', 'lt').replace('equal', 'eq')
            name = name + '_' + operator
                
        self.sections = section_names
        self._bc_name_to_idx = bc_name_to_idx
        self._cf = cf

    def get(self, section: str, parameter: str):
        return self._cf.get(self._bc_name_to_idx[section], parameter)


class IncFile:
    """A useful class for creating .inc-files for GAMS models 
    Args:
    prefix (str): The first part of the .inc file.
    body (str): The main part of the .inc file.
    suffix (str): The last part of the .inc file.
    name (str): The name of the .inc file.
    path (str): The path to save the file, defaults to 'Balmorel/base/data'.
    """
    def __init__(self, prefix='', body='', suffix='',
                 name='name', path='Balmorel/base/data/'):
        self.prefix = prefix
        self.body = body
        self.suffix = suffix
        self.name = name
        self.path = path

    def save(self):
        if self.path[-1] != '/':
            self.path += '/'
        if self.name[-4:] != '.inc':
            self.name += '.inc'  
        
        with open(self.path + self.name, 'w') as f:
            f.write(self.prefix) 
            f.write(self.body)
            f.write(self.suffix)

def ReadIncFilePrefix(name, incfile_prefix_path, weather_year):
    
    if ('WND' in name) | ('SOLE' in name) | ('DE' in name) | ('WTR' in name):
        string = "* Weather year %d from Antares\n"%(weather_year+ 1) + ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())
    else:
        string = ''.join(open(incfile_prefix_path + '/%s.inc'%name).readlines())
    
    return string


class AntaresOutput:
    
    def __init__(self, result_name: str, folder_name: str='Antares', wk_dir: str='.'):
        # Set path to result
        self.path = os.path.join(wk_dir, folder_name, 'output', result_name)
        try:
            self.mc_years = os.listdir(os.path.join(self.path, 'economy/mc-ind'))
            self.mc_years.sort()
        except FileNotFoundError:
            self.mc_years = None
        self.name = result_name
        self.wk_dir = wk_dir
        
    # Function to load area results
    def load_area_results(self, node: str, result_type: str='values', temporal: str='hourly',
                          mc_year: str='mc-all'):
        if mc_year == 'mc-all':
            return pd.read_table(os.path.join(self.path, 'economy/%s/areas/%s/%s-%s.txt'%(mc_year.lower(), node.lower(), 
                                                                                            result_type.lower(), temporal.lower())),
                                                                                            skiprows=[0,1,2,3,5,6]) 
        else:
            mc_year = convert_int_to_mc_year(mc_year)
            return pd.read_table(os.path.join(self.path, 'economy/mc-ind/%s/areas/%s/%s-%s.txt'%(mc_year.lower(), node.lower(), 
                                                                                            result_type.lower(), temporal.lower())),
                                                                                            skiprows=[0,1,2,3,5,6]) 
    
    # Function to load column result from many areas
    def collect_result_areas(self, nodes: list, column: str, result_type: str='values', temporal: str='hourly',
                            mc_year: str='mc-all'):
        res = pd.DataFrame(columns=nodes)
        for node in nodes:
            res[node] = self.load_area_results(node, result_type, temporal, mc_year)[column]

        res.columns.name = column
        return res
    
    def load_link_results(self, nodes: list[str, str], result_type: str='values', 
                          temporal: str='hourly', mc_year: str='mc-all'):
        """
        Will load results from nodes[0] -> nodes[1]
        """
        if mc_year == 'mc-all':
            return pd.read_table(os.path.join(self.path, 'economy/%s/links/%s - %s/%s-%s.txt'%(mc_year.lower(), nodes[0].lower(), 
                                                                                               nodes[1].lower(), 
                                                                                               result_type.lower(), temporal.lower())),
                                                                                               skiprows=[0,1,2,3,5,6]) 
        else:
            mc_year = convert_int_to_mc_year(mc_year)
            return pd.read_table(os.path.join(self.path, 'economy/mc-ind/%s/links/%s - %s/%s-%s.txt'%(mc_year.lower(), nodes[0].lower(),
                                                                                            nodes[1].lower(), 
                                                                                            result_type.lower(), temporal.lower())),
                                                                                            skiprows=[0,1,2,3,5,6]) 

    # Function to load and calculate median results
    def collect_mcyears(self, column: str, node_or_nodes: Union[str, list], 
                        result_type: str='values', temporal: str='hourly'):
        """
        If providing node_or_nodes is a string, it will be interpreted as an area result, 
        while anything else is assumed link result 
        """
        
        # Choose function
        if type(node_or_nodes) == str:
            func = self.load_area_results
        else:
            func = self.load_link_results
            
        if self.mc_years != None:
            for mc_year in self.mc_years:
                
                # Create temporary variable at first mc_year
                if not('temp' in locals()):
                    # Load
                    temp = func(node_or_nodes, result_type, temporal, mc_year)
                    
                    # Only keep desired column
                    temp = pd.DataFrame(data=temp[column].values, index=temp.index,
                                        columns=[mc_year])
                    
                    # Convert name of column to mc_year name
                    temp.columns = [mc_year] 

                # Append to temporary variable
                else:
                    temp[mc_year] = func(node_or_nodes, result_type, temporal, mc_year)[column].values
                    
            # Calculate quantile
            return temp
        
        else:
            # print('No mc-year results')
            return 0

def convert_int_to_mc_year(mc_year: int):
    # Make mc_year into correct format
    mc_year = ''.join(['0' for i in range(5-len(str(mc_year)))]) + str(mc_year) 
    return mc_year

#%% ------------------------------- ###
###          6. Utilities           ###
### ------------------------------- ###

# By ChatGPT
def find_and_copy_files(source_folder, destination_folder, file_contains):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through the source folder and its subdirectories
    for foldername, subfolders, filenames in os.walk(source_folder):
        for filename in filenames:
            # Check if the file has the desired extension
            if file_contains in filename:
                source_path = os.path.join(foldername, filename)
                destination_path = os.path.join(destination_folder, filename)

                # Copy the file to the destination folder
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {filename}")

def check_antares_compilation(wait_sec: int, max_waits: int, N_errors: int):
    """
    Peri-Processing usually takes around 100 s, while Antares compilation is around 60 s
    """
    i = 0    
    compile_finished = False
    while not(compile_finished) and i < max_waits:
        
        # Check if Peri-Processing is being run
        periproc_finished = open('Workflow/MetaResults/periprocessing_finished.txt', 'r')
        periproc_finished = periproc_finished.readline()
        periproc_finished = periproc_finished == 'True'
        print('\nChecking if peri-processing has finished..', periproc_finished)
        # print(periproc_finished)
        
        # Check latest Antares log
        latest_ant_log = os.listdir('Antares/logs')
        latest_ant_log.sort()
        latest_ant_log = latest_ant_log[-1]
        
        print('Reading log:', latest_ant_log)    

        log = open(os.path.join('Antares/logs', latest_ant_log), 'r')

        # Convert to pandas series for finding text
        log = pd.Series(log.readlines())
        
        # Check if the simulation has compiled (i.e., started)
        compile_finished = len(log[log.str.find('Starting the simulation') != -1]) == 1
        # print(log[log.str.find('Starting the simulation') != -1].to_string())
        
        # Check both events
        compile_finished = compile_finished and periproc_finished
        
        # If it didn't compile, then wait
        if not(compile_finished):
            print('Peri-processing is still running or Antares still compiling, waiting %0.2f min..'%(wait_sec/60))
            print('Remember to check if last Antares run ever started - otherwise this might be stuck!')
            sys.stdout.flush()
            
            time.sleep(wait_sec)
            i += 1
            
            
    if compile_finished and periproc_finished:
        print('Peri-processing is not running and Antares finished compiling, starting peri-processing and Antares execution now!')
        
        # Set periprocessing_finished to false (will be set to true after peri-processing finishes)
        with open('Workflow/MetaResults/periprocessing_finished.txt', 'w') as f:
            f.write('False')
    else:
        print('Waited for %0.2f min. and exiting now as this must be due to an error - or maybe last Antares run never started'%(i*wait_sec / 60))
        N_errors += 1
        
    return compile_finished, N_errors
    

if __name__ == '__main__':
    
    print('Test of loading binding constraint:')
    cf = BC()
    
    print(
        'Load fr_psp type and operator:',
        cf.get('fr_psp', 'type'),
        cf.get('fr_psp', 'operator')
    )
    