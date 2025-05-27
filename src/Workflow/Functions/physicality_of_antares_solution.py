"""
Check Physicality of Antares Solution

Will compare the electricity supplied from Antares results to the heat and hydrogen profiles of
Balmorel input data, and the capacity of other generators and storages for heat and hydrogen
from Balmorel results. 

Created on 26.05.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
import click
from pybalmorel import Balmorel, MainResults
from pybalmorel.utils import symbol_to_df

#%% ------------------------------- ###
###          1. Functions           ###
### ------------------------------- ###

class BalmorelFullTimeseries:
    def __init__(self, **kwargs):
        """A class to load the entire, absolute profiles (scaled by annual demands) from Balmorel input.
        Note that WEIGTH_S = 168 and WEIGHT_T = 1 is assumed.
        """
        
        gams_dir = kwargs.pop('gams_system_directory', None)
        
        self.model = Balmorel('Balmorel', gams_system_directory=gams_dir)
        self.model.locate_results()
        sc_folders = self.model.scfolder_to_scname.keys()
        
        # Placeholders
        symbols = {
            'heat' : 'DH',
            'electricity' : 'DE',
            'hydrogen' : 'HYDROGEN_DH2'
        }
        
        node_name = {
            'heat' : 'AAA',
            'electricity' : 'RRR',
            'hydrogen' : 'RRR'
        }
        
        S = ['S0%d'%i for i in range(1, 10)] + ['S%d'%i for i in range(10, 53)]
        T = ['T00%d'%i for i in range(1, 10)] + ['T0%d'%i for i in range(10, 100)] + ['T%d'%i for i in range(100, 169)] 
        self.profile_index = pd.MultiIndex.from_product((S, T))

        self.symbols = {key : [symbols[key], 
                               '%s_VAR_T'%symbols[key],
                               '%sUSER'%symbols[key],
                               node_name[key]] for key in symbols.keys()}
        
        self.profiles = {sc : {com : {} for com in symbols.keys()} for sc in sc_folders}
        
        self.results = {sc : False for sc in self.model.scenarios}
        
    def load_data(self, scenario: str, **kwargs):
        """Load input data from Balmorel scenario. 
        NOTE: This might require a lot of RAM for large-scale scenarios, consider only loading pickled files instead of setting them all as attributes.

        Args:
            scenario (str): The scenario that has a MainResults file. Will automatically find scenario folder
        """
        
        # Load GAMS database
        overwrite = kwargs.pop('overwrite', False)
        self.model.load_incfiles(self.model.scname_to_scfolder[scenario], overwrite=overwrite)
        self.results[scenario] = MainResults(files='MainResults_%s.gdx'%scenario, paths='Balmorel/%s/model'%self.model.scname_to_scfolder[scenario])
        sc_folder = self.model.scname_to_scfolder[scenario]
        
        # Load GAMS symbols from database
        for commodity in self.symbols.keys():
            file_base_name = f'Workflow/OverallResults/{scenario}_{commodity}'
            for i in range(2):
                # Does a pickle file exist and we don't want to overwrite?
                if os.path.exists('%s_%s.pkl'%(file_base_name, self.symbols[commodity][i])) and not(overwrite):
                    with open(file_base_name + '_' + self.symbols[commodity][i] + '.pkl', 'rb') as f:
                        setattr(self, self.symbols[commodity][i], pkl.load(f))
                        
                # Load the symbols and create a pickle file
                else:
                    f0 = symbol_to_df(self.model.input_data[sc_folder], self.symbols[commodity][i])
                    with open(file_base_name + '_' + self.symbols[commodity][i] + '.pkl', 'wb') as f:
                        pkl.dump(f0, f)
                    setattr(self, self.symbols[commodity][i], f0)   

                
        
    def get_input_profile(self, scenario: str, year: int, commodity: str, region: str, user: str, **kwargs):
        """Get the absolute profile from Balmorel input data. Assuming WEIGTH_S = 168 and WEIGHT_T = 1.
        NOTE: Doesn't take into account that GAMS interprets 0 as 'nothing', meaning there can be temporal gaps in the profiles. 

        Args:
            scenario (str): Scenario
            year (int): Model year
            commodity (str): Commodity, either 'heat', 'electricity' or 'hydrogen'
            region (str): The node to get a profile from
            user (str): The demand user group. Defaults to all, which will load all users.

        Raises:
            KeyError: _description_
        """
        
        commodity = commodity.lower()
        sc_folder = self.model.scname_to_scfolder[scenario]
        
        # Check if model input data was loaded (allows for overwriting if desired)
        if not(sc_folder in self.model.input_data.keys()):
            self.load_data(scenario, **kwargs)
        
        # Check if profile previously loaded
        if commodity != 'hydrogen' and user in self.profiles[sc_folder][commodity].keys():
            return self.profiles[sc_folder][commodity][user]
        elif commodity == 'hydrogen' and 'H2USER' in self.profiles[sc_folder][commodity].keys():
            return self.profiles[sc_folder][commodity]['H2USER']
        
        # If not, start loading symbols
        try:
            DC = getattr(self, self.symbols[commodity][0])
            DC_VAR_T = getattr(self, self.symbols[commodity][1])
        except KeyError:
            raise KeyError(f"Commodity '{commodity}' not yet covered by the BalmorelDemandProfileData class")
            
        # Region and year filtering
        year = str(year)
        region = region.upper()
        user_name = self.symbols[commodity][2]
        node_name = self.symbols[commodity][3]
        DC = DC.query(f'YYY == @year and {node_name} == @region')
        DC_VAR_T = DC_VAR_T.query(f'{node_name} == @region')
        
        # Make artifical H2 user
        if commodity.lower() == 'hydrogen':
            # Only one user for hydrogen
            user = 'H2USER'
            DC[user_name] = user
            DC_VAR_T[user_name] = user
        
        # Query the choice of electricity or heat user
        querystring = f"{user_name} == '{user}'" 
        DC0 = DC.query(querystring)
        DC_VAR_T0 = DC_VAR_T.query(querystring)
        
        # Create absolute profile
        profile = DC_VAR_T0.pivot_table(index=[user_name, 'SSS', 'TTT'], values='Value', aggfunc='sum')
        annual_demand = DC0.pivot_table(index=user_name, values='Value', aggfunc='sum').sum()
        absolute_profile = (annual_demand * profile / profile.sum()).loc[user]
        
        # Fill missing values as 0 (GAMS interprets 0 as 'nothing')
        absolute_profile = absolute_profile.reindex(self.profile_index, fill_value=0)
        
        # Store
        self.profiles[sc_folder][commodity][user] = absolute_profile 

        return absolute_profile

#%% ------------------------------- ###
###            2. Main              ###
### ------------------------------- ###

@click.command()
@click.option('--dark-style', is_flag=True, required=False, help='Dark plot style')
@click.argument('scenario', type=str)
@click.argument('commodity', type=str)
def main(scenario: str, commodity: str, dark_style: bool):

    # Set global style of plot
    if dark_style:
        plt.style.use('dark_background')
        fc = 'none'
    else:
        fc = 'white'
        
    # Load data
    model_year = 2050
    node_category = 'Area'
    node = 'DE_A'
    user = 'RESH'
    balmorel_input = BalmorelFullTimeseries()
    profile = balmorel_input.get_input_profile(scenario, 2050, commodity, node, user)
    efficiencies = (
        pd.read_csv('Balmorel/%s/model/GDATA.csv'%balmorel_input.model.scname_to_scfolder[scenario])
        .query('GDATASET == "GDFE"')
        .astype({'Val' : float})
        .pivot_table(index='GGG', values='Val')
    )

    # Find capacities of other generation and storages
    gencap = (
        balmorel_input
        .results[scenario]
        .get_result('G_CAP_YCRAF')
        .query(f"Year == '{model_year}' and {node_category} == '{node}' and Commodity == '{commodity.upper()}'")
    )
    stocap = (
        balmorel_input
        .results[scenario]
        .get_result('G_STO_YCRAF')
        .query(f"Year == '{model_year}' and {node_category} == '{node}' and Commodity == '{commodity.upper()}'")
    ) 

    # To what extend can other generation capacities fulfill demand
    print(profile - gencap['Value'].mul(1e3).sum())



if __name__ == '__main__':
    main()
