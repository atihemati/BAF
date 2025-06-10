"""
Created on 16.05.2024

@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import configparser
import gams
import pickle
from pybalmorel.utils import symbol_to_df
from Workflow.Functions.GeneralHelperFunctions import AntaresOutput

style = 'report'

if style == 'report':
    plt.style.use('default')
    fc = 'white'
elif style == 'ppt':
    plt.style.use('dark_background')
    fc = 'none'

with open('Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
    A2B_regi = pickle.load(f)
    
with open('Pre-Processing/Output/A2B_regi_h2.pkl', 'rb') as f:
    A2B_regi_h2 = pickle.load(f)


def try_to_load(func, *args, **kwargs):
    try:
        return func(*args, **kwargs) 
    except Exception as e:
        if type(e) == AttributeError:
            print('No result')
            return 1
        elif type(e) == FileNotFoundError:
            print('No values here or not finished')
            return 2
        else:
            raise e

def all_flows_positive(flows, tests, SC, flow_type, flow_name):
    if type(flows) != int:
        flows = flows['FLOW LIN..2']
        if np.all(flows >= 0):
            # print('All flows from')
            tests = pd.concat((tests, pd.DataFrame(index=[SC], columns=['Test', 'Subset', 'Result'], 
                                                    data=[[flow_type, flow_name, True]]))) 
            
            
        else:
            tests = pd.concat((tests, pd.DataFrame(index=[SC], columns=['Test', 'Subset', 'Result'], 
                                                    data=[[flow_type, flow_name, False]]))) 
            # print('%s, flow negative in %s - x_c3!'%(SC, reg))
        
    return tests

#%% ------------------------------- ###
###             1. Main             ###
### ------------------------------- ###

tests = pd.DataFrame(columns=['Test', 'Subset', 'Result'])
for path in ['.','../eur-system/balmorel-antares']:
    for result in os.listdir(os.path.join(path, 'Antares/output')):
        
        # Load Antares Result
        res = AntaresOutput(result, wk_dir=path)
        SC = res.name.split('eco-')[1]
        SC_name = SC.split('_')[0].replace('ht', 'HT').replace('lt', 'LT').replace('fictdemfunc', 'FictDemFunc').replace('capcred','CapCred').replace('max', 'Max').replace('exp', 'Exp').replace('marketvalue', 'MarketValue').replace('cons', 'Cons').replace('risk', 'Risk')
        SC_balm_name = SC_name + '_' + SC.split('_')[1].capitalize()
        year = SC.split('_y-')[1]
        print('Loading %s from %s'%(SC, res.path))
        
        # Load Config
        Config = configparser.ConfigParser()
        Config.read(os.path.join(path, 'Workflow/MetaResults/%s_meta.ini'%SC_name))
        SC_folder = Config.get('RunMetaData', 'SC_folder')
        
        # Load Balmorel Result
        ws = gams.GamsWorkspace()
        # print(os.path.abspath(os.path.join(path, "Balmorel/%s/model/MainResults_%s.gdx"%(SC_folder, SC_balm_name))))
        db = ws.add_database_from_gdx(os.path.abspath(os.path.join(path, "Balmorel/%s/model/MainResults_%s.gdx"%(SC_folder, SC_balm_name))))
        cap = symbol_to_df(db, 'G_CAP_YCRAF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Var', 'Units', 'Value'])
        
        for reg in A2B_regi.keys():
        # for reg in ['DKW1']:
            ### 1.0 Check x_c3 flow            
            flows = try_to_load(res.load_link_results, [reg.lower(), 'x_c3']) # Minmimum flow (should never be negative)
            tests = all_flows_positive(flows, tests, SC, 'Positive x_c3 flow?', '%s - x_c3'%reg)
                    
            # Now, check if max flows are equal to electrolyser capacity (load MainResults)
            if type(flows) != int:
                # Get max flows now
                maxflow = flows['FLOW LIN..3'].max() # Minmimum flow (should never be negative)
                
                # print(cap.loc[(cap.Y == year) & (cap.R == A2B_regi[reg][0]) & (cap.Tech == 'ELECTROLYZER'), 'Value'].sum())
                elcap = cap.loc[(cap.Y == year) & (cap.R == A2B_regi[reg][0]) & (cap.Tech == 'ELECTROLYZER'), 'Value'].sum() * 1000 # MWel
                elcap_high = elcap / 0.64
                elcap_low  = elcap / 0.76
                # print('Electrolyser el capacity (65%) in %s, %s: %0.2f MWel'%(SC, reg, elcap_high))
                # print('Electrolyser el capacity (72%) in %s, %s: %0.2f MWel'%(SC, reg, elcap_low))
                # print('Max x_c3 flow in         %s, %s: %0.2f MWel'%(SC, reg, maxflow))
                if (maxflow <= elcap_high):
                    tests = pd.concat((tests, pd.DataFrame(index=[SC], columns=['Test', 'Subset', 'Result'], 
                                            data=[['Flow obey electrolyser capacity?', '%s - x_c3'%reg, True]]))) 
                else:
                    tests = pd.concat((tests, pd.DataFrame(index=[SC], columns=['Test', 'Subset', 'Result'], 
                                            data=[['Flow obey electrolyser capacity?', '%s - x_c3'%reg, False]])))
                    print('%s - x_c3 maxflow didnt obey electrolyser capacity: '%reg)
                    print('Electrolyser capacity low %0.2f MWel </= Maxflow of %0.2f MWel </= Electrolyser capacity high %0.2f MWel'%(elcap_low, maxflow, elcap_high)) 

        for reg in A2B_regi_h2.keys():
            
            ### 1.1 Check x_c3 to hydrogen area flow (should only be positive)
            flows = try_to_load(res.load_link_results, ['x_c3', reg.lower()]) # Minmimum flow (should never be negative)
            tests = all_flows_positive(flows, tests, SC, 'Positive x_c3 flow?', 'x_c3 - %s'%reg)
            
            ### 1.2 Check z_taking flow            
            flows = try_to_load(res.load_link_results, [reg.lower(), 'z_taking']) # Minmimum flow (should never be negative)
            tests = all_flows_positive(flows, tests, SC, 'Positive z_taking flow?', '%s - z_taking'%reg)
            
            # Now, check if max flows are equal to fuel cell capacity (load MainResults)
            
            
print('All tests positive?', np.all(tests['Result']))
tests.to_csv('Workflow/OverallResults/Testresults.csv')