"""
Created on 30.03.2023 by
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)

IN ONE SENTENCE:
Transfers results from Balmorel to Antares input

ASSUMPTIONS IN SECTIONS:
- 0.6 Dictionaries are hard-coded, based on current Antares/Balmorel set definitions
      Country list assumes country key is the same for Balmorel+Antares, and that it's in the first 2 letters of the regions
- 1.2 Peak production in VRE series = Peak capacity (but 5% loss inherent in profile, see Pre-Processing.py)
- 4.3 Full transmission capacity available all hours 
- 5.4 Constant hydrogen demand assumed
"""
#%% ------------------------------- ###
###       0. Script Settings        ###
### ------------------------------- ###


try:
    print('\n|--------------------------------------------------|')   
    print('              PERI-PROCESSING')
    print('|--------------------------------------------------|\n') 
    
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import pandas as pd
    from pandas.errors import EmptyDataError
    import numpy as np
    import matplotlib.pyplot as plt
    import gams
    import platform
    OS = platform.platform().split('-')[0] # Assuming that linux will be == HPC!

    import os
    if ('Workflow' in __file__) | ('Pre-Processing' in __file__):
        os.chdir(os.path.dirname(os.path.dirname(__file__)))

    import sys
    import pickle
    import configparser
    from Workflow.Functions.GeneralHelperFunctions import symbol_to_df, create_transmission_input, get_marginal_costs, doLDC, get_efficiency, get_capex


    if not('SC_name' in locals()): 
        try:
            # Try to read something from the command line
            SC_name = sys.argv[2]
        except:
            # Otherwise, read config from top level
            print('Reading SC from Config.ini..') 
            Config = configparser.ConfigParser()
            Config.read('Config.ini')
            SC_name = Config.get('RunMetaData', 'SC')

    ### 0.0 Load configuration file
    Config = configparser.ConfigParser()
    Config.read('Workflow/MetaResults/%s_meta.ini'%SC_name)
    SC_folder = Config.get('RunMetaData', 'SC_Folder')
    UseAntaresData = Config.getboolean('PeriProcessing', 'UseAntaresData')
    UsePseudoBenders = Config.getboolean('PostProcessing', 'PseudoBenders')
    UseFlexibleDemand = Config.getboolean('PeriProcessing', 'UseFlexibleDemand')

    # Plot settings
    style = Config.get('Analysis', 'plot_style')
    if style == 'report':
        plt.style.use('default')
        fc = 'white'
    elif style == 'ppt':
        plt.style.use('dark_background')
        fc = 'none'

    ### 0.2 Checking if running this script by itself
    if __name__ == '__main__':
        test_mode = 'Y' # Set to N if you're running iterations
        print('\n----------------------------\n\n        Test mode ON\n\n----------------------------\n')
        wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  
        year = '2050' 	# Manual testing
        year = sys.argv[1]  # Execution script using a command prompt followed by the year

    else:
        test_mode = 'N'
        wk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  

    ant_study = '/Antares' # the specific antares study

    ### 0.3 Technologies transfered from Balmorel, with marginal costs
    with open(wk_dir + '/Pre-Processing/Output/BalmTechs.pkl', 'rb') as f:
        BalmTechs = pickle.load(f)

    with open(wk_dir + '/Workflow/OverallResults/%s_AT.pkl'%SC_name, 'rb') as f:
        fAntTechno = pickle.load(f)

    ### 0.4 Dictionaries for Balmorel/Antares set translation

    # Renewables
    B2A_ren = {'SOLAR-PV' : 'solar',
                'WIND' : 'wind'}

    # Regions
    with open(wk_dir + '/Pre-Processing/Output/B2A_regi.pkl', 'rb') as f:
        B2A_regi = pickle.load(f)
    with open(wk_dir + '/Pre-Processing/Output/B2A_regi_h2.pkl', 'rb') as f:
        B2A_regi_h2 = pickle.load(f)

    with open(wk_dir + '/Pre-Processing/Output/A2B_regi.pkl', 'rb') as f:
        A2B_regi = pickle.load(f)
    with open(wk_dir + '/Pre-Processing/Output/A2B_regi_h2.pkl', 'rb') as f:
        A2B_regi_h2 = pickle.load(f)


    # Countries
    C = pd.Series(list(B2A_regi.keys())).str[:2].unique()

    # Weights
    with open(wk_dir + '/Pre-Processing/Output/B2A_DH2_weights.pkl', 'rb') as f:
        B2A_DH2_weights = pickle.load(f) # Hydrogen
    with open(wk_dir + '/Pre-Processing/Output/B2A_DE_weights.pkl', 'rb') as f:
        B2A_DE_weights = pickle.load(f) # Electricity


    # Base factor on fictive electricity demand 
    fict_de_factor = 1

    # GDATA
    ws = gams.GamsWorkspace()
    ALLENDOFMODEL = ws.add_database_from_gdx(wk_dir + '/Balmorel/%s/model/all_endofmodel.gdx'%SC_folder)
    GDATA = symbol_to_df(ALLENDOFMODEL, 'GDATA', ['G', 'Par', 'Value']).groupby(by=['G', 'Par']).aggregate({'Value' : 'sum'})
    FDATA = symbol_to_df(ALLENDOFMODEL, 'FDATA', ['F', 'Type', 'Value']).groupby(by=['F', 'Type']).aggregate({'Value' : 'sum'})
    FPRICE = symbol_to_df(ALLENDOFMODEL, 'FUELPRICE1', ['Y', 'R', 'F', 'Value']).groupby(by=['Y', 'R', 'F']).aggregate({'Value' : 'sum'})
    EMI_POL = symbol_to_df(ALLENDOFMODEL, 'EMI_POL', ['Y', 'C', 'Group', 'Par', 'Value']).groupby(by=['Y', 'C', 'Group', 'Par']).aggregate({'Value' : 'sum'})
    ANNUITYCG = symbol_to_df(ALLENDOFMODEL, 'ANNUITYCG', ['C', 'G', 'Value']).groupby(by=['C', 'G']).aggregate({'Value' : 'sum'})

    # Distribution loss
    DISLOSSEL = symbol_to_df(ALLENDOFMODEL, 'DISLOSS_E', ['R', 'Value']).pivot_table(index='R',
                                                                                   values='Value')
    DISLOSSH2 = symbol_to_df(ALLENDOFMODEL, 'DISLOSS_H2', ['R', 'Value']).pivot_table(index='R',
                                                                                   values='Value')

    ### 0.5 Iteration Data
    i = Config.getint('RunMetaData', 'CurrentIter')
    
    ## Scenario
    SC = SC_name + '_Iter%d'%i 
 

    ### 0.6 Placeholders
    if UsePseudoBenders:
        CapacityInRegion = {}

    print('Loading results for year %s from Balmorel/%s/model/MainResults_%s.gdx\n'%(year, SC_folder, SC))
    ### 0.7 Load MainResults
    db = ws.add_database_from_gdx(wk_dir + "/Balmorel/%s/model/MainResults_%s.gdx"%(SC_folder, SC))

    ### 0.7 Flexible Demand Profile
    if UseFlexibleDemand:   
        flexdem_profile = pd.read_csv('Pre-Processing/Output/inverse_traffic_count.csv', header=None)
        flexdem_profile = np.hstack([flexdem_profile[0].values]*53)[:8760]

    #%% ------------------------------- ###
    ###   1. Wind and Solar Capacities  ###
    ### ------------------------------- ###

    print('\nVRE capacities to Antares...\n')


    ### 1.2 Capacities to dataframe
    cap = symbol_to_df(db, "G_CAP_YCRAF",
                    ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 'Tech', 'Var', 'Unit', 'Value'])

    ## Placeholder for saving total generation capacities from Balmorel in aggregated form
    # cap_sum = pd.DataFrame({tech : np.zeros(len(cap.R.unique())) for tech in ['WIND', 'SUN', 'BIOGAS']},
    #                   index=cap.R.unique())
    # sim_year = cap['Y'][0] # Assuming we're only running one year!
    
    ## Load mapped areas
    VRE_MAPPING = pd.read_csv(wk_dir + '/Pre-Processing/Output/AreaMapping.csv', index_col=0)

    for tech in B2A_ren.keys(): 
        
        # Filter tech
        if tech == 'WIND':
            idx = ((cap['Tech'] == 'WIND-OFF') | (cap['Tech'] == 'WIND-ON')) & (cap['Y'] == year)    
        else:
            idx = (cap['Tech'] == tech) & (cap['Y'] == year)
        p = '..' + ant_study + '/input/%s/series/'%(B2A_ren[tech])
        
        # Iterate through Antares areas
        for a in A2B_regi.keys():
                
            if not(a in ['ITCO']):
                # Read Antares Config file for area a
                area_config = configparser.ConfigParser()
                area_config.read('Antares/input/renewables/clusters/%s/list.ini'%a.lower())
                    
                # Sum capacity from Balmorel Regions
                tech_cap = 0
                # If not using antares data: Weight wrt. electricity demand and divide by amount of areas
                if not(UseAntaresData):
                    for BalmR in A2B_regi[a]: 
                        # Get amount of antares regions 
                        idx2 = idx & (cap.R == BalmR)
                        if not(cap.loc[idx2].empty):
                            tech_cap += cap.loc[idx2, 'Value'].sum()*1000 * B2A_DE_weights[BalmR][a]
                else:
                    # If Balmorel is higher resolved:
                    if len(A2B_regi[a]) > 1:
                        for BalmArea in A2B_regi[a]:
                            tech_cap += cap.loc[idx & (cap.R == BalmArea), 'Value'].sum() * 1000
                    else:                   
                        idx_cap = idx & (cap.A == a + '_A')
                        tech_cap = cap.loc[idx_cap, 'Value'].sum() * 1000
                        capex = get_capex(cap, idx_cap, GDATA, ANNUITYCG)
                    
                if (tech_cap > 1e-5):
                    area_config.set('%s_%s_0'%(a, B2A_ren[tech]), 'nominalcapacity', str(tech_cap))
                    area_config.set('%s_%s_0'%(a, B2A_ren[tech]), 'enabled', 'true')
                else:
                    area_config.set('%s_%s_0'%(a, B2A_ren[tech]), 'nominalcapacity', '0')
                    area_config.set('%s_%s_0'%(a, B2A_ren[tech]), 'enabled', 'false')
                                        
                # Save data
                # ASSUMPTION: Peak production = 95% of Capacity (See pre-processing script)
                # ((f * tech_cap).astype(int)).to_csv(p + B2A_ren[tech] + '_%s.txt'%a, sep='\t', header=None, index=None)
                with open('Antares/input/renewables/clusters/%s/list.ini'%a.lower(), 'w') as configfile:
                    area_config.write(configfile)
                print(a, B2A_ren[tech], round(tech_cap, 2), 'MW')

            # Save technoeconomic data to file
            if tech == 'WIND':
                techname = 'wind onshore'
            else:
                techname = 'solar pv'
            fAntTechno.loc[(i, year, a, techname), 'CAPEX'] = capex
            fAntTechno.loc[(i, year, a, techname), 'OPEX'] = 0
            fAntTechno.loc[(i, year, a, techname), 'Power Capacity'] = tech_cap 
            
    #%% ------------------------------- ###
    ###      2. Thermal Capacities      ###
    ### ------------------------------- ###

    print('\nThermal capacities to Antares...\n')

    # Get costs
    eco = symbol_to_df(db, 'ECO_G_YCRAG', ['Y', 'C', 'R', 'A', 'G', 'F', 
                                        'Tech', 'Var', 'Subvar', 'Unit', 'Value'])

    # Get production
    pro = symbol_to_df(db, 'PRO_YCRAGF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity', 
                                        'Tech', 'Unit', 'Value'])

    # Read the binding constraint
    Config = configparser.ConfigParser()
    Config.read('Antares/input/bindingconstraints/bindingconstraints.ini')

    # Placeholders for modulation and data
    thermal_modulation = '\n'.join(['1\t1\t1\t0' for i in range(8760)]) + '\n'
    thermal_data = '\n'.join(['1\t1\t0\t0\t0\t0' for i in range(365)]) + '\n'


    ### 2.1 Go through regions
    for area in A2B_regi.keys():
        
        if not(area in ['ITCO']):
            ### 2.2 Get tech capacities
            if UsePseudoBenders:
                CapacityInRegion[area] = {}
            thermal_cap = "" # String for .ini file
            
            # Technologies as defined by aggregated tech categories in BalmTechs dict
            for tech in BalmTechs.keys():
                if 'CCS' in tech:
                    CCStech = True
                else:
                    CCStech = False
                
                # Fuels as defined by BalmTechs dict
                for fuel in BalmTechs[tech].keys():
    
                    tech_cap = 0
                    mc_cost = 0
                    Nreg = 0 # Amount of Balmorel regions with this technology
                    eff = 0 # Efficiency
                    capex = 0
                    for BalmArea in A2B_regi[area]:
                        # Get weight from amount of corresponding areas in Balmorel
                        weight = B2A_DE_weights[BalmArea][area]
                                    
                        # Index for capacities
                        idx_cap = (cap['Commodity'] == 'ELECTRICITY') & (cap.R == BalmArea) & (cap.F == fuel) & (cap.Tech == tech.replace('-CCS', '')) & (cap.Y == year)    
                        
                        # Index for marginal costs
                        idx = (eco['Var'] == 'COSTS') & ((eco['Subvar'] == 'GENERATION_OPERATIONAL_COSTS') |\
                            (eco['Subvar'] == 'GENERATION_FUEL_COSTS') | (eco['Subvar'] == 'GENERATION_CO2_TAX')) & (eco['Tech'] == tech.replace('-CCS', '')) & (eco['F'] == fuel) &\
                                (eco['R'] == BalmArea) & (eco['Y'] == year)

                        # Index for production
                        idx2 = (pro['Commodity'] == 'ELECTRICITY') & (pro['R'] == BalmArea) & (pro['F'] == fuel) & (pro['Tech'] == tech.replace('-CCS', '')) & (pro['Y'] == year)
                        
                        # Filtering CCS techs
                        if CCStech:
                            idx_cap = idx_cap & (cap.G.str.find('CCS') != -1) 
                            idx = idx & (eco.G.str.find('CCS') != -1)
                            idx2 = idx2 & (pro.G.str.find('CCS') != -1)
                        else:
                            idx_cap = idx_cap & (cap.G.str.find('CCS') == -1) 
                            idx = idx & (eco.G.str.find('CCS') == -1)
                            idx2 = idx2 & (pro.G.str.find('CCS') == -1)
                            
                        
                        tech_cap += weight*cap.loc[idx_cap, 'Value'].sum()*1e3
                        # Get marginal costs of production
                        if cap.loc[idx_cap, 'Value'].sum()*1e3 > 1e-5:
                        
                            # print(tech, fuel)
                            eff += get_efficiency(cap, idx_cap, GDATA)
                            capex += get_capex(cap, idx_cap, GDATA, ANNUITYCG)
                            Nreg += 1 # The technology existed in this region, so increment by one (used to average after)                   
                            
                            mc_cost_temp = get_marginal_costs(year, cap, idx_cap, fuel, GDATA, FPRICE, FDATA, EMI_POL)
                            
                            if not(pd.isna(mc_cost_temp)):
                                mc_cost += mc_cost_temp # Add to sum of marginal costs over Balmorel regions

                    # Only enable tech if there's a real capacity (filtering away LP low value results)
                    if tech_cap > 1e-5:
                        enabled = 'true'
                        
                        # Average marginal costs across Balmorel regions
                        try:
                            mc_cost = mc_cost / Nreg 
                            eff = eff / Nreg
                            em_factor = BalmTechs[tech][fuel]['CO2'] / eff
                            print(area, tech, fuel, '\nMarginal cost: %0.2f eur/MWh'%mc_cost, '\nCapacity: %0.2f MW'%tech_cap, '\nEfficiency: %0.2f pct\n'%(eff*100))
                        except ZeroDivisionError:
                            em_factor = 0
                            print('This capacity was not used')
                            
                        # No negative or zero marginal costs in Antares
                        if mc_cost <= 0:
                            mc_cost = 1
                            
                        # Store for Pseudo-Benders binding constraint
                        if UsePseudoBenders:
                            tech_cap = 5e5 # A very high number
                            CapacityInRegion[area]['_'.join((tech.lower(), fuel.lower()))] = True
                    
                    else:
                        # print(area, tech, fuel, '\nCapacity: %0.2f MW\n'%tech_cap)
                        enabled = 'false'
                        em_factor = 0
                        
                    # print(tech, fuel, 'co2: ', BalmTechs[tech][fuel]['CO2'])
                    
                    # Save capacity to string for .ini file
                    thermal_cap = thermal_cap +\
                                    """[%s_%s]\n
                                    name = %s_%s\n
                                    group = %s\n
                                    enabled = %s\n
                                    unitcount = 1\n
                                    nominalcapacity = %d\n
                                    co2 = %0.2f\n
                                    marginal-cost = %d\n
                                    market-bid-cost = %d\n\n"""%(tech.lower(), fuel.lower(),
                                                                    tech.lower(), fuel.lower(),
                                                                    fuel.lower(), enabled,
                                                                    int(round(tech_cap)),
                                                                    em_factor,
                                                                    int(round(mc_cost)),
                                                                    int(round(mc_cost)))
                    
                    # Create transmission capacity for hydrogen offtake, for fuel cell:
                    if (tech == 'FUELCELL') & (fuel == 'HYDROGEN'):
                        if ('z_h2_c3_' + area.lower() in A2B_regi_h2.keys()):
                                                            
                            # Capacity
                            create_transmission_input(wk_dir, ant_study, 'z_h2_c3_' + area.lower(), 'z_taking', tech_cap*2, 0)
                    
                            # Efficiency 
                            generator = '{reg}%{virtual_node}'.format(reg='z_h2_c3_' + area.lower(), virtual_node='z_taking')
                            for section in Config.sections():
                                if generator in Config.options(section):
                                    # print('%s is in section %s'%(generator, section))
                                    # print('Setting %s to efficiency %0.2f'%(generator, eff))
                                    Config.set(section, generator, '-' + str(round(eff, 6)))
                                    
                                    if tech_cap > 1e-5:
                                        Config.set(section, 'enabled', 'true')
                                    else:
                                        Config.set(section, 'enabled', 'false')
                    
                    # Save capacity timeseries (assuming no outage!)
                    temp = pd.Series(np.ones(8760) * tech_cap).astype(int)
                    if bool(enabled):
                        try:
                            temp.to_csv(wk_dir+'/Antares' + '/input/thermal/series/%s/%s_%s/series.txt'%(area.lower(), tech.lower(), fuel.lower()), sep='\t', header=False, index=False)
                        
                        except OSError:
                            os.mkdir(wk_dir+'/Antares' + '/input/thermal/series/%s/%s_%s'%(area.lower(), tech.lower(), fuel.lower()))
                            temp.to_csv(wk_dir+'/Antares' + '/input/thermal/series/%s/%s_%s/series.txt'%(area.lower(), tech.lower(), fuel.lower()), sep='\t', header=False, index=False) 
                        
                        try:
                            with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/modulation.txt'%(area.lower(), tech.lower(), fuel.lower()), 'w') as f:
                                f.write(thermal_modulation)        
                            with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/data.txt'%(area.lower(), tech.lower(), fuel.lower()), 'w') as f:
                                f.write(thermal_data) 
                                
                        except OSError:
                            os.mkdir(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s'%(area.lower(), tech.lower(), fuel.lower()))
                            with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/data.txt'%(area.lower(), tech.lower(), fuel.lower()), 'w') as f:
                                f.write(thermal_data) 
                            with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/modulation.txt'%(area.lower(), tech.lower(), fuel.lower()), 'w') as f:
                                f.write(thermal_modulation)

                    # Save technoeconomic data to file
                    fAntTechno.loc[(i, year, area, tech.lower()+'_'+fuel.lower()), 'CAPEX'] = capex
                    fAntTechno.loc[(i, year, area, tech.lower()+'_'+fuel.lower()), 'OPEX'] = mc_cost
                    fAntTechno.loc[(i, year, area, tech.lower()+'_'+fuel.lower()), 'Power Capacity'] = tech_cap 
                    
            # Save capacity in .ini
            with open(wk_dir + ant_study + '/input/thermal/clusters/%s/list.ini'%(area.lower()), 'w') as f:
                f.write(thermal_cap)     
            
            ### 2.3 Get Electrolyser Capacity
            idx_cap = (cap.Commodity == 'HYDROGEN') & (cap.Tech == 'ELECTROLYZER') & (cap.Y == year)
            temp = cap.loc[idx_cap]

            tech_cap = 0
            eff = 0
            N_reg = 0
            for BalmArea in A2B_regi[area]:
                weight = B2A_DE_weights[BalmArea][area]
                tech_cap += weight * temp[temp.R == BalmArea].Value.sum()*1e3 # MW H2 out
                if temp.loc[temp.R == BalmArea, 'Value'].sum()*1000 > 1e-6:   
                    eff += get_efficiency(cap, idx_cap & (cap.R == BalmArea), GDATA)
                    N_reg += 1
                        
            # Efficiency 
            generator = '{reg}%{tech}'.format(reg=area.lower(), tech='x_c3')
            for section in Config.sections():
                if generator in Config.options(section):
                    # print('%s is in section %s'%(generator, section))
                    # print('Setting %s to efficiency %0.2f'%(generator, eff))
                    Config.set(section, generator, str(round(eff, 6)))
                    
                    if tech_cap > 1e-5:
                        # Convert to el capacity in
                        eff = eff / N_reg
                        tech_cap = tech_cap / eff
                        
                        print(area, 'Electrolyser\nCapacity: %0.2f MW_EL'%tech_cap)
                        print('Efficiency: %0.2f pct\n'%(eff*100))
                        
                        Config.set(section, 'enabled', 'true')
                    else:
                        Config.set(section, 'enabled', 'false')
                
                
            
            # Save it
            try:
                create_transmission_input(wk_dir, ant_study, area.lower(), 'x_c3', 
                                        [tech_cap, 0], 0) 
                create_transmission_input(wk_dir, ant_study, 'x_c3', 'z_h2_c3_' + area.lower(), 
                                        [tech_cap*eff*1.01, 0], 0) # small overestimation of efficiency to take care of infeasibility due to rounding error (binding constraint should take care of correct flows however)
            except FileNotFoundError:
                print('No electrolyser option for %s\n'%area)

            # Save technoeconomic data to file
            fAntTechno.loc[(i, year, area, 'electrolyser'), 'OPEX'] = mc_cost
            fAntTechno.loc[(i, year, area, 'electrolyser'), 'Power Capacity'] = tech_cap 

    # Save configfile
    with open('Antares/input/bindingconstraints/bindingconstraints.ini', 'w') as configfile:
        Config.write(configfile)
    Config.clear()

        

    #%% 2.4 Steam Methane Reforming Capacity
    for area in A2B_regi_h2.keys():
        thermal_cap = ""
        for CCStech in [True, False]:    
            
            ### 2.5 Get tech capacities and costs
            tech = 'steam-methane-reforming'
            if CCStech:
                tech += '-ccs' 
            tech_cap = 0
            mc_cost = 0
            Nreg = 0
            eff = 0
            em_factor = 0
            capex = 0 
            for BalmArea in A2B_regi_h2[area]:
                weight = B2A_DH2_weights[BalmArea][area]
                
                if CCStech:
                    idx = (cap['Commodity'] == 'HYDROGEN') & (cap.R == BalmArea) & (cap.F == 'NATGAS') & (cap.Tech == 'STEAMREFORMING') & (cap.G.str.find('CCS') != -1)
                else:
                    idx = (cap['Commodity'] == 'HYDROGEN') & (cap.R == BalmArea) & (cap.F == 'NATGAS') & (cap.Tech == 'STEAMREFORMING') & (cap.G.str.find('CCS') == -1) 

                tech_cap += weight*cap.loc[idx, 'Value'].sum()*1e3
                
                if cap.loc[idx, 'Value'].sum()*1e3 > 1e-5:
                    eff += get_efficiency(cap, idx, GDATA)
                    mc_cost_temp = get_marginal_costs(year, cap, idx, 'NATGAS', GDATA, FPRICE, FDATA, EMI_POL)
                    Nreg += 1
                    capex += get_capex(cap, idx, GDATA, ANNUITYCG)
                        
                    if not(pd.isna(mc_cost_temp)):
                        mc_cost += mc_cost_temp
                                
            if tech_cap > 1e-6:
                enabled = 'true'
                
                try:
                    mc_cost = mc_cost / Nreg 
                    eff = eff / Nreg
                    # print(area, tech, fuel, '\nMarginal cost: %0.2f eur/MWh'%mc_cost, '\nCapacity: %0.2f MW\n'%tech_cap)
                except ZeroDivisionError:
                    print('This capacity was not used')
                            
                if CCStech:
                    em_factor = 0.20196/eff * 0.1 
                else:
                    em_factor = 0.20196/eff
                
                print(area, '%s\nMarginal cost: %0.2f\nCapacity: %0.2f MW_H2'%(tech, mc_cost, tech_cap))
                    

            else:
                enabled = 'false'
            
            
            
            # Save capacity to string for .ini file
            thermal_cap +=  """[%s_%s]\n
                            name = %s_%s\n
                            group = %s\n
                            enabled = %s\n
                            unitcount = 1\n
                            nominalcapacity = %d\n
                            co2 = %0.2f\n
                            marginal-cost = %d\n
                            market-bid-cost = %d\n\n"""%(tech, 'natgas',
                                                            tech, 'natgas',
                                                            'natgas', enabled,
                                                            int(round(tech_cap)),
                                                            em_factor, 
                                                            mc_cost,
                                                            mc_cost)
            
            # Save capacity timeseries
            temp = pd.Series(np.ones(8760) * tech_cap).astype(int)
            if bool(enabled):
                try:
                    temp.to_csv(wk_dir+'/Antares' + '/input/thermal/series/%s/%s_%s/series.txt'%(area.lower(), tech.lower(), 'natgas'), sep='\t', header=False, index=False)
                
                except OSError:
                    os.mkdir(wk_dir+'/Antares' + '/input/thermal/series/%s/%s_%s'%(area.lower(), tech.lower(), 'natgas'))
                    temp.to_csv(wk_dir+'/Antares' + '/input/thermal/series/%s/%s_%s/series.txt'%(area.lower(), tech.lower(), 'natgas'), sep='\t', header=False, index=False) 
                
                try:
                    with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/modulation.txt'%(area.lower(), tech.lower(), 'natgas'), 'w') as f:
                        f.write(thermal_modulation)        
                    with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/data.txt'%(area.lower(), tech.lower(), 'natgas'), 'w') as f:
                        f.write(thermal_data) 
                        
                except OSError:
                    os.mkdir(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s'%(area.lower(), tech.lower(), 'natgas'))
                    with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/data.txt'%(area.lower(), tech.lower(), 'natgas'), 'w') as f:
                        f.write(thermal_data)
                    with open(wk_dir+'/Antares' + '/input/thermal/prepro/%s/%s_%s/modulation.txt'%(area.lower(), tech.lower(), 'natgas'), 'w') as f:
                        f.write(thermal_modulation) 
                           
            # Save technoeconomic data to file
            tech_name = tech + '_natgas'
            fAntTechno.loc[(i, year, area, tech_name), 'CAPEX'] = capex
            fAntTechno.loc[(i, year, area, tech_name), 'OPEX'] = mc_cost
            fAntTechno.loc[(i, year, area, tech_name), 'Power Capacity'] = tech_cap 
     
        # Save capacity in .ini
        with open(wk_dir + ant_study + '/input/thermal/clusters/%s/list.ini'%(area.lower()), 'w') as f:
            f.write(thermal_cap)  



    #%% ------------------------------- ###
    ###      3. Storage Capacities      ###
    ### ------------------------------- ###

    print('\nStorage capacities to Antares...\n')

    ### 3.1 Placeholders and data
    h2_tank_list = ''
    h2_cavern_list = {}

    # Load results on energy capacity
    sto = symbol_to_df(db, 'G_STO_YCRAF', ['Y', 'C', 'R', 'A', 'G', 'F', 'Commodity',
                                        'Tech', 'Var', 'Unit', 'Value'])

    Hydro = configparser.ConfigParser()
    Hydro.read(wk_dir + ant_study + '/input/hydro/hydro.ini')


    ### 3.2 Battery Storage
    for area in A2B_regi.keys():
        
        
        if not(area in ['ITCO']):
            energy_cap = 0
            power_cap = 0
            capex = 0
            for BalmArea in A2B_regi[area]:
                ### Battery capacity
                # energy_cap += sto.loc[(sto.R == BalmArea) & (sto.Tech == 'INTRASEASONAL-ELECT-STORAGE') & (sto.G.str.find('BAT-LITHIO-PEAK') != -1), 'Value'].sum() * 1e3 # MWh
                idx_cap = (cap.R == BalmArea) & (cap.Tech == 'INTRASEASONAL-ELECT-STORAGE') & (cap.G.str.find('BAT-LITHIO') != -1) & (cap.Y == year)
                idx_sto = (sto.R == BalmArea) & (sto.Tech == 'INTRASEASONAL-ELECT-STORAGE') & (sto.G.str.find('BAT-LITHIO') != -1) & (sto.Y == year)
                power_cap += cap.loc[idx_cap, 'Value'].sum() * 1e3 # MW unloading capacity 
                capex += get_capex(sto, idx_sto, GDATA, ANNUITYCG)
            
            if power_cap > 1e-6:
                print('%s Li-Ion (Daily) Energy Capacity: <= %d MWh'%(area, power_cap*24))
            # Check GDATA, charge and discharge power capacities are the same    
            # GDATA[(GDATA.G.str.find('BAT-LITHIO-PEAK') != -1) & ((GDATA.Par == 'GDSTOHUNLD') | (GDATA.Par == 'GDSTOHLOAD'))]
            
            ### Daily Energy Capacity
            with open(wk_dir + ant_study + '/input/bindingconstraints/battery_energylimit_%s.txt'%area.lower(), 'w') as f:
                for k in range(366):
                    f.write(str(int(power_cap*24)) + '\t0\t0\n')
            
            ### 'Pumping' Capacity (Charge)
            create_transmission_input(wk_dir, ant_study, '0_battery_pmp', area.lower(), [0, power_cap], 0)

            ### 'Turb' Capacity (Discharge)
            create_transmission_input(wk_dir, ant_study, '0_battery_turb', area.lower(), [power_cap, 0], 0)

        # Save technoeconomic data to file
        fAntTechno.loc[(i, year, area, 'battery'), 'OPEX'] = 0
        fAntTechno.loc[(i, year, area, 'battery'), 'CAPEX'] = capex
        fAntTechno.loc[(i, year, area, 'battery'), 'Energy Capacity'] = power_cap*24 
        fAntTechno.loc[(i, year, area, 'battery'), 'Power Capacity'] = power_cap 
            

    ### 3.3 Hydrogen Storage
    for area in A2B_regi_h2.keys():
        energy_cap = 0
        power_cap = 0
        energy_cap_tank = 0
        power_cap_tank = 0
        capex_cav = 0
        capex_tank = 0
        for BalmArea in A2B_regi_h2[area]:
            # Get weight from amount of corresponding areas in Balmorel
            weight = B2A_DH2_weights[BalmArea][area]
            
            ### Find Caverns
            idx_sto = (sto.R == BalmArea) & (sto.Tech == 'H2-STORAGE') & (sto.G.str.find('CAVERN') != -1) & (sto.Y == year)
            energy_cap += weight*sto.loc[idx_sto, 'Value'].sum() * 1e3 # MWh
            power_cap += weight*cap.loc[(cap.R == BalmArea) & (cap.Tech == 'H2-STORAGE') & (cap.G.str.find('CAVERN') != -1) & (cap.Y == year), 'Value'].sum() * 1e3 # MW unloading capacity
            capex_cav += get_capex(sto, idx_sto, GDATA, ANNUITYCG)
        
            ### Find Tanks
            idx_sto = (sto.R == BalmArea) & (sto.Tech == 'H2-STORAGE') & (sto.G.str.find('TNKC') != -1) & (sto.Y == year)
            energy_cap_tank += weight*sto.loc[idx_sto, 'Value'].sum() * 1e3 # MWh
            power_cap_tank += weight*cap.loc[(cap.R == BalmArea) & (cap.Tech == 'H2-STORAGE') & (cap.G.str.find('TNKC') != -1) & (cap.Y == year), 'Value'].sum() * 1e3 # MW unloading capacity used
            capex_tank += get_capex(sto, idx_sto, GDATA, ANNUITYCG)
    
        # Check GDATA, charge and discharge power capacities are the same    
        # GDATA[(GDATA.G.str.find('TNKC') != -1) & ((GDATA.Par == 'GDSTOHUNLD') | (GDATA.Par == 'GDSTOHLOAD'))]
        
        # Save tanks
        if energy_cap_tank > 1e-6:
            print('\nArea %s'%area)
            print('H2 Tank Energy Capacity: \t%d MWh'%(energy_cap_tank))
            print('H2 Tank Power Capacity: \t%d MW'%(power_cap_tank))
        
        # Save capacity in .ini
        h2_tank_list = h2_tank_list +\
        """[%s]\nname = %s\nunitcount = 1\nnominalcapacity = %d\n\n[%s]\nname = %s\nunitcount = 1\nnominalcapacity = %d\n\n"""%(area.lower().replace('z_h2_c3_', ''), area.lower().replace('z_h2_c3_', ''), int(round(energy_cap_tank)),
                                                                                                area.lower().replace('z_h2_c3_', '') + ' 2', area.lower().replace('z_h2_c3_', '') + ' 2', int(round(energy_cap_tank)))
        # Save energy capacity timeseries
        temp = pd.Series(np.ones(8760) * energy_cap_tank).astype(int)
        temp.to_csv(wk_dir + ant_study + '/input/thermal/series/0_h2tank_capa/%s/series.txt'%(area.lower().replace('z_h2_c3_', '') + ' 2'), sep='\t', header=False, index=False)
        temp.to_csv(wk_dir + ant_study + '/input/thermal/series/0_h2tank_capa/%s/series.txt'%(area.lower().replace('z_h2_c3_', '')), sep='\t', header=False, index=False)
        
        # Save power capacity timeseries
        create_transmission_input(wk_dir, ant_study, '0_h2tank_pmp', area.lower(), [0, power_cap_tank], 0)
        create_transmission_input(wk_dir, ant_study, '0_h2tank_turb', area.lower(), [power_cap_tank, 0], 0)
        
        # Save technoeconomic data to file
        fAntTechno.loc[(i, year, area, 'h2 tank'), 'OPEX'] = 0
        fAntTechno.loc[(i, year, area, 'h2 tank'), 'CAPEX'] = capex_tank
        fAntTechno.loc[(i, year, area, 'h2 tank'), 'Power Capacity'] = power_cap_tank
        fAntTechno.loc[(i, year, area, 'h2 tank'), 'Energy Capacity'] = energy_cap_tank 
        
        print('\nArea %s'%area)
        print('H2 Cavern Energy Capacity: \t%d MWh'%(energy_cap))
        print('H2 Cavern Power Capacity: \t%d MW'%(power_cap))
        if energy_cap > 1e-6:
            # These options can just be set
            Hydro.set('reservoir', area.lower(), 'true')
            Hydro.set('reservoir capacity', area.lower(), str(round(energy_cap)))
            
            # These options could dissappear in 2020, as they are only used for salt-cavern storage modelling
            try:
                Hydro.set('use water', area.lower(), 'false')
            except configparser.NoSectionError:
                pass
                
            try:
                Hydro.set('use heuristic', area.lower(), 'true')
            except configparser.NoSectionError:
                Hydro.add_section('use heuristic')
                Hydro.set('use heuristic', area.lower(), 'true')
                
        else:
            Hydro.set('reservoir', area.lower(), 'false')
            try:
                Hydro.remove_option('reservoir capacity', area.lower())
                Hydro.remove_option('use heuristic', area.lower())
                Hydro.remove_option('use water', area.lower())
            except configparser.NoSectionError:
                pass
        with open(wk_dir + ant_study + '/input/hydro/common/capacity/maxpower_%s.txt'%area.lower(), 'w') as f:
            for k in range(8760):
                f.write(str(int(power_cap))+'\t24\t'+str(int(power_cap))+'\t24\n')

        # Save technoeconomic data to file
        fAntTechno.loc[(i, year, area, 'h2 cavern'), 'OPEX'] = 0
        fAntTechno.loc[(i, year, area, 'h2 cavern'), 'CAPEX'] = capex_cav
        fAntTechno.loc[(i, year, area, 'h2 cavern'), 'Power Capacity'] = power_cap 
        fAntTechno.loc[(i, year, area, 'h2 cavern'), 'Energy Capacity'] = energy_cap 
        
    # Check GDATA, charge and discharge power capacities are the same    
    # GDATA[(GDATA.G.str.find('CAVERN') != -1) & ((GDATA.Par == 'GDSTOHUNLD') | (GDATA.Par == 'GDSTOHLOAD'))]

    # Save H2 salt-cavern energy capacity  
    with open(wk_dir + ant_study + '/input/hydro/hydro.ini', 'w') as f:
        Hydro.write(f)
            
    # Save H2 tank capacity  
    with open(wk_dir + ant_study + '/input/thermal/clusters/0_h2tank_capa/list.ini', 'w') as f:
        f.write(h2_tank_list)
    

    
    #%% ------------------------------- ###
    ###    4. Transmission Capacities   ###
    ### ------------------------------- ###

    print('\nTransmission capacities to Antares...\n')

    ### 4.1 Read All Links
    links = pd.read_csv(wk_dir + '/Pre-Processing/Data/Links.csv', sep=';') # PRODUCED BY HAND...

    ### 4.2 Read Balmorel Results
    trans = symbol_to_df(db, "X_CAP_YCR", 
                        ['Y', 'C', 'RE', 'RI', 'Var', 'Units', 'Value'])
    trans.loc[:, 'Commodity'] = 'ELECTRICITY'
    transH2 = symbol_to_df(db, "XH2_CAP_YCR", 
                        ['Y', 'C', 'RE', 'RI', 'Var', 'Units', 'Value'])
    if len(transH2) > 0:
        transH2.loc[:, 'Commodity'] = 'HYDROGEN'
        trans = trans._append(transH2, ignore_index=True) # Append H2

    print('Paranthesis is capacity in opposite direction')
    ### 4.3 Go through all links
    for n,row in links.iterrows():
        
        # Filter through capacities
        idx = (trans.Commodity == row.carrier.upper()) & (trans.Y == year)

        # Choose correct dictionary
        mapper = A2B_regi if row.carrier == 'electricity' else A2B_regi_h2
        capsfunc = str.upper if row.carrier == 'electricity' else str.lower
        
        ## Capacity from
        # Find areas from
        if row.comment == 'from_aggregate':
            idx2 = trans.RE != trans.RE    
            for exp in mapper[capsfunc(row['from'])]:
                idx2 = idx2 | (trans.RE == exp)
        else:
            # Harmonised spatial resolution
            idx2 = trans.RE == mapper[capsfunc(row['from'])][0]
        
        # Find areas toE
        if row.comment == 'to_aggregate':
            idx3 = trans.RI != trans.RI
            for imp in mapper[capsfunc(row['to'])]:
                idx3 = idx3 | (trans.RI == imp)
        else:
            # Harmonised spatial resolution
            idx3 = trans.RI == mapper[capsfunc(row['to'])][0]
        
        # Sum capacity
        trans_cap_from = trans.loc[idx & idx2 & idx3, 'Value'].sum() * 1e3 # MW

        ## Capacity to
        # Find areas from
        if row.comment == 'from_aggregate':
            idx2 = trans.RE != trans.RE    
            for exp in mapper[capsfunc(row['to'])]:
                idx2 = idx2 | (trans.RE == exp)
        else:
            # Harmonised spatial resolution
            idx2 = trans.RE == mapper[capsfunc(row['to'])][0]
        
        # Find areas toE
        if row.comment == 'to_aggregate':
            idx3 = trans.RI != trans.RI
            for imp in mapper[capsfunc(row['from'])]:
                idx3 = idx3 | (trans.RI == imp)
        else:
            # Harmonised spatial resolution
            idx3 = trans.RI == mapper[capsfunc(row['from'])][0]
        
        # Sum capacity
        trans_cap_to = trans.loc[idx & idx2 & idx3, 'Value'].sum() * 1e3 # MW
        print(row['from'], row['to'],trans_cap_from.astype(int), '(',trans_cap_to.astype(int), ') MW')


        # Save it 
        create_transmission_input(wk_dir, ant_study, row['from'], row['to'], [trans_cap_from, trans_cap_to], 0.01)

    #%% ------------------------------- ###
    ###           5. Demand             ###
    ### ------------------------------- ###

    ### NOT transferring electricity-to-heat electricity demand at the moment, 
    ### due to too hard heuristics. Will be investigated in another paper.
    ### Will need to model heating in Antares, or make looser demand assumption on
    ### electricity to heat

    print('Annual electricity demands to Antares...\n')

    ### 5.1 Load exogenous electricity demands from first iteration
    db0 = ws.add_database_from_gdx(wk_dir + "/Balmorel/%s/model/MainResults_%s.gdx"%(SC_folder, SC.replace('Iter%d'%i, 'Iter0')))
    dem_iter0 = symbol_to_df(db0, 'EL_DEMAND_YCR',
                    ['Y', 'C', 'R', 'Type', 'Unit', 'Value']) # First iteration is to get 'real' exogenous demand, not fictive
   

    ### 5.2 Go through regions
    for area in A2B_regi.keys():
        
        
        if not(area in ['ITCO']):
            # Load Antares demand
            ant_dem = pd.read_csv(wk_dir+ant_study + '/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None)
            
            # Plot demands
            # fig, ax = plt.subplots()
            # ant_dem.plot(ax=ax)
            # ax.set_title(area + ' - Before Balm Demands')
            # ax.legend(ncol=2, loc='center', bbox_to_anchor=(1.2, .5), title='Stochastic Year')
                
            ann_dem = 0 # Annual demand in Antares node    
            flex_dem = 0 # Annual flexible demand
            for BalmArea in A2B_regi[area]:

                # Get weight from amount of corresponding areas in Balmorel
                weight = B2A_DE_weights[BalmArea][area]
            
                # Filter area and PtX, sum all demands
                # NOTE: Maybe this should be divided into industry, datacenter and residential+other profiles! (flat industry + datacenter profile + Antares Profile)
                idx = (dem_iter0.Type == 'EXOGENOUS') & (dem_iter0.R == BalmArea) & (dem_iter0.Y == year) 
                print('Exogenous demand in %s, year %s'%(BalmArea, year), round(dem_iter0.loc[idx, 'Value'].sum()), 'TWh')
                
                # Increment demand and add distribution loss
                ann_dem += weight * dem_iter0.loc[idx, 'Value'].sum() / (1 - DISLOSSEL.loc[BalmArea, 'Value']) 
                
                print('Assigning to %s...'%(area))
                
                
                # Flexible demand
                if UseFlexibleDemand:
                    idx = (dem_iter0.Type == 'ENDO_FLEXDEM') & (dem_iter0.R == BalmArea) & (dem_iter0.Y == year) 
                    flex_dem += weight * dem_iter0.loc[idx, 'Value'].sum()

            print('Resulting annual electricity demand in %s = %0.2f TWh\n'%(area, ann_dem))

            # Save
            # NOTE: Maybe do as noted above instead, so: ant_dem * (DE from rese + other) + DE_industry/8760 + DE_datacenter/8760
            ant_dem = np.round(ant_dem * ann_dem * 1e6).astype(int) # To timeseries
            ant_dem.to_csv(wk_dir + ant_study + '/input/load/series/load_%s.txt'%(area.lower()), sep='\t', header=None, index=None)

            if UseFlexibleDemand:
                print('Resulting flexible annual electricity demand in %s = %0.2f TWh\n'%(area, flex_dem))
                
                # Create transmission capacity using charging profile assumption
                create_transmission_input(wk_dir, ant_study, area.lower(), '0_flexdem', [0, 0], 0) # Assuming flat capacity! (i.e. non-flexible)
                with open('Antares/input/links/0_flexdem/capacities/%s_indirect.txt'%area.lower(), 'w') as f:
                    # Convert normalised profile to charging profile (100 kWh consumed pr week = annual demand, 1.5 kW charger assumed)
                    f.write(pd.Series(flexdem_profile*flex_dem*1e6/100/52*3).to_string(index=False, header=False))
                
                # Create binding constraint energy demand
                with open('Antares/input/bindingconstraints/flexdem_%s.txt'%area.lower(), 'w') as f:
                    for k in range(366):
                        f.write('0\t'+str(int(round(flex_dem/52/7*1e6))) +'\t0\n')
            else:
                create_transmission_input(wk_dir, ant_study, area.lower(), '0_flexdem', [0, 0], 0) # Assuming flat capacity! (i.e. non-flexible)
                
                # Create binding constraint energy demand
                with open('Antares/input/bindingconstraints/flexdem_%s.txt'%area.lower(), 'w') as f:
                    for k in range(366):
                        f.write('0\t0\t0\n')
            
            # Plot the new demands
            # fig, ax = plt.subplots()
            # ant_dem.plot(ax=ax)
            # ax.set_title(area + ' - After Balm Demands')
            # ax.annotate('Total Dem: %d TWh'%(ant_dem.sum().mean()/1e6), xy=(.75, .75))
            # ax.legend(ncol=2, loc='center', bbox_to_anchor=(1.2, .5), title='Stochastic Year')
            # fig.savefig('MetaResults/' + '_'.join((SC, 'AntExoElDem', area)) + '.png', bbox_inches='tight')
            
    #%% 5.3 Load exogenous hydrogen demands from first iteration
    db0 = ws.add_database_from_gdx(wk_dir + "/Balmorel/%s/model/MainResults_%s.gdx"%(SC_folder, SC.replace('Iter%d'%i, 'Iter0')))
    dem_iter0 = symbol_to_df(db0, 'H2_DEMAND_YCR',
                    ['Y', 'C', 'R', 'Type', 'Unit', 'Value']) # First iteration is to get 'real' exogenous demand, not fictive

    print('Annual hydrogen demands to Antares...\n')

    ### 5.4 Go through regions
    for area in A2B_regi_h2.keys():
            
        # Load Antares demand
        # ant_dem = pd.read_csv(wk_dir+ant_study + '/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None)
        
        # Plot demands
        # fig, ax = plt.subplots()
        # ant_dem.plot(ax=ax)
        # ax.set_title(area + ' - Before Balm Demands')
        # ax.legend(ncol=2, loc='center', bbox_to_anchor=(1.2, .5), title='Stochastic Year')
        # ant_dem = pd.read_csv(wk_dir+ant_study + '/input/load/series/load_%s_normalised-data.txt'%(area), sep='\t', header=None)
        
        ann_dem = 0 # Annual demand in Antares node    
        for BalmArea in A2B_regi_h2[area]:

            # Get weight 
            weigth = B2A_DH2_weights[BalmArea][area] # Fix this differentiation between capacity and demand... 
            
            
            # Filter area and PtX, sum all demands
            idx = (dem_iter0.Type == 'EXOGENOUS') & (dem_iter0.R == BalmArea) & (dem_iter0.Y == year) 
            print('Exogenous demand in %s, year %s'%(BalmArea, year), round(dem_iter0.loc[idx, 'Value'].sum()), 'TWh')
            
            # Increment factor
            ann_dem += weight * dem_iter0.loc[idx, 'Value'].sum() / (1 - DISLOSSH2.loc[BalmArea, 'Value'])
            
            print('Assigning to %s...'%(area))

        print('Resulting annual hydrogen demand in %s = %0.2f TWh\n'%(area, ann_dem))

        # Save antares hydrogen demand profile
        # ant_dem = np.round(ant_dem * ann_dem * 1e6).astype(int) # To timeseries
        # ant_dem.to_csv(wk_dir + ant_study + '/input/load/series/load_%s.txt'%(area.lower()), sep='\t', header=None, index=None)
        # Save flat profile
        pd.Series(np.ones(8760)*ann_dem/8760*1e6).to_csv(wk_dir + ant_study + '/input/load/series/load_%s.txt'%(area.lower()), sep='\t', header=None, index=None)

        # Plot the new demands
        # fig, ax = plt.subplots()
        # ant_dem.plot(ax=ax)
        # ax.set_title(area + ' - After Balm Demands')
        # ax.annotate('Total Dem: %d TWh'%(ant_dem.sum().mean()/1e6), xy=(.75, .75))
        # ax.legend(ncol=2, loc='center', bbox_to_anchor=(1.2, .5), title='Stochastic Year')
        # fig.savefig('MetaResults/' + '_'.join((SC, 'AntExoElDem', area)) + '.png', bbox_inches='tight')
        
    #%% ------------------------------- ###
    ### 6. Weekly Resource Constraints  ###         
    ### ------------------------------- ###

    ### 6.1 Calculates residual demand profiles (electricity load - VRE profile) 
    ###     and uses this normalised series to factor on annual resource availability 

    GMAXF = symbol_to_df(ALLENDOFMODEL, 'IGMAXF', ['Y', 'CRA', 'F', 'Value'])
    GMAXFS = symbol_to_df(ALLENDOFMODEL, 'GMAXFS', ['Y', 'CRA', 'F', 'S', 'Value'])
    CCCRRR = pd.DataFrame([rec.keys for rec in ALLENDOFMODEL['CCCRRR']], columns=['C', 'R']).groupby(by=['C']).aggregate({'R' : ', '.join})
    CCCRRR['Done?'] = False

    # Load the stochastic years used 
    with open('Antares/settings/generaldata.ini', 'r') as f:
        Config = ''.join(f.readlines())    
    stochyears = [int(stochyear.split('\n')[0].replace(' ', '').replace('+=','')) for stochyear in Config.split('playlist_year')[1:]]


    Config = configparser.ConfigParser()
    for area in A2B_regi.keys():
            
        if not(area in ['ITCO']):
            Config.read('Antares/input/renewables/clusters/%s/list.ini'%area.lower())
                
            load = pd.read_csv(wk_dir + '/Antares/input/load/series/load_%s.txt'%(area.lower()), header=None, index_col=None, delimiter='\t') 
            load = load.loc[:, stochyears].mean(axis=1)

            for VRE in ['wind', 'solar']:
                
                # Production series
                try:
                    f = pd.read_table('Antares/input/renewables/series/{area}/{area}_{VRE}_0/series.txt'.format(area=area.lower(), VRE=VRE), header=None)
                
                    # Get capacity input
                    vrecap = Config.getfloat('%s_%s_0'%(area, VRE), 'nominalcapacity')
                    
                    # Calculate mean absolute production profile through stochastic years
                    vre = f.loc[:, stochyears].mean(axis=1)*vrecap
                    load = load - vre # Residual load
                    
                except EmptyDataError:
                    pass
                    # print('No profile for %s in %s'%(VRE, area))


            # Plot Residual LDC
            # fig, ax = plt.subplots()
            # x, y = doLDC(resload, 100)
            # ax.plot(np.cumsum(x), y)
            
            # Sum weekly residual loads
            resload_week = load.rolling(window=168).sum()
            resload_week = resload_week[167::168] # Only snapshots in the end of each week
            resload_week.index = [i for i in range(1, 53)]
            resload_week = resload_week - resload_week.min() # Zero availability in best month
            resload_week = resload_week / resload_week.sum() # Normalise energy
            
            # All fuels, except municipal waste
            fuels = [fuel for fuel in pd.DataFrame(BalmTechs).index.to_list() if fuel != 'MUNIWASTE' and fuel != 'HYDROGEN' and fuel != 'NUCLEAR']


            Config.clear()
            # Read the binding constraint
            Config.read('Antares/input/bindingconstraints/bindingconstraints.ini')

            R = A2B_regi[area][0] # Just any region - regions are all within a country
            country = CCCRRR[CCCRRR.R.str.find(R) != -1].index[0] 
            
            ### 6.2 Set Efficiency of Generators in Area, if it has a capacity
            for fuel in fuels:
                for tech in BalmTechs.keys():
                    
                    # Calculate average efficiency of all G types
                    N_reg = 0
                    eff = 0
                    for BalmArea in A2B_regi[area]:
                        idx_cap = (cap['Commodity'] == 'ELECTRICITY') & (cap.R == BalmArea) & (cap.F == fuel) & (cap.Tech == tech) & (cap.Y == year)
                        if cap.loc[idx_cap, 'Value'].sum()*1000 > 1e-6:   
                            eff += get_efficiency(cap, idx_cap, GDATA)
                            N_reg += 1
                    
                    if N_reg > 0:
                        eff = eff / N_reg

                        generator = '{reg}.{tech}_{fuel}'.format(reg=area.lower(), tech=tech.lower(), fuel=fuel.lower())
                        for section in Config.sections():
                            if generator in Config.options(section):
                                # print('%s is in section %s'%(generator, section))
                                # print('Setting %s to efficiency %0.2f'%(generator, eff))
                                Config.set(section, generator, str(round(1/eff, 2)))
            
                ### 6.3 Calculate Weekly Fuel Limits for all fuels but Muniwaste, if not already done
                if not(CCCRRR.loc[country, 'Done?']):
                    try:
                        pot = GMAXF.loc[(GMAXF.F == fuel) & (GMAXF.CRA == country) & (GMAXF.Y == year), 'Value'].values[0]/3.6 # To MWh
                    except IndexError:
                        pot = 0
                    
                    # Write it
                    with open('Antares/input/bindingconstraints/%sres_%s.txt'%(fuel.lower(), country.lower()), 'w') as f:
                        for week_distribution in resload_week:
                            for i in range(7):
                                
                                if pot > 0:
                                    # If there is a potential specified
                                    f.write('%0.2f\t0\t0\n'%(week_distribution*pot/7))
                                else:
                                    # If there is no potential specified, put a very high limit
                                    f.write('%0.2f\t0\t0\n'%(1e12))

                        # The last week
                        if pot > 0:
                            for i in range(2):
                                f.write('%0.2f\t0\t0\n'%(week_distribution*pot/7))
                        else:                
                            for i in range(2):
                                f.write('%0.2f\t0\t0\n'%(1e12))
                                
            ### 6.4 Input weekly fuel limit for muniwaste in region
            ## Calculate average efficiency of all G types
            N_reg = 0
            eff = 0
            for BalmArea in A2B_regi[area]:
                idx_cap = (cap['Commodity'] == 'ELECTRICITY') & (cap.R == BalmArea) & (cap.F == 'MUNIWASTE') & (cap.Tech == tech) & (cap.Y == year)
                if cap.loc[idx_cap, 'Value'].sum()*1000 > 1e-6:   
                    eff += get_efficiency(cap, idx_cap, GDATA)
                    N_reg += 1
            
            if N_reg > 0:
                eff = eff / N_reg

                generator = '{reg}.{tech}_muniwaste'.format(reg=area.lower(), tech=tech.lower())
                for section in Config.sections():
                    if generator in Config.options(section):
                        # print('%s is in section %s'%(generator, section))
                        # print('Setting %s to efficiency %0.2f'%(generator, eff))
                        Config.set(section, generator, str(round(1/eff, 2)))
            
            # Save configfile
            with open('Antares/input/bindingconstraints/bindingconstraints.ini', 'w') as configfile:
                Config.write(configfile)
            Config.clear()
                
            
            ## Write potential
            idx = (GMAXFS.F == 'MUNIWASTE') & (GMAXFS.Y == year) 
            idx2 = GMAXFS.CRA != GMAXFS.CRA
            
            # Aggregate, in case Balmorel is higher resolved
            weight = 0
            for BalmArea in A2B_regi[area]:
                idx2 = idx2 | (GMAXFS.CRA == BalmArea)
                        
                # Disaggregate, if Antares is higher resolved
                weight += B2A_DE_weights[BalmArea][area] / len(A2B_regi[area])
            # print('%s weight: %0.2f'%(area, weight))
                
            pot = GMAXFS.loc[idx & idx2].groupby(by=['S']).aggregate({'Value' : "sum"})
            with open('Antares/input/bindingconstraints/muniwasteres_%s.txt'%(area.lower()), 'w') as f:
                for week in pot.index:
                    pot0 = pot.loc[week, 'Value']/3.6 * weight # To MWh
                    for i in range(7):
                        if pot0 > 0:
                            # If there is a potential specified
                            f.write('%0.2f\t0\t0\n'%(pot0/7))
                        else:
                            # If there is no potential specified, put a very high limit
                            f.write('%0.2f\t0\t0\n'%(1e12))

                # The last week
                if pot0 > 0:
                    for i in range(2):
                        f.write('%0.2f\t0\t0\n'%(pot0/7))
                else:                
                    for i in range(2):
                        f.write('%0.2f\t0\t0\n'%(1e12))
                
            # Done. Don't have to do this for the next region in the same country
            CCCRRR.loc[country, 'Done?'] = True

    #%% ------------------------------- ###
    ###  7. Pseudo-Benders Constraints  ###         
    ### ------------------------------- ###

    ### 7.1 Read Binding Constraints 
    Config = configparser.ConfigParser()
    Config.read('Antares/input/bindingconstraints/bindingconstraints.ini')
    CapConfig = configparser.ConfigParser()

    # Placeholder for searching through all sections (more efficient later)
    relevant_sections = pd.DataFrame({sec : True for sec in Config.sections()}, index=['Relevant?']).T
    for sec in relevant_sections.index:
        # Disregard irrelevant sections
        if not('PseudoBenders' in Config.get(sec, 'name')):
            relevant_sections.loc[sec] = False
    
    
    if UsePseudoBenders:
        for area in CapacityInRegion.keys():
            
            # Don't read transformer region
            if not(area in ['ITCO']):
                
                # Get configfile
                CapConfig.read('Antares/input/thermal/clusters/%s/list.ini'%area)
                
                # Get technologies with capacities
                for tech in CapacityInRegion[area].keys():
                    
                    # Get capacity
                    tech_cap = CapConfig.getfloat(tech, 'nominalcapacity')
                    # And now set it to a very high number, so it's not binding
                    CapConfig.set(tech, 'nominalcapacity', str(tech_cap + 50000))
                        
                    found_tech = False # Does a constraint exist for this technology?
                    
                    # Search through relevant sections
                    if len(relevant_sections[relevant_sections.values]) > 0:
                        for sec in relevant_sections[relevant_sections.values].index:
                        
                            if tech.upper() in Config.get(sec, 'name'):
                                # Enable PseudoBenders-Tech-Pmax constraint
                                Config.set(sec, 'enabled', 'true')

                                # The constraint existed for this tech
                                found_tech = True

                    if not(found_tech):                    
                        # Create PseudoBenders-Tech-PMax Constraint if it didn't exist
                        new_sec = str(relevant_sections.index.astype(int).max() + 1)
                        
                        Config.add_section(new_sec)
                        Config.set(new_sec, 'name', '_'.join(('PseudoBendersPmax', area, tech.upper()))) 
                        Config.set(new_sec, 'id', '_'.join(('PseudoBendersPmax', area, tech)).lower()) 
                        Config.set(new_sec, 'enabled', 'true')
                        Config.set(new_sec, 'type', 'hourly')
                        Config.set(new_sec, 'operator', 'less')
                        Config.set(new_sec, 'filter-year-by-year', '')
                        Config.set(new_sec, 'filter-synthesis', 'hourly')
                        Config.set(new_sec, '.'.join((area.lower(), tech)), '1')
                                            
                        # Append to relevant sections, but set to false since we dealt with it
                        relevant_sections.loc[new_sec, 'Relevant?'] = False
                        
                    # Write upper bound timeseries
                    with open('Antares/input/bindingconstraints/%s.txt'%('_'.join(('PseudoBendersPmax', area, tech)).lower()), 'w') as f:
                        for h in range(8784):
                            f.write('%0.2f\t0\t0\n'%(tech_cap))

                
                # Save capacity configfile
                with open('Antares/input/thermal/clusters/%s/list.ini'%(area.lower()), 'w') as configfile:
                    CapConfig.write(configfile)
                CapConfig.clear()
                

    else:
        # Search through relevant sections
        if len(relevant_sections[relevant_sections.values]) > 0:
            for sec in relevant_sections[relevant_sections.values].index:

                # Disable PseudoBenders-Tech-Pmax constraint
                Config.set(sec, 'enabled', 'false')

    # Save configfile
    with open('Antares/input/bindingconstraints/bindingconstraints.ini', 'w') as configfile:
        Config.write(configfile)

    # Save Antares technoeconomic data
    with open(wk_dir + '/Workflow/OverallResults/%s_AT.pkl'%SC_name, 'wb') as f:
        pickle.dump(fAntTechno, f)

    print('\n|--------------------------------------------------|')   
    print('              END OF PERI-PROCESSING')
    print('|--------------------------------------------------|\n')  

    # Set periprocessing_finished to true (will be set to true after peri-processing finishes)
    with open('Workflow/MetaResults/periprocessing_finished.txt', 'w') as f:
        f.write('True')

except Exception as e:
    # If there's an error, we still want to signal that we are finished occupying the Antares compilation
    with open('Workflow/MetaResults/periprocessing_finished.txt', 'w') as f:
        f.write('True')
     
    # Raise the error
    raise e