
from Workflow.Functions.GeneralHelperFunctions import AntaresInput, AntaresOutput
from pybalmorel import MainResults

# Load Results
antin = AntaresInput()

## Cases to compare
case = 'fullyear+operun'
case = 'fullyear'
case = 'timeslices'
case = 'fullyear+lessprecision'
case = 'fullyear+lessprecision+capexfom'
if case == 'fullyear+operun':
    antout = AntaresOutput('20250613-1415eco-baf_test_new_fullyear_iter0_y-2050')
    mr = MainResults('MainResults_baf_test_new_fullyear_Iter0.gdx', paths='Balmorel/base/model',
                    system_directory='/opt/gams/48.5')
elif case == 'fullyear':
    antout = AntaresOutput('20250613-1318eco-baf_2ndtest_fullyear_iter0_y-2050')
    mr = MainResults('MainResults_baf_test_new_fullyear_Iter0.gdx', paths='Balmorel/base/model',
                    system_directory='/opt/gams/48.5')
elif case == 'fullyear+lessprecision':
    antout = AntaresOutput('20250613-1435eco-baf_fullyeartest_lessprecision')
    mr = MainResults('MainResults_baf_test_new_fullyear_Iter0.gdx', paths='Balmorel/base/model',
                    system_directory='/opt/gams/48.5')
elif case == 'fullyear+lessprecision+capexfom':
    antout = AntaresOutput('20250613-1559eco-baf_test_fullyear_capexfom')
    mr = MainResults('MainResults_baf_test_new_fullyear_Iter0.gdx', paths='Balmorel/base/model',
                    system_directory='/opt/gams/48.5')
elif case == 'timeslices':
    antout = AntaresOutput('20250613-0931eco-baf_2ndtest_iter0_y-2050')
    mr = MainResults('MainResults_baf_test_new_Iter0.gdx', paths='Balmorel/base/model',
                    system_directory='/opt/gams/48.5')

ENDO_EL = {'HEAT' : 0,
           'HYDROGEN' : 0}
for area in ['DE', 'FR', 'ES']:
    for commodity in ['HEAT', 'HYDROGEN']:
        res = antout.load_link_results([area, '_'.join([area, commodity])], temporal='annual')['FLOW LIN.'].sum() / 1e6
        print('%s in %s:\t%0.0f'%(commodity, area, res), 'TWh')   
        ENDO_EL[commodity] += res
        
        
# Compare to Balmorel
df = mr.get_result('EL_DEMAND_YCR')


print('Total HEAT\t', ENDO_EL['HEAT'])
print('Total HYDROGEN\t', ENDO_EL['HYDROGEN'])
print('Sum of both: ', ENDO_EL['HEAT'] + ENDO_EL['HYDROGEN'])

print(df.pivot_table(index='Category', values='Value', aggfunc='sum').loc[['ENDOGENOUS_ELECT2HEAT', 'ENDO_H2']].round().to_string())
print(df.pivot_table(index='Category', values='Value', aggfunc='sum').loc[['ENDOGENOUS_ELECT2HEAT', 'ENDO_H2']].sum().round())


## Profiles
plot_profiles = False
if plot_profiles:
    fig, ax = mr.plot_profile('heat', 2050)
    fig.savefig('heat_production.png')
    fig, ax = mr.plot_profile('hydrogen', 2050)
    fig.savefig('hydrogen_production.png')