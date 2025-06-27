
from Workflow.Functions.GeneralHelperFunctions import AntaresOutput
from pybalmorel import MainResults

def main(antares_result: str, 
         balmorel_result: str,
         gams_system_directory: str):

    # Load Results
    antout = AntaresOutput(antares_result)
    mr = MainResults(balmorel_result, paths='Balmorel/base/model', system_directory=gams_system_directory )

    ENDO_EL = {'HEAT' : 0,
            'HYDROGEN' : 0}
    for area in ['DE', 'FR', 'ES']:
        for commodity in ['HEAT', 'HYDROGEN']:
            res = antout.load_link_results([area, '_'.join([area, commodity])], temporal='annual')['FLOW LIN.']   
            ENDO_EL[commodity] += float(res.sum()) / 1e6
                        
    # Compare to Balmorel
    df = mr.get_result('EL_DEMAND_YCR')

    print('\n', '='*6, 'Antares', '='*6)
    print('Heat: ', ENDO_EL['HEAT'], 'TWh')
    print('Hydrogen: ', ENDO_EL['HYDROGEN'], 'TWh')
    print('\n', '='*6, 'Balmorel', '='*6)
    print(df.query('Category in ["ENDOGENOUS_ELECT2HEAT", "ENDO_H2"]').pivot_table(index='Category', values='Value', aggfunc='sum').to_string(), '\n')

if __name__ == '__main__':
    antares_result = '20250619-1638eco-baf_test_new_noh2_iter0_y-2050'
    balmorel_result = 'MainResults_baf_test_new_noH2_Iter0.gdx'
    gams_system_directory = '/opt/gams/48.5'
    main(antares_result, balmorel_result, gams_system_directory)