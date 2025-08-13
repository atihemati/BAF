# plot loss function

import pandas as pd
import matplotlib.pyplot as plt
import click

@click.group()
def CLI():
    pass

@CLI.command()
@click.argument('scenario', type=str, required=True)
def sc(scenario: str):
        
    df = pd.read_csv('Balmorel/analysis/output/' + scenario + '_adeq.csv')

    loss_values = df['ENS_TWh'] + df['LOLE_h']
    df['loss'] = loss_values
    
    df = df.pivot_table(index='epoch', values='loss', aggfunc='sum')
            
    fig, ax = plt.subplots()
    print(df)
    df.plot(ax=ax)
    plt.show()

@CLI.command()
def all():

    df = pd.DataFrame()    
    for scenario in ['test2_operun', 'test2_onlydispatch_dispatch', 
                     'test2_OD_96_dispatch', 'test2_OD_48_4_dispatch',
                     'test2_OD_168_8_dispatch',
                     ]:
        temp = pd.read_csv('Balmorel/analysis/output/' + scenario + '_adeq.csv')

        loss_values = temp['ENS_TWh'] + temp['LOLE_h']
        temp['loss'] = loss_values
        temp['scenario'] = scenario
        
        temp = temp.pivot_table(index='epoch', values='loss', columns='scenario', aggfunc='sum')
        
        # df = df.join(temp)
        df = pd.concat([df, temp], axis=1)

    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_ylabel('Loss Value')
    ax.set_xlabel('Epoch')
    # ax.set_ylim([0, df.max().max()*1.2])
    plt.show()

if __name__ == '__main__':
    CLI()