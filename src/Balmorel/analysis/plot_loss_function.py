# plot loss function

import pandas as pd
import matplotlib.pyplot as plt
import click


@click.command()
@click.argument('scenario', type=str, required=True)
def CLI(scenario: str):
        
    df = pd.read_csv('Balmorel/analysis/output/' + scenario + '_adeq.csv')

    loss_values = df['ENS_TWh'] + df['LOLE_h']
    df['loss'] = loss_values
    
    df = df.pivot_table(index='epoch', values='loss', aggfunc='sum')
            
    fig, ax = plt.subplots()
    print(df)
    df.plot(ax=ax)
    plt.show()

if __name__ == '__main__':
    CLI()