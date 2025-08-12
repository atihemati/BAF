# plot loss function

import pandas as pd
import matplotlib.pyplot as plt
import os
import click


@click.command()
@click.argument('scenario', type=str, required=True)
def CLI(scenario: str):
    l = pd.Series(os.listdir('analysis/output'))
    idx = l.str.find(scenario) == 0
    
    files = l[idx].sort_values()
    
    df = pd.DataFrame()
    for file in files:
        
        if '_adeq.csv' in file:
            temp = pd.read_csv('analysis/output/' + file)

            loss_value = temp[['ENS_TWh', 'LOLE_h']].sum().sum()
            
            if 'E' in file:
                epoch = int(file.replace(scenario, '').split('_')[0].replace('E', ''))
            else:
                epoch = 0
            df = pd.concat((df, pd.DataFrame({'epoch' : [epoch],
                                              'loss' : [loss_value]})), 
                           ignore_index=True)
            
    fig, ax = plt.subplots()
    df = df.sort_values(by='epoch')
    print(df)
    df.plot(ax=ax, x='epoch', y='loss')
    plt.show()

if __name__ == '__main__':
    CLI()