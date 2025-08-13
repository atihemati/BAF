"""
TITLE

Description

Created on 13.08.2025
@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###             0. CLI              ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import click
import os
from neural_network import pretrain, train

@click.command()
@click.argument('scenario', type=str, required=True)
@click.option('--dark-style', is_flag=True, required=False, help='Dark plot style')
@click.option('--plot-ext', type=str, default='.pdf', required=False, help='The extension of the plot, defaults to ".pdf"')
@click.pass_context
def CLI(ctx, scenario: str, dark_style: bool, plot_ext: str):
    """
    Description of the CLI
    """
    
    # Set global style of plot
    if dark_style:
        plt.style.use('dark_background')
        fc = 'none'
    else:
        fc = 'white'

    # Store global options in the context object
    ctx.ensure_object(dict)
    ctx.obj['fc'] = fc
    ctx.obj['plot_ext'] = plot_ext

    epoch = 0    
    model = pretrain(5)
    os.chdir('Balmorel')

    while epoch < 100:
        for runtype in ['capacity', 'dispatch', 'operun']:
            
            os.system('rm operun/data/*.inc')
            
            if runtype == 'capacity':
                os.system('cp -f operun/capexp_data/*.inc operun/data')
            else:
                os.system('cp -f operun/S.inc operun/data')
                os.system(f'cp -f operun/T_{runtype}.inc operun/data/T.inc')
                os.system(f'mv operun/model/balopt_{runtype}.opt operun/model/balopt.opt')
                
            os.chdir('operun/model')
            os.system(f'gams Balmorel --scenario_name "{scenario}_{runtype}_E{epoch}"')

            os.chdir('../../')

            # Copy the simex folder
            # cp simex -r simex_$name

            if runtype != "capacity":
                # Rename balopt back
                os.system(f'mv "operun/model/balopt.opt" "operun/model/balopt_{runtype}.opt"')
        
        os.system(f'pixi run python analysis/analyse.py adequacy "{scenario}_operun" {epoch}')
        os.chdir('../')
        model = train(model, f"{scenario}_operun", epoch)
        os.chdir('Balmorel')
        
        epoch += 1


#%% ------------------------------- ###
###            2. Utils             ###
### ------------------------------- ###

@click.pass_context
def plot_style(ctx, fig, ax, name: str, legend: bool = True):
    
    ax.set_facecolor(ctx.obj['fc'])
    
    if legend:
        ax.legend(loc='center', bbox_to_anchor=(.5, 1.15), ncol=3)
    
    fig.savefig(name + ctx.obj['plot_ext'], bbox_inches='tight', transparent=True)
    
    return fig, ax

#%% ------------------------------- ###
###             3. Main             ###
### ------------------------------- ###
if __name__ == '__main__':
    CLI()
