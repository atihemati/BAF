import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% ------------------------------- ###
###          1. Plotting            ###
### ------------------------------- ###

def fplot(ax):
    ax.minorticks_on()
    ax.grid(which='minor', linestyle='--', alpha=.3)
    ax.grid(which='major', linestyle='--', alpha=.7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
def newplot(nrows=1, ncols=1, figsize=(5,4), dpi=140, fc='none'):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    fig.set_facecolor(fc)
    if type(ax) == np.ndarray:
        for a in np.asarray(ax).reshape(-1):
            a.set_facecolor(fc)
            fplot(a)
    else:
        ax.set_facecolor(fc)
        fplot(ax)
    
    return fig, ax


def stacked_bar(df, axis, legend, ax='None', args='None'):
  """
  Make a stacked bar plot

  Args:
      df (_type_): _description_
      axis (_type_): _description_
      legend (_type_): _description_
  """
  if type(axis) != list:
      axis = [axis]
  if type(legend) != list:
      legend = [legend]
  
  if args == 'None':
      args = {}
  
  if ax != 'None': 
    # df.groupby([r for r in axis + legend]).aggregate(np.sum).unstack().plot(y='Value', x=axis, kind='bar', stacked=True, ax=ax, **args)
    df.pivot_table(columns=legend, index=axis).plot(kind='bar', stacked=True, ax=ax, **args)
  else:
    # df.groupby([r for r in axis + legend]).aggregate(np.sum).unstack().plot(y='Value', x=axis, kind='bar', stacked=True, **args)
    df.pivot_table(columns=legend, index=axis).plot(kind='bar', stacked=True, **args)


def set_style(style):
    if style == 'report':
        plt.style.use('default')
        fc = 'white'
        plotly_theme = 'plotly'
    elif style == 'ppt':
        plt.style.use('dark_background')
        fc = 'none'
        plotly_theme = 'plotly_dark'

    # 0.1 Custom colormap  
    # Importing a custom colormap, adequate for communicating to people with colour-vision deficiency or colour-blindness
    # Read more here: 
    # Crameri, Fabio, Grace E. Shephard, and Philip J. Heron. “The Misuse of Colour in Science Communication.” Nature Communications 11, no. 1 (October 28, 2020): 5444. https://doi.org/10.1038/s41467-020-19160-7.
    #
    # and read more on appropriate colours on the website:
    # https://www.fabiocrameri.ch/colourmaps/
    #
    # to install:
    # pip install cmcrameri
    try: 
        import cmcrameri.cm as cmc
        print('Using colormap adequate for color-deficient and -blind communication')
        
        if style == 'report':
            # color_list = [cmc.batlow(i) for i in [0, 126, 256, 189, 80, 150]] # Custom discrete colours
            color_list = [cmc.batlowWS(i) for i in range(1, 256)] # batlowS is a categorical
        elif style == 'ppt':
            color_list = [cmc.batlowKS(i) for i in range(1, 256)] # batlowS is a categorical
            
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
        plt.rcParams['image.cmap'] = 'cmc.batlow'
    except ModuleNotFoundError:
        print('Using default colormap')
        
    return fc, plotly_theme

#%% ------------------------------- ###
###             2. Data             ###
### ------------------------------- ###

def nested_dict_to_df(nested_dict):
    """Turns a 4D dictionary into a dataframe with multiindex

    Args:
        nested_dict (dict): A 4D nested dictionary (on all levels!)

    Returns:
        df: A dataframe with multiindex
    """
    df = pd.DataFrame.from_dict({(i, j, k, l): nested_dict[i][j][k][l] 
                             for i in nested_dict.keys() 
                             for j in nested_dict[i].keys() 
                             for k in nested_dict[i][j].keys() 
                             for l in nested_dict[i][j][k].keys()},
                            orient='index')
    mux = pd.MultiIndex.from_tuples(df.index)
    df.index = mux
    df.columns = ['Value']
    return df

