#%% ------------------------------- ###
###       0. Script Settings        ###
### ------------------------------- ###

import shutil
import os
import pandas as pd

#%% ------------------------------- ###
###     1. Delete Antares Output    ###
### ------------------------------- ###

### 1.0 Deleting Files
p = '../Antares/output/' 

l = pd.Series(os.listdir(p))

# Get run from specific year
idx = (l.str.find('eco') != -1)
ant_res = l[idx] # Should be the most recent

for res in ant_res:
    try:
        print('\nDeleting MC year outputs from run\n%s\n'%(res))
        shutil.rmtree(p + res + '/economy/mc-ind')
        print('Deletion done\n')
    except FileNotFoundError:
        print('\nMC year outputs already deleted from\n%s\n'%res)
