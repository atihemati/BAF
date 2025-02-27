"""
Created on 01.05.2024

@author: Mathias Berg Rosendal, PhD Student at DTU Management (Energy Economics & Modelling)
"""
#%% ------------------------------- ###
###        0. Script Settings       ###
### ------------------------------- ###

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import platform

# Get operating system
OS = platform.platform().split('-')[0]


#%% ------------------------------- ###
###           1. Zip Data           ###
### ------------------------------- ###

branch = 'eur-system'
commit = 'b5dc7520'

l = os.listdir('Balmorel')

if OS == 'Linux':
    out = os.system('zip -r -q "BAF-Data_%s_%s.zip" "Balmorel/base/data"'%(branch, commit)) 
    out = os.system('zip -r -q "BAF-Data_%s_%s.zip" "Antares/input"'%(branch, commit)) 
    out = os.system('zip -r -q "BAF-Data_%s_%s.zip" "Pre-Processing/Data"'%(branch, commit)) 
    out = os.system('zip -r -q "BAF-Data_%s_%s.zip" "Pre-Processing/Output"'%(branch, commit)) 
    out = os.system('zip -r -q BAF-Data_%s_%s.zip Pre-Processing/Output Pre-Processing/Data Antares/input Balmorel/base/data'%(branch, commit)) 

