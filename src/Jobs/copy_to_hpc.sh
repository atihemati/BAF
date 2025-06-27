#!/bin/bash

# Copy all results from HPC to local
scp Workflow/MetaResults/* mberos@transfer.gbar.dtu.dk:/work3/mberos/BAF/src/Workflow/MetaResults
scp Workflow/OverallResults/* mberos@transfer.gbar.dtu.dk:/work3/mberos/BAF/src/Workflow/OverallResults
scp Balmorel/base/model/MainResults_*.gdx mberos@transfer.gbar.dtu.dk:/work3/mberos/BAF/src/Balmorel/base/model 

# Change local GAMS system directory in meta files
# ls Workflow/MetaResults/*_meta.ini | xargs sed -i 's|/appl/gams/47.6.0|/opt/gams/48.5|g'