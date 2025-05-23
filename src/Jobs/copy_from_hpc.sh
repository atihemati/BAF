#!/bin/bash

# Copy all results from HPC to local
scp mberos@transfer.gbar.dtu.dk:/work3/mberos/BAF/src/Workflow/MetaResults/* Workflow/MetaResults
scp mberos@transfer.gbar.dtu.dk:/work3/mberos/BAF/src/Workflow/OverallResults/* Workflow/OverallResults
scp mberos@transfer.gbar.dtu.dk:/work3/mberos/BAF/src/Balmorel/base/model/MainResults_*.gdx Balmorel/base/model