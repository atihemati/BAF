###!/bin/sh�
### General options�
### -- specify queue --�
#BSUB -q man
### -- set the job Name --�
#BSUB -J PurgeMCyears
### -- ask for number of cores (default: 1) --�
#BSUB -n 1
### -- specify that the cores must be on the same host --�
#BSUB -R "span[hosts=1]"
### -- specify that we need X GB of memory per core/slot --�
#BSUB -R "rusage[mem=5GB]"
### -- specify that we want the job to get killed if it exceeds X GB per core/slot --�
#BSUB -M 5.1GB
### -- set walltime limit: hh:mm --�
#BSUB -W 1:00
### -- set the email address --�
##BSUB -u mberos@dtu.dk
### -- send notification at start --�
###BSUB -B
### -- send notification at completion --�
###BSUB -N
### -- Specify the output and error file. %J is the job-id --�
### -- -o and -e mean append, -oo and -eo mean overwrite --�
#BSUB -o ./Logs/CC&MarketVal_%J.out
#BSUB -e ./Logs/CC&MarketVal_%J.err
# here follow the commands you want to execute with input.in as the input file

### Load modules and find binaries
module load python3/3.9.11
module load R/4.2.3-mkl2023update1
source ../../.BAF-Env/bin/activate

### Get paths to binaries and Python-API for GAMS
export PATH=/zhome/c0/2/105719/Desktop/Antares-8.6.1/bin:$PATH
export PATH=/appl/gams/37.1.0:$PATH
export LD_LIBRARY_PATH=/appl/gams/37.1.0:$LD_LIBRARY_PATH
export PYTHONPATH=/appl/gams/37.1.0/apifiles/Python/gams:/appl/gams/37.1.0/apifiles/Python/api_39:$PYTHONPATH

# Running Master
cd Workflow
python3 PurgeMCyears.py
