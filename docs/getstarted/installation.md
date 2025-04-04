# Installation

[GAMS 47+](https://www.gams.com/download/), [Antares 8.7](https://github.com/AntaresSimulatorTeam/Antares_Simulator/releases/tag/v8.7.0) and the described [Python environment](#python-environment) are required to run the BAF. Running pre-processing (and some of the visualisations) also requires [R 4.2.3](https://cran.r-project.org/) and the setup described in [R setup](#r-setup). 

:::{warning}
BEFORE RUNNING pixi install: 

Make sure to edit the paths in `src/Jobs/set-R-path-for-pixi.*` (* = .bat for Windows, .sh for Mac/Linux) to YOUR R directories, if it is not part of your path already. Otherwise, the installation of rpy2 will use the 4.1.3 version of R, which is too old for antaresViz. 

If you accidentally did this, you can fix it later by
1. Changing the paths in `src/Jobs/set-R-path-for-pixi`
2. Removing rpy2 by running `pixi remove rpy2`
3. Install it again by running `pixi add rpy2==3.5.11` 
:::

## Python Environment
Can be installed using [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [pixi](https://pixi.sh/latest/). A pixi installation will ensure that *all* packages are the same as in the time of developing BAF, while a conda installation through the `environment.yaml` could lead to different sub-packages being installed.

For pixi, simply [install pixi](https://pixi.sh/latest/#installation) and run `pixi install` in the top level of the folder. For conda, run `conda env create -f environment.yaml`.

## R Setup
The pre-processing scripts and some visualisations use [R packages for Antares](https://github.com/rte-antares-rpackage), developed by RTE. Open up R and install them with the commands:
```
install.packages("antaresViz")
```