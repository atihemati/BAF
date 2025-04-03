# Installation

[GAMS 47+](https://www.gams.com/download/), [Antares 8.7](https://github.com/AntaresSimulatorTeam/Antares_Simulator/releases/tag/v8.7.0) and the described Python environment are required to run the BAF.

The pre-processing scripts requires R and the [R packages for Antares](https://github.com/rte-antares-rpackage), developed by RTE. However, this is only required if you want to do structual changes through python/R on the Antares model.

## Python Environment
Can be installed using [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [pixi](https://pixi.sh/latest/). A pixi installation will ensure that *all* packages are the same as in the time of developing BAF, while a conda installation through the `environment.yaml` could lead to different sub-packages being installed.

For pixi, simply [install pixi](https://pixi.sh/latest/#installation) and run `pixi install` in the top level of the folder. For conda, run `conda env create -f environment.yaml`.