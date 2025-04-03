# Pre-Processing

This page will explain how to Pre-Processing to prepare the Balmorel and Antares models. The main part of the Antares model was prepared using the [OSMOSE WP1 model](https://zenodo.org/records/7323821), while the Balmorel model is based on the [master branch from 7. January 2025](https://github.com/balmorelcommunity/Balmorel/tree/d89f89e99c9a4d14cc48ea2f5910da11c4cd5018). The output of these steps result in the Balmorel data of the [BAF](https://github.com/Mathias157/Balmorel_Data.git) branch, which can just be downloaded if you just wish to apply the framework.

## Tasks
The preprocessing is organised through tasks, which can be run through pixi with `pixi run <task-name>`. If you used a conda installation, check the exact commands in the pixi.toml under the `[tasks]` category.

### Generate Mapping

To generate spatial and technological mapping dictionaries between the models, run the `generate-mapping` task.
The Python function `generate_mapping` is hard-coded with names of technologies, fuels and regions. The tasks below depends on the output of this task.

### Generate Antares VRE

Generates the Antares solar, on- and offshore wind profiles with renewables in the cluster setting. 

### Generate Balmorel Hydro

Generates the Balmorel hydro data for reservoir, run-of-river and pumped storage.

