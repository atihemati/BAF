# Balmorel and Antares Soft-Linking Framework

This Balmorel-Antares soft-linking framework (BAF) was used in [Rosendal et al. 2025](https://doi.org/10.1016/j.apenergy.2025.125512), in an investigation of soft-linking strategies for coupling investment and operational models. The specific data of that study could unfortunately not be shared, but the framework will be reused here for a second, open-data application.

Get started by reading the [documentation](https://github.com/Mathias157/BAF/blob/master/docs/Balmorel_Antares_Soft_Coupling_Framework_Documentation.pdf).

## Installation
This framework requires GAMS 37+, Python 3.9.11 and Antares 8.6.1. 

The appropriate conda environment can be created using the environment file in this directory and the following command:
```
conda env create -f environment.yaml
```

## Storing the Data
All code is stored in git, but most of the input data is not tracked!
Execute the following commands in powershell to zip data:

Windows:

```
powershell Compress-Archive -Path "Balmorel/base/data, Antares/input, Pre-Processing/Data, Pre-Processing/Output" -DestinationPath "BAF-Data_branch_version.zip"
```

Linux:

```
zip -r -q BAF-Data_branch_version.zip Pre-Processing/Output Pre-Processing/Data Antares/input Balmorel/base/data
```

## Unzipping the Data on HPC
Use the following command to unzip - -qq disables logging output, -o overwrites existing files 
unzip -qq -o BAF-Data_branch_version.zip

If unzipping the data file on a HPC, you may need to ensure writing capabilities on the extracted files by doing the following commands on the extracted folders: 
```
chmod -R +x data
chmod -R +x Pre-Processing
chmod -R +x input
```

Otherwise, these files will not be editable, which is needed in the framework