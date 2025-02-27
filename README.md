# Balmorel-Antares Soft-Coupling Framework v1.0

This framework was used in [Rosendal et al. 2025](https://doi.org/10.1016/j.apenergy.2025.125512), in an investigation of soft-linking strategies for coupling investment and operational models, in this case Balmorel and Antares. 

Get started by reading the [documentation](https://github.com/Mathias157/balmorel-antares/blob/master/Documentation/Balmorel_Antares_Soft_Coupling_Framework_Documentation.pdf).

This framework requires GAMS, Python and Antares. 
The specific data can unfortunately not be shared, but is located in the local DTU drive (M:\Project\RQI\BAF-Data_eur-system_7fb5073a.zip) of Mathias Berg Rosendal.

## Storing the Data
All code is stored in git, but most of the input data is not tracked!
Execute the following commands in powershell to zip data:

Windows:
powershell Compress-Archive -Path "Balmorel/base/data, Antares/input, Pre-Processing/Data, Pre-Processing/Output" -DestinationPath "BAF-Data_branch_version.zip"

Linux:
zip -r -q BAF-Data_branch_version.zip Pre-Processing/Output Pre-Processing/Data Antares/input Balmorel/base/data

## Unzipping the Data on HPC
Use the following command to unzip - -qq disables logging output, -o overwrites existing files 
unzip -qq -o BAF-Data_branch_version.zip

If unzipping the data file on a HPC, you may need to ensure writing capabilities on the extracted files by doing the following commands on the extracted folders: 
chmod -R +x data
chmod -R +x Pre-Processing
chmod -R +x input

Otherwise, these files will not be editable, which is needed in the framework