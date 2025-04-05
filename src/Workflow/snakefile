# Define the final outputs that we want to generate
rule all:
    input:
        [
            "Antares/input/renewables/series/ch/onshore/series.txt",
            "Balmorel/base/data/WND_VAR_T.inc",
            "Balmorel/base/data/WTRRRVAR_T.inc"
        ]


# Rule for generating mappings
rule generate_mappings:
    output:
        [
            "Pre-Processing/Output/B2A_regi.pkl",
            "Pre-Processing/Output/A2B_regi.pkl"
        ]
    shell:
        "pixi run generate-mappings"

# Rule for generating VRE profiles for Antares
rule generate_antares_vre:
    input:
        "Pre-Processing/Output/A2B_regi.pkl"
    output:
        "Antares/input/renewables/series/ch/onshore/series.txt"
    shell:
        "pixi run generate-antares-vre"

# Rule for generating Balmorel timeseries (VRE and exogenous electricity)
rule generate_balmorel_timeseries:
    input:
        [
            "Pre-Processing/Output/A2B_regi.pkl",
            "Pre-Processing/Data/IncFile PreSuffixes/WND_VAR_T.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WND_VAR_T_OFF.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/SOLE_VAR_T.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/DE_VAR_T.inc",
        ]
    output:
        "Balmorel/base/data/WND_VAR_T.inc"
    shell:
        "pixi run generate-balmorel-timeseries"

# Rule for generating Balmorel timeseries for hydropower
rule generate_balmorel_hydro:
    input:
        [
            "Pre-Processing/Output/A2B_regi.pkl",
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRRFLH.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRSFLH.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRRVAR_T.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/WTRRSVAR_S.inc",
            "Pre-Processing/Data/IncFile PreSuffixes/HYRSMAXVOL_G.inc",
        ]
    output:
        "Balmorel/base/data/WTRRRVAR_T.inc"
    shell:
        "pixi run generate-balmorel-hydro"