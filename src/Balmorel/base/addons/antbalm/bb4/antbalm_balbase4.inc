

* Note that this CAPCRED option is set by the soft-linking framework, not balopt
* In case you are running Balmorel, stand-alone, comment first line and outcomment second line below
$ifi not %CAPCRED%==yes $goto NoCapCred     

    QANTBALM_RESCAP
$ifi not %H2CAPCRED%==yes $goto NoH2CapCred   
    QANTBALM_H2RESCAP

$label NoH2CapCred
$label NoCapCred
