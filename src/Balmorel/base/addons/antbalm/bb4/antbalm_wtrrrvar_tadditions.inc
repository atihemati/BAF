$ifi not %ADJUSTHYDRO%==yes $goto dont_adjust_hydro
$onmulti
$if     EXIST '../data/ANTBALM_WTRRRVAR_T.inc' $INCLUDE         '../data/ANTBALM_WTRRRVAR_T.inc';
$if not EXIST '../data/ANTBALM_WTRRRVAR_T.inc' $INCLUDE '../../base/data/ANTBALM_WTRRRVAR_T.inc';
$offmulti
$label dont_adjust_hydro