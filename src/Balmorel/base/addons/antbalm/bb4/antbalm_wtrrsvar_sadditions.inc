$ifi not %ADJUSTHYDRO%==yes $goto dont_adjust_hydro
$onmulti
$if     EXIST '../data/ANTBALM_WTRRSVAR_S.inc' $INCLUDE         '../data/ANTBALM_WTRRSVAR_S.inc';
$if not EXIST '../data/ANTBALM_WTRRSVAR_S.inc' $INCLUDE '../../base/data/ANTBALM_WTRRSVAR_S.inc';
$offmulti
$label dont_adjust_hydro