$ifi not %ADJUSTHYDRO%==yes $goto dont_adjust_hydro
$onmulti
$if     EXIST '../data/ANTBALM_HYRSMAXVOL_G.inc' $INCLUDE         '../data/ANTBALM_HYRSMAXVOL_G.inc';
$if not EXIST '../data/ANTBALM_HYRSMAXVOL_G.inc' $INCLUDE '../../base/data/ANTBALM_HYRSMAXVOL_G.inc';
$offmulti
$label dont_adjust_hydro