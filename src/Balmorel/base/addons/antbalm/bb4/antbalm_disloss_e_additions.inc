$ifi not %USEANTARESDATA%==yes $goto NOUSEANTARESDATA
$onmulti
$if     EXIST '../data/ANTBALM_DISLOSS_E.inc' $INCLUDE         '../data/ANTBALM_DISLOSS_E.inc';
$if not EXIST '../data/ANTBALM_DISLOSS_E.inc' $INCLUDE '../../base/data/ANTBALM_DISLOSS_E.inc';
$offmulti
$label NOUSEANTARESDATA