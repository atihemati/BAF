* Include fictive demand
$ifi not %FICTDEM%==yes $goto NoFictDem
$onmulti
$if     EXIST '../data/ANTBALM_FICTDE.inc' $INCLUDE         '../data/ANTBALM_FICTDE.inc';
$if not EXIST '../data/ANTBALM_FICTDE.inc' $INCLUDE '../../base/data/ANTBALM_FICTDE.inc';
$offmulti
$label NoFictDem
