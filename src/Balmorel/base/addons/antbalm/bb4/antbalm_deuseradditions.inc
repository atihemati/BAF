
$ifi not %FICTDEM% == yes $goto NoFictDem
* Include fictive demand
$onmulti
$if     EXIST '../data/ANTBALM_FICTDEUSER.inc' $INCLUDE         '../data/ANTBALM_FICTDEUSER.inc';
$if not EXIST '../data/ANTBALM_FICTDEUSER.inc' $INCLUDE '../../base/data/ANTBALM_FICTDEUSER.inc';
$offmulti
$label NoFictDem
