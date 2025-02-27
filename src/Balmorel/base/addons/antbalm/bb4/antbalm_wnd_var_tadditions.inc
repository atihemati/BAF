
$onmulti
$if     EXIST '../data/ANTBALM_WND_VAR_T.inc' $INCLUDE         '../data/ANTBALM_WND_VAR_T.inc';
$if not EXIST '../data/ANTBALM_WND_VAR_T.inc' $INCLUDE '../../base/data/ANTBALM_WND_VAR_T.inc';
$offmulti