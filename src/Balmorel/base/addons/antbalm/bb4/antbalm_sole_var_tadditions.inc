
$onmulti
$if     EXIST '../data/ANTBALM_SOLE_VAR_T.inc' $INCLUDE         '../data/ANTBALM_SOLE_VAR_T.inc';
$if not EXIST '../data/ANTBALM_SOLE_VAR_T.inc' $INCLUDE '../../base/data/ANTBALM_SOLE_VAR_T.inc';
$offmulti