$onmulti
$if     EXIST '../data/ANTBALM_AGKN.inc' $INCLUDE         '../data/ANTBALM_AGKN.inc';
$if not EXIST '../data/ANTBALM_AGKN.inc' $INCLUDE '../../base/data/ANTBALM_AGKN.inc';
$offmulti
