
$onmulti
$if     EXIST '../data/ANTBALM_DISCOST_E.inc' $INCLUDE         '../data/ANTBALM_DISCOST_E.inc';
$if not EXIST '../data/ANTBALM_DISCOST_E.inc' $INCLUDE '../../base/data/ANTBALM_DISCOST_E.inc';
$offmulti
