
$onmulti
$if     EXIST '../data/ANTBALM_XCOST.inc' $INCLUDE         '../data/ANTBALM_XCOST.inc';
$if not EXIST '../data/ANTBALM_XCOST.inc' $INCLUDE '../../base/data/ANTBALM_XCOST.inc';
$offmulti
