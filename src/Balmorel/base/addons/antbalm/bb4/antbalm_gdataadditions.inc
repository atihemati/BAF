$onmulti
$if     EXIST '../data/ANTBALM_GDATA.inc' $INCLUDE         '../data/ANTBALM_GDATA.inc';
$if not EXIST '../data/ANTBALM_GDATA.inc' $INCLUDE '../../base/data/ANTBALM_GDATA.inc';
$offmulti
