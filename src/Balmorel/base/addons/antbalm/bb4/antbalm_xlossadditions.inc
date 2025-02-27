$onmulti
$if     EXIST '../data/ANTBALM_XLOSS.inc' $INCLUDE         '../data/ANTBALM_XLOSS.inc';
$if not EXIST '../data/ANTBALM_XLOSS.inc' $INCLUDE '../../base/data/ANTBALM_XLOSS.inc';
$offmulti