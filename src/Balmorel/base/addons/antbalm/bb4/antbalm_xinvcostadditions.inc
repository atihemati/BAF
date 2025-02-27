$onmulti
$if     EXIST '../data/ANTBALM_XINVCOST.inc' $INCLUDE         '../data/ANTBALM_XINVCOST.inc';
$if not EXIST '../data/ANTBALM_XINVCOST.inc' $INCLUDE '../../base/data/ANTBALM_XINVCOST.inc';
$offmulti
