$onmulti
$if     EXIST '../data/ANTBALM_XKFX.inc' $INCLUDE         '../data/ANTBALM_XKFX.inc';
$if not EXIST '../data/ANTBALM_XKFX.inc' $INCLUDE '../../base/data/ANTBALM_XKFX.inc';
$offmulti