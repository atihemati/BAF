$onmulti
$if     EXIST '../data/ANTBALM_GKFX.inc' $INCLUDE         '../data/ANTBALM_GKFX.inc';
$if not EXIST '../data/ANTBALM_GKFX.inc' $INCLUDE '../../base/data/ANTBALM_GKFX.inc';
$offmulti
