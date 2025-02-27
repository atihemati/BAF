$onmulti
$if     EXIST '../data/ANTBALM_G.inc' $INCLUDE         '../data/ANTBALM_G.inc';
$if not EXIST '../data/ANTBALM_G.inc' $INCLUDE '../../base/data/ANTBALM_G.inc';
$offmulti
