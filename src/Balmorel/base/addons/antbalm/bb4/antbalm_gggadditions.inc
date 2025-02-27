$onmulti
$if     EXIST '../data/ANTBALM_GGG.inc' $INCLUDE         '../data/ANTBALM_GGG.inc';
$if not EXIST '../data/ANTBALM_GGG.inc' $INCLUDE '../../base/data/ANTBALM_GGG.inc';
$offmulti
