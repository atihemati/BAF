$onmulti
$if     EXIST '../data/ANTBALM_SUBTECHGROUPKPOT.inc' $INCLUDE         '../data/ANTBALM_SUBTECHGROUPKPOT.inc';
$if not EXIST '../data/ANTBALM_SUBTECHGROUPKPOT.inc' $INCLUDE '../../base/data/ANTBALM_SUBTECHGROUPKPOT.inc';
$offmulti