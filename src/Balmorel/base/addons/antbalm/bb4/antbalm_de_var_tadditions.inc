* Include fictive demand profile
$if     EXIST '../data/ANTBALM_DE_VAR_T.inc' $INCLUDE         '../data/ANTBALM_DE_VAR_T.inc';
$if not EXIST '../data/ANTBALM_DE_VAR_T.inc' $INCLUDE '../../base/data/ANTBALM_DE_VAR_T.inc';

$ifi not %FICTDEM%==yes $goto NoFictDem
$if     EXIST '../data/ANTBALM_FICTDE_VAR_T.inc' $INCLUDE         '../data/ANTBALM_FICTDE_VAR_T.inc';
$if not EXIST '../data/ANTBALM_FICTDE_VAR_T.inc' $INCLUDE '../../base/data/ANTBALM_FICTDE_VAR_T.inc';
$label NoFictDem
