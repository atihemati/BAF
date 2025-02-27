***** ANTARES-BALMOREL SOFT LINK STUDY *****
* Mathias Berg Rosendal, 2023

* Market value on generator production
$ifi %MARKETVAL%==yes        - SUM((IR, S, T), SUM((IA,IGE)$(RRRAAA(IR,IA) AND IAGK_HASORPOT(Y,IA,IGE)), VGE_T(Y, IA, IGE, S, T) * ANTBALM_MARKETVAL(Y,IR,IGE)) )

* Market value on generator capacity (It's only feasible with the upper bound on VGKNACCUMNET below)
*    - SUM(IR, SUM((IA,G)$(RRRAAA(IR,IA) AND IAGKNY(Y,IA,G) AND (NOT IGKN_ES(G))), VGKNACCUMNET(Y,IA,G)*ANTBALM_MARKETVAL(Y,IR,G)))
* Upper bound for feasibility - make sure not to bypass VRE constraints as given by subtechgroupkpot etc
* This command unfortunately still overwrite VRE potentials, correct it
* Should be inserted to 'antbalm_eqndecdef' or the like
* VGKNACCUMNET.UP('2050',IA,G)$((GDATA(G,'GDTECHGROUP') NE WINDTURBINE_OFFSHORE) AND (GDATA(G,'GDTECHGROUP') NE WINDTURBINE_ONSHORE) AND (GDATA(G,'GDTECHGROUP') NE SOLARPV)) = 100000;


* Market value storage (should probably be modelled differently)
*    + SUM(G,  VGE_T() * MVPARAM(G) )


* Slack variable on minimum capacity, randomly assigned a high value of 1000000 €/MW (comparable to onshore wind capex)
*$ifi %CAPCRED%==yes    + 1e6 * VRESCAPSLACK