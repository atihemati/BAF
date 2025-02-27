* Flexdem add-on. Added by Mathias Berg Rosendal

$ifi not %seasonal_demand%==yes + SUM(FLEXUSER, VFLEXDEMAND_T(FLEXUSER,IY411,IR,IS3,T)$(FLEXYDEMAND(IY411,IR,FLEXUSER)))
$ifi %seasonal_demand%==yes + SUM(FLEXUSER, VFLEXDEMAND_T(FLEXUSER,IY411,IR,IS3,T)$(FLEXSDEMAND(IY411,IS3,IR,FLEXUSER)))


