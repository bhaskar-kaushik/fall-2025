version 17
clear all
set more off

webuse nlsw88, clear
drop if mi(occupation)
recode occupation (8 9 10 11 12 13 = 7)
gen white = race==1

mlogit occupation age white collgrad, base(7)

* optional reporting
mlogit, rrr
predict double p1-p7
egen yhat = rowmax(p1-p7)
tab occupation
tab yhat
