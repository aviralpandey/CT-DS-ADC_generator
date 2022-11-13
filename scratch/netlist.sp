* Test Netlist 

* Anonymous circuit.Package
* Written by SpiceNetlister
* 

.SUBCKT CommonSource_e46bce81272b203318dfe064ef514e9e_ 
+ VSS VDD vin vout 
* No parameters

xnmos 
+ vout vin VSS VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='"1"' l='"0.15"' nf='1' mult='1' 

rres 
+ VDD vout 
+ 1000.0 
* No parameters


.ENDS

.SUBCKT CommonSourceTb 
+ VSS 
* No parameters

vVDD_src 
+ VDD VSS 
+ dc '1800m' 
+ ac='0' 

vvin_src 
+ vin VSS 
+ dc '1500m' 
+ ac='1' 

xdut 
+ VSS VDD vin vout 
+ CommonSource_e46bce81272b203318dfe064ef514e9e_ 
* No parameters


.ENDS

xtop 0 CommonSourceTb // Top-Level DUT 

.lib "/opt/conda/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice" tt 
.save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[vth]
.save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[gm]
.save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[id]
.save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[cgg]
.save all
.op

.ac dec 10 1.0 100000.0

