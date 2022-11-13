* Test Netlist 

* Anonymous circuit.Package
* Written by SpiceNetlister
* 

.SUBCKT CommonSource_1e4f9b44d45e0416f9135fadc4dd89e0_ 
+ VSS VDD vin vout 
* No parameters

xnmos 
+ vout vin VSS VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='1' l='150m' nf='1' mult='1' 

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
+ CommonSource_1e4f9b44d45e0416f9135fadc4dd89e0_ 
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

