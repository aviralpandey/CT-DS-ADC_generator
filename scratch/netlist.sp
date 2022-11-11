* Test Netlist 

* Anonymous circuit.Package
* Written by SpiceNetlister
* 

.SUBCKT DiffClkGen_504864ef49642a9555681843b46b1e09_ 
+ VSS ck_p ck_n 
* No parameters

vvp 
+ ck_p VSS 
+ pulse ('900.5m' '899.5m' '1n' '800p' '800p' '1200p' '4n') 
* No parameters


vvn 
+ ck_n VSS 
+ pulse ('899.5m' '900.5m' '1n' '800p' '800p' '1200p' '4n') 
* No parameters


.ENDS

.SUBCKT StrongArm_60672f1d8be76c6894b207aade9ca2cb_ 
+ VDD VSS clk out_p out_n inp_p inp_n 
* No parameters

xnclk 
+ tail_pre clk VSS VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='4.0' l='0.15' nf='4' 

vtail_meas 
+ tail_meas_p tail_pre 
+ dc '0' 
* No parameters


xninp 
+ ninp_d inp_p tail_meas_p VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='2.0' l='0.15' nf='2' 

xninn 
+ ninn_d inp_n tail_meas_p VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='2.0' l='0.15' nf='2' 

vninp_meas 
+ ninp_meas_p ninp_d 
+ dc '0' 
* No parameters


vninn_meas 
+ ninn_meas_p ninn_d 
+ dc '0' 
* No parameters


xnlatp 
+ nlatp_d out_p ninp_meas_p VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='2.0' l='0.15' nf='2' 

xnlatn 
+ nlatn_d out_n ninn_meas_p VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='2.0' l='0.15' nf='2' 

vnlatp_meas 
+ out_n nlatp_d 
+ dc '0' 
* No parameters


vnlatn_meas 
+ out_p nlatn_d 
+ dc '0' 
* No parameters


xplatp 
+ platp_d out_p VDD VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='2.0' l='0.15' nf='2' 

xplatn 
+ platn_d out_n VDD VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='2.0' l='0.15' nf='2' 

vplatp_meas 
+ platp_d out_n 
+ dc '0' 
* No parameters


vplatn_meas 
+ platn_d out_p 
+ dc '0' 
* No parameters


xprstp 
+ out_p clk VDD VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='4.0' l='0.15' nf='4' 

xprstn 
+ out_n clk VDD VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='4.0' l='0.15' nf='4' 

xprstp2 
+ ninp_d clk VDD VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='4.0' l='0.15' nf='4' 

xprstn2 
+ ninn_d clk VDD VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='4.0' l='0.15' nf='4' 

.ENDS

.SUBCKT Nor2_7a38543875e00a57732f66d365a3c641_ 
+ VDD VSS i fb z 
* No parameters

xpi 
+ pi_d i VDD VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='2.0' l='0.15' nf='2' 

xpfb 
+ z fb pi_d VDD 
+ sky130_fd_pr__pfet_01v8 
+ w='2.0' l='0.15' nf='2' 

xnfb 
+ z fb VSS VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='2.0' l='0.15' nf='2' 

xni 
+ z i VSS VSS 
+ sky130_fd_pr__nfet_01v8 
+ w='2.0' l='0.15' nf='2' 

.ENDS

.SUBCKT SrLatch_7a38543875e00a57732f66d365a3c641_ 
+ VDD VSS out_p out_n inp_p inp_n 
* No parameters

xnorp 
+ VDD VSS inp_p out_p out_n 
+ Nor2_7a38543875e00a57732f66d365a3c641_ 
* No parameters


xnorn 
+ VDD VSS inp_n out_n out_p 
+ Nor2_7a38543875e00a57732f66d365a3c641_ 
* No parameters


.ENDS

.SUBCKT Comparator_4a6328736fa1c527904479c045dfbe4f_ 
+ VDD VSS clk out_p out_n inp_p inp_n 
* No parameters

xsa 
+ VDD VSS clk sout_p sout_n inp_p inp_n 
+ StrongArm_60672f1d8be76c6894b207aade9ca2cb_ 
* No parameters


xsr 
+ VDD VSS out_p out_n sout_p sout_n 
+ SrLatch_7a38543875e00a57732f66d365a3c641_ 
* No parameters


.ENDS

.SUBCKT SlicerTb_b66cb43e40711054cbf1d4c40fc10cee_ 
+ VSS 
* No parameters

vvvdd 
+ VDD VSS 
+ dc '1800m' 
* No parameters


xinpgen 
+ VSS inp_p inp_n 
+ DiffClkGen_504864ef49642a9555681843b46b1e09_ 
* No parameters


vvclk 
+ clk VSS 
+ pulse ('0' '1800m' '0' '1p' '1p' '1n' '2n') 
* No parameters


cclp 
+ out_p VSS 
+ 5.000000000000001e-15 
* No parameters


ccln 
+ out_n VSS 
+ 5.000000000000001e-15 
* No parameters


xdut 
+ VDD VSS clk out_p out_n inp_p inp_n 
+ Comparator_4a6328736fa1c527904479c045dfbe4f_ 
* No parameters


.ENDS

xtop 0 SlicerTb_b66cb43e40711054cbf1d4c40fc10cee_ // Top-Level DUT 

.lib "/Users/cemyalcin/opt/anaconda3/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice" tt 
.tran 1e-10 1.2000000000000002e-08 

