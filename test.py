from decimal import Decimal
from pathlib import Path
import os
from typing import Tuple

import hdl21 as h
from hdl21.primitives import Vdc
from vlsirtools.spice import SimOptions, SupportedSimulators, ResultFormat


CONDA_PREFIX = os.environ.get("CONDA_PREFIX", None)

sim_options = SimOptions(
    rundir=Path("./scratch"),
    fmt=ResultFormat.SIM_DATA,
    simulator=SupportedSimulators.NGSPICE,
)


@h.paramclass
class MosParams:
    w = h.Param(dtype=float, desc="Channel Width")
    l = h.Param(dtype=float, desc="Channel Length")
    nf = h.Param(dtype=int, desc="Number of fingers")


nch = h.ExternalModule(
    name="sky130_fd_pr__nfet_01v8", desc="Sky130 NMOS", 
    port_list=[h.Inout(name="D"), h.Inout(name="G"), h.Inout(name="S"), h.Inout(name="B")], 
    paramtype=MosParams)
pch = h.ExternalModule(
    name="sky130_fd_pr__pfet_01v8", desc="Sky130 PMOS", 
    port_list=[h.Inout(name="D"), h.Inout(name="G"), h.Inout(name="S"), h.Inout(name="B")], 
    paramtype=MosParams)


@h.paramclass
class CommonSourceParams:
    nmos_params = h.Param(dtype=MosParams, desc="NMOS Parameters")
    res_value = h.Param(dtype=float, desc="Drain resistor Value")


@h.generator
def common_source_amp_gen(params: CommonSourceParams) -> h.Module:
    @h.module
    class CommonSource:
        VSS = h.Inout()
        VDD = h.Inout()
        vin = h.Input()
        vout = h.Output()
        nmos = nch(params.nmos_params)(D=vout, G=vin, S=VSS, B=VSS)
        res = h.Resistor(r=params.res_value)(p=VDD, n=vout)
    
    return CommonSource


def get_amp_performance(amp: h.Module) -> Tuple[float, float]:
    """ Measure the Amplifier Output DC Voltage and Supply Current """
    tb = h.sim.tb("CommonSourceTb")
    tb.VDD = h.Signal()
    tb.vout = h.Signal()
    tb.vin = h.Signal()
    tb.VDD_src = Vdc(Vdc.Params(dc=Decimal('1.8')))(p=tb.VDD, n=tb.VSS)
    tb.vin_src = Vdc(Vdc.Params(dc='1.5', ac='1.0'))(p=tb.vin, n=tb.VSS)
    tb.dut = amp(VDD=tb.VDD, VSS=tb.VSS, vin=tb.vin, vout=tb.vout)

    sim = h.sim.Sim(tb=tb)
    sim.lib(f"{CONDA_PREFIX}/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice", 'tt')
    
    sim.op()
    sim.ac(h.sim.LogSweep(start='1', stop='100e3', npts='10'))
    sim.literal(".save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[vth]")
    sim.literal(".save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[gm]")
    sim.literal(".save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[id]")
    sim.literal(".save @m.xtop.xdut.xnmos.msky130_fd_pr__nfet_01v8[cgg]")
    sim.literal(".save all")
    results = sim.run(sim_options)
    op_results = results.an[0].data
    ac_results = results.an[1].data
    vout_dc = op_results['v(xtop.vdd)']
    dc_current = op_results['i(v.xtop.vvdd_src)']
    vout_ac = ac_results['v(xtop.vdd)']
    breakpoint()
    return vout_dc, dc_current
    

if __name__ == '__main__':
    get_amp_performance(
        common_source_amp_gen(CommonSourceParams(MosParams(w=1, l=0.15, nf=1), 1e3)))
