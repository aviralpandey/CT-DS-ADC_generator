import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import hdl21 as h
import hdl21.sim as hs
from hdl21 import Diff
from hdl21.pdk import Corner
from hdl21.sim import Sim, LogSweep
from hdl21.prefix import m, µ, f, n, PICO
from hdl21.primitives import Vdc, Idc, C, Vpulse
from vlsirtools.spice import SimOptions, SupportedSimulators, ResultFormat


CONDA_PREFIX = os.environ.get("CONDA_PREFIX", None)


sim_options = SimOptions(
    rundir=Path("./scratch"),
    fmt=ResultFormat.SIM_DATA,
    simulator=SupportedSimulators.NGSPICE,
)


@h.paramclass
class DiffClkParams:
    """Differential Clock Generator Parameters"""

    period = h.Param(dtype=hs.ParamVal, desc="Period")
    delay = h.Param(dtype=hs.ParamVal, desc="Delay")
    vd = h.Param(dtype=hs.ParamVal, desc="Differential Voltage")
    vc = h.Param(dtype=hs.ParamVal, desc="Common-Mode Voltage")
    trf = h.Param(dtype=hs.ParamVal, desc="Rise / Fall Time")


@h.generator
def DiffClkGen(p: DiffClkParams) -> h.Module:
    """# Differential Clock Generator
    For simulation, from ideal pulse voltage sources"""

    ckg = h.Module()
    ckg.VSS = VSS = h.Port()
    ckg.ck = ck = Diff(role=Diff.Roles.SINK, port=True)

    def vparams(polarity: bool) -> Vpulse.Params:
        """Closure to create the pulse-source parameters for each differential half.
        Argument `polarity` is True for positive half, False for negative half."""
        # Initially create the voltage levels for the positive half
        v1 = p.vc + p.vd / 2
        v2 = p.vc - p.vd / 2
        if not polarity:  # And for the negative half, swap them
            v1, v2 = v2, v1
        return Vpulse.Params(
            v1=v1,
            v2=v2,
            period=p.period,
            rise=p.trf,
            fall=p.trf,
            width=p.period / 2 - p.trf,
            delay=p.delay,
        )

    # Create the two complementary pulse-sources
    ckg.vp = Vpulse(vparams(True))(p=ck.p, n=VSS)
    ckg.vn = Vpulse(vparams(False))(p=ck.n, n=VSS)
    return ckg


@h.paramclass
class MosParams:
    w = h.Param(dtype=float, desc="Channel Width")
    l = h.Param(dtype=float, desc="Channel Length")
    nf = h.Param(dtype=int, desc="Number of fingers")


nch = h.ExternalModule(
    name="sky130_fd_pr__nfet_01v8", desc="Sky130 NMOS", 
    port_list=[h.Inout(name="d"), h.Inout(name="g"), h.Inout(name="s"), h.Inout(name="b")], 
    paramtype=MosParams)
pch = h.ExternalModule(
    name="sky130_fd_pr__pfet_01v8", desc="Sky130 PMOS", 
    port_list=[h.Inout(name="d"), h.Inout(name="g"), h.Inout(name="s"), h.Inout(name="b")], 
    paramtype=MosParams)


@h.paramclass
class LatchParams:
    nor_pi = h.Param(dtype=MosParams, desc="Latch NOR input PMOS params")
    nor_pfb = h.Param(dtype=MosParams, desc="Latch NOR feedback PMOS params")
    nor_ni = h.Param(dtype=MosParams, desc="Latch NOR input NMOS params")
    nor_nfb = h.Param(dtype=MosParams, desc="Latch NOR feedback NMOS params")


@h.generator
def nor2(params: LatchParams) -> h.Module:
    @h.module
    class Nor2:
        """# Nor2 for SR Latch
        Inputs `i` and `fb` are designated for input and feedback respectively.
        The feedback input is the faster of the two."""
        VDD, VSS = h.Ports(2)
        i, fb = h.Inputs(2)
        z = h.Output()

        pi = pch(params.nor_pi)(g=i, s=VDD, b=VDD)
        pfb = pch(params.nor_pfb)(g=fb, d=z, s=pi.d, b=VDD)
        nfb = nch(params.nor_nfb)(g=fb, d=z, s=VSS, b=VSS)
        ni = nch(params.nor_ni)(g=i, d=z, s=VSS, b=VSS)
    return Nor2


@h.generator
def sr_latch(params: LatchParams) -> h.Module:
    @h.module
    class SrLatch:
        VDD, VSS = h.Ports(2)
        inp = Diff(port=True, role=Diff.Roles.SINK)
        out = Diff(port=True, role=Diff.Roles.SOURCE)

        norp = nor2(params)(i=inp.p, z=out.n, fb=out.p, VDD=VDD, VSS=VSS)
        norn = nor2(params)(i=inp.n, z=out.p, fb=out.n, VDD=VDD, VSS=VSS)
    
    return SrLatch


@h.paramclass
class StrongarmParams:
    tail = h.Param(dtype=MosParams, desc="Tail FET params")
    inp_pair = h.Param(dtype=MosParams, desc="Input pair FET params")
    inv_n = h.Param(dtype=MosParams, desc="Inverter NMos params")
    inv_p = h.Param(dtype=MosParams, desc="Inverter PMos params")
    reset = h.Param(dtype=MosParams, desc="Reset Device params")
    meas_vs = h.Param(
            dtype=bool, 
            desc="True to add voltage sources to measure device currents")


@h.generator
def strongarm(params: StrongarmParams) -> h.Module:
    if params.meas_vs:
        @h.module
        class StrongArm:
            VDD, VSS = h.Ports(2)
            inp = Diff(port=True, role=Diff.Roles.SINK)
            out = Diff(port=True, role=Diff.Roles.SOURCE)
            clk = h.Input()
            
            tail_pre = h.Signal()
            mid_pre = Diff()
            inv_n_pre = Diff()
            inv_p_pre = Diff()

            nclk = nch(params.tail)(g=clk, s=VSS, b=VSS, d=tail_pre)
            tail_meas = Vdc(Vdc.Params(dc=0))(n=tail_pre)

            ## Input Pair
            ninp = nch(params.inp_pair)(g=inp.p, s=tail_meas.p, b=VSS)
            ninn = nch(params.inp_pair)(g=inp.n, s=tail_meas.p, b=VSS)
            ninp_meas = Vdc(Vdc.Params(dc=0))(n=ninp.d)
            ninn_meas = Vdc(Vdc.Params(dc=0))(n=ninn.d)
            ## Latch nch
            nlatp = nch(params.inv_n)(g=out.p, s=ninp_meas.p, b=VSS)
            nlatn = nch(params.inv_n)(g=out.n, s=ninn_meas.p, b=VSS)
            nlatp_meas = Vdc(Vdc.Params(dc=0))(n=nlatp.d, p=out.n)
            nlatn_meas = Vdc(Vdc.Params(dc=0))(n=nlatn.d, p=out.p)
            ## Latch Pmos
            platp = pch(params.inv_p)(g=out.p, s=VDD, b=VDD)
            platn = pch(params.inv_p)(g=out.n, s=VDD, b=VDD)
            platp_meas = Vdc(Vdc.Params(dc=0))(p=platp.d, n=out.n)
            platn_meas = Vdc(Vdc.Params(dc=0))(p=platn.d, n=out.p)
            ## Reset pch
            prstp = pch(params.reset)(g=clk, d=out.p, s=VDD, b=VDD)
            prstn = pch(params.reset)(g=clk, d=out.n, s=VDD, b=VDD)
            prstp2 = pch(params.reset)(g=clk, d=ninp_meas.p, s=VDD, b=VDD)
            prstn2 = pch(params.reset)(g=clk, d=ninn_meas.p, s=VDD, b=VDD)
    
        return StrongArm
    else:
        @h.module
        class StrongArm:
            VDD, VSS = h.Ports(2)
            inp = Diff(port=True, role=Diff.Roles.SINK)
            out = Diff(port=True, role=Diff.Roles.SOURCE)
            clk = h.Input()

            nclk = nch(params.tail)(g=clk, s=VSS, b=VSS)
            ## Input Pair
            ninp = nch(params.inp_pair)(g=inp.p, s=nclk.d, b=VSS)
            ninn = nch(params.inp_pair)(g=inp.n, s=nclk.d, b=VSS)
            ## Latch nch
            nlatp = nch(params.inv_n)(g=out.p, d=out.n, s=ninp.d, b=VSS)
            nlatn = nch(params.inv_n)(g=out.n, d=out.p, s=ninn.d, b=VSS)
            ## Latch Pmos
            platp = pch(params.inv_p)(g=out.p, d=out.n, s=VDD, b=VDD)
            platn = pch(params.inv_p)(g=out.n, d=out.p, s=VDD, b=VDD)
            ## Reset pch
            prstp = pch(params.reset)(g=clk, d=out.p, s=VDD, b=VDD)
            prstn = pch(params.reset)(g=clk, d=out.n, s=VDD, b=VDD)
    
        return StrongArm


@h.paramclass
class ComparatorParams:
    strongarm = h.Param(dtype=StrongarmParams, desc="Strongarm Parameters")
    latch = h.Param(dtype=LatchParams, desc="Latch Parameters")


@h.generator
def comparator(params: ComparatorParams) -> h.Module:
    @h.module
    class Comparator:
        """# StrongArm Based Comparator """
        VDD, VSS = h.Ports(2)
        inp = Diff(port=True, role=Diff.Roles.SINK)
        out = Diff(port=True, role=Diff.Roles.SOURCE)
        clk = h.Input()

        sout = Diff()
        
        sa = strongarm(params.strongarm)(
            inp=inp, out=sout, clk=clk, VDD=VDD, VSS=VSS)
        sr = sr_latch(params.latch)(inp=sout, out=out, VDD=VDD, VSS=VSS)

    return Comparator
    
    
""" 
# Comparator Tests 
"""


@h.paramclass
class Pvt:
    """Process, Voltage, and Temperature Parameters"""

    p = h.Param(dtype=Corner, desc="Process Corner", default=Corner.TYP)
    v = h.Param(dtype=h.Prefixed, desc="Supply Voltage Value (V)", default=1800 * m)
    t = h.Param(dtype=int, desc="Simulation Temperature (C)", default=25)


@h.paramclass
class TbParams:
    dut = h.Param(dtype=ComparatorParams, desc="DUT params")
    pvt = h.Param(
        dtype=Pvt, desc="Process, Voltage, and Temperature Parameters", default=Pvt()
    )
    vd = h.Param(dtype=h.Prefixed, desc="Differential Voltage (V)", default=100 * m)
    vc = h.Param(dtype=h.Prefixed, desc="Common-Mode Voltage (V)", default=900 * m)
    cl = h.Param(dtype=h.Prefixed, desc="Load Cap (Single-Ended) (F)", default=5 * f)


@h.generator
def ComparatorTb(p: TbParams) -> h.Module:
    """Comparator Testbench"""

    # Create our testbench
    tb = h.sim.tb("SlicerTb")
    # Generate and drive VDD
    tb.VDD = VDD = h.Signal()
    tb.vvdd = Vdc(Vdc.Params(dc=p.pvt.v))(p=VDD, n=tb.VSS)

    # Input-driving balun
    tb.inp = Diff()
    tb.inpgen = DiffClkGen(
        DiffClkParams(period=4 * n, delay=1 * n, vc=p.vc, vd=p.vd, trf=800 * PICO)
    )(ck=tb.inp, VSS=tb.VSS)

    # Clock generator
    tb.clk = clk = h.Signal()
    tb.vclk = Vpulse(
        Vpulse.Params(
            delay=0,
            v1=0,
            v2=p.pvt.v,
            period=2 * n,
            rise=100 * PICO,
            fall=100 * PICO,
            width=1 * n,
        )
    )(p=clk, n=tb.VSS)

    # Output & Load Caps
    tb.out = Diff()
    Cload = C(C.Params(c=p.cl))
    tb.clp = Cload(p=tb.out.p, n=tb.VSS)
    tb.cln = Cload(p=tb.out.n, n=tb.VSS)

    # Create the Slicer DUT
    tb.dut = comparator(p.dut)(
        inp=tb.inp,
        out=tb.out,
        clk=clk,
        VDD=VDD,
        VSS=tb.VSS)
    return tb


def test_comparator_sim():
    """Comparator Test(s)"""

    w = 1.0
    l = 0.15
    nf = 2
    comparator_params = ComparatorParams(
        strongarm=StrongarmParams(
            tail=MosParams(w=w*nf*2, l=l, nf=2*nf),
            inp_pair=MosParams(w=w*nf, l=l, nf=nf),
            inv_n=MosParams(w=w*nf, l=l, nf=nf),
            inv_p=MosParams(w=w*nf, l=l, nf=nf),
            reset=MosParams(w=w*nf*2, l=l, nf=2*nf),
            meas_vs=True),
        latch=LatchParams(
            nor_pi=MosParams(w=w*nf, l=l, nf=nf),
            nor_pfb=MosParams(w=w*nf, l=l, nf=nf),
            nor_ni=MosParams(w=w*nf, l=l, nf=nf),
            nor_nfb=MosParams(w=w*nf, l=l, nf=nf)))
    # Create our parametric testbench
    params = TbParams(pvt=Pvt(), vc=900 * m, vd=1 * m, dut=comparator_params)

    # Create our simulation input
    @hs.sim
    class ComparatorSim:
        tb = ComparatorTb(params)
        tr = hs.Tran(tstop=12 * n, tstep=100*PICO)

    # Add the PDK dependencies
    ComparatorSim.lib(f"{CONDA_PREFIX}/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice", 'tt')
    ComparatorSim.literal(".option METHOD=Gear")

    # Run Spice, save important results
    results = ComparatorSim.run(sim_options)
    tran_results = results.an[0].data
    np.savez('strongarm_results.npz', 
        t = tran_results['time'],
        v_out_diff = tran_results['v(xtop.out_p)'] - tran_results['v(xtop.out_n)'],
        v_in_diff = tran_results['v(xtop.inp_p)'] - tran_results['v(xtop.inp_n)'],
        v_clk = tran_results['v(xtop.clk)'],
        i_tail = tran_results['i(v.xtop.xdut.xsa.vtail_meas)'],
        i_inp_pair_cm = tran_results['i(v.xtop.xdut.xsa.vninp_meas)'] +\
                tran_results['i(v.xtop.xdut.xsa.vninn_meas)'],
        i_latch_n_pair_cm = tran_results['i(v.xtop.xdut.xsa.vnlatn_meas)'] +\
                tran_results['i(v.xtop.xdut.xsa.vnlatp_meas)'],
        i_latch_p_pair_cm = tran_results['i(v.xtop.xdut.xsa.vplatn_meas)'] +\
                tran_results['i(v.xtop.xdut.xsa.vplatp_meas)'],
        v_casc_cm =  (tran_results['v(xtop.xdut.xsa.ninn_meas_p)'] +
                tran_results['v(xtop.xdut.xsa.ninp_meas_p)']) / 2,
        v_out_cm =  (tran_results['v(xtop.xdut.sout_p)'] +
                tran_results['v(xtop.xdut.sout_n)']) / 2,
    )


def extract_windows(t, clk, threshold):
    """ Given the clock waveform, this will extract a bunch of single periods
    that are the periods at which a certain clock starts and ends """
    clk_above_thres_idcs = np.where(clk > threshold)[0]
    print(clk_above_thres_idcs.shape)
    print(np.diff(clk_above_thres_idcs))
    rising_cross_idcs = clk_above_thres_idcs[np.where(
        np.diff(np.concatenate(
            ([0], clk_above_thres_idcs), axis=0)) > 1)[0]]
    rising_cross_idcs = rising_cross_idcs.tolist() + [len(clk)]
    return [(s,e) for s, e in zip(rising_cross_idcs[:-1], rising_cross_idcs[1:])]


def plot_windows():
    data = np.load('strongarm_results.npz')
    windows = extract_windows(data['t'], data['v_clk'], 0.7)
    for (s, e) in windows:
        plt.figure()
        plt.plot(data['t'][s:e], data['v_clk'][s:e])
    plt.show()


def plot_data():
    data = np.load('strongarm_results.npz')
    t = data['t']
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, data['v_clk'])
    ax[1].plot(t, data['v_in_diff'])
    ax[2].plot(t, data['v_out_diff'])
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, data['v_clk'])
    ax[1].plot(t, data['i_tail'], label='tail current')
    ax[1].plot(t, data['i_inp_pair_cm'], label='input pair current')
    ax[1].plot(t, data['i_latch_n_pair_cm'], label='Latch NMOS current')
    ax[1].plot(t, data['i_latch_p_pair_cm'], label='Latch PMOS current')
    ax[1].legend()
    ax[2].plot(t, data['v_casc_cm'], label='v_casc_cm')
    ax[2].plot(t, data['v_out_cm'], label='v_out_cm')
    ax[2].legend()

    plt.show()
    breakpoint()


if __name__ == '__main__':
    # test_comparator_sim()
    # plot_data()
    plot_windows()
