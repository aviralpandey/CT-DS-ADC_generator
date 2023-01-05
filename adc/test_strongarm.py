""" 
# Comparator Tests 
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import hdl21 as h
import hdl21.sim as hs
from hdl21 import Diff
from hdl21.pdk import Corner
from hdl21.prefix import m, f, n, PICO
from hdl21.primitives import Vdc, Idc, C, Vpulse
from vlsirtools.spice import SimOptions, SupportedSimulators, ResultFormat
import sky130, sitepdks as _

sim_options = SimOptions(
    rundir=Path("./scratch"),
    fmt=ResultFormat.SIM_DATA,
    simulator=SupportedSimulators.NGSPICE,
)

# Local DUT Imports
from .strongarm import (
    comparator,
    ComparatorParams,
    StrongarmParams,
    MosParams,
    LatchParams,
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
    tb.dut = comparator(p.dut)(inp=tb.inp, out=tb.out, clk=clk, VDD=VDD, VSS=tb.VSS)
    return tb


def test_comparator_sim():
    """Comparator Test(s)"""

    w = 1.0
    l = 0.15
    nf = 2
    comparator_params = ComparatorParams(
        strongarm=StrongarmParams(
            tail=MosParams(w=w * nf * 2, l=l, nf=2 * nf),
            inp_pair=MosParams(w=w * nf, l=l, nf=nf),
            inv_n=MosParams(w=w * nf, l=l, nf=nf),
            inv_p=MosParams(w=w * nf, l=l, nf=nf),
            reset=MosParams(w=w * nf * 2, l=l, nf=2 * nf),
            meas_vs=True,
        ),
        latch=LatchParams(
            nor_pi=MosParams(w=w * nf, l=l, nf=nf),
            nor_pfb=MosParams(w=w * nf, l=l, nf=nf),
            nor_ni=MosParams(w=w * nf, l=l, nf=nf),
            nor_nfb=MosParams(w=w * nf, l=l, nf=nf),
        ),
    )
    # Create our parametric testbench
    params = TbParams(pvt=Pvt(), vc=900 * m, vd=1 * m, dut=comparator_params)

    # Create our simulation input
    @hs.sim
    class ComparatorSim:
        tb = ComparatorTb(params)
        tr = hs.Tran(tstop=12 * n, tstep=100 * PICO)

    # Add the PDK dependencies
    ComparatorSim.lib(sky130.install.model_lib, "tt")
    ComparatorSim.literal(".option METHOD=Gear")

    # Run Spice, save important results
    results = ComparatorSim.run(sim_options)
    tran_results = results.an[0].data
    np.savez(
        "strongarm_results.npz",
        t=tran_results["time"],
        v_out_diff=tran_results["v(xtop.out_p)"] - tran_results["v(xtop.out_n)"],
        v_in_diff=tran_results["v(xtop.inp_p)"] - tran_results["v(xtop.inp_n)"],
        v_clk=tran_results["v(xtop.clk)"],
        i_tail=tran_results["i(v.xtop.xdut.xsa.vtail_meas)"],
        i_inp_pair_cm=tran_results["i(v.xtop.xdut.xsa.vninp_meas)"]
        + tran_results["i(v.xtop.xdut.xsa.vninn_meas)"],
        i_latch_n_pair_cm=tran_results["i(v.xtop.xdut.xsa.vnlatn_meas)"]
        + tran_results["i(v.xtop.xdut.xsa.vnlatp_meas)"],
        i_latch_p_pair_cm=tran_results["i(v.xtop.xdut.xsa.vplatn_meas)"]
        + tran_results["i(v.xtop.xdut.xsa.vplatp_meas)"],
        v_casc_cm=(
            tran_results["v(xtop.xdut.xsa.ninn_meas_p)"]
            + tran_results["v(xtop.xdut.xsa.ninp_meas_p)"]
        )
        / 2,
        v_out_cm=(
            tran_results["v(xtop.xdut.sout_p)"] + tran_results["v(xtop.xdut.sout_n)"]
        )
        / 2,
    )


def extract_windows(t, clk, threshold):
    """Given the clock waveform, this will extract a bunch of single periods
    that are the periods at which a certain clock starts and ends"""
    clk_above_thres_idcs = np.where(clk > threshold)[0]
    print(clk_above_thres_idcs.shape)
    print(np.diff(clk_above_thres_idcs))
    rising_cross_idcs = clk_above_thres_idcs[
        np.where(np.diff(np.concatenate(([0], clk_above_thres_idcs), axis=0)) > 1)[0]
    ]
    rising_cross_idcs = rising_cross_idcs.tolist() + [len(clk)]
    return [(s, e) for s, e in zip(rising_cross_idcs[:-1], rising_cross_idcs[1:])]


def plot_windows():
    data = np.load("strongarm_results.npz")
    windows = extract_windows(data["t"], data["v_clk"], 0.7)
    for (s, e) in windows:
        plt.figure()
        plt.plot(data["t"][s:e], data["v_clk"][s:e])
    plt.show()


def plot_data():
    data = np.load("strongarm_results.npz")
    t = data["t"]
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, data["v_clk"])
    ax[1].plot(t, data["v_in_diff"])
    ax[2].plot(t, data["v_out_diff"])
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, data["v_clk"])
    ax[1].plot(t, data["i_tail"], label="tail current")
    ax[1].plot(t, data["i_inp_pair_cm"], label="input pair current")
    ax[1].plot(t, data["i_latch_n_pair_cm"], label="Latch NMOS current")
    ax[1].plot(t, data["i_latch_p_pair_cm"], label="Latch PMOS current")
    ax[1].legend()
    ax[2].plot(t, data["v_casc_cm"], label="v_casc_cm")
    ax[2].plot(t, data["v_out_cm"], label="v_out_cm")
    ax[2].legend()

    plt.show()
    breakpoint()


if __name__ == "__main__":
    # test_comparator_sim()
    # plot_data()
    plot_windows()
