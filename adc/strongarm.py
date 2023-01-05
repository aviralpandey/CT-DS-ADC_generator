import hdl21 as h
from hdl21 import Diff
from hdl21.primitives import Vdc


@h.paramclass
class MosParams:
    w = h.Param(dtype=float, desc="Channel Width")
    l = h.Param(dtype=float, desc="Channel Length")
    nf = h.Param(dtype=int, desc="Number of fingers")


nch = h.ExternalModule(
    name="sky130_fd_pr__nfet_01v8",
    desc="Sky130 NMOS",
    port_list=[
        h.Inout(name="d"),
        h.Inout(name="g"),
        h.Inout(name="s"),
        h.Inout(name="b"),
    ],
    paramtype=MosParams,
)
pch = h.ExternalModule(
    name="sky130_fd_pr__pfet_01v8",
    desc="Sky130 PMOS",
    port_list=[
        h.Inout(name="d"),
        h.Inout(name="g"),
        h.Inout(name="s"),
        h.Inout(name="b"),
    ],
    paramtype=MosParams,
)


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
        dtype=bool, desc="True to add voltage sources to measure device currents"
    )


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
        """# StrongArm Based Comparator"""

        VDD, VSS = h.Ports(2)
        inp = Diff(port=True, role=Diff.Roles.SINK)
        out = Diff(port=True, role=Diff.Roles.SOURCE)
        clk = h.Input()

        sout = Diff()

        sa = strongarm(params.strongarm)(inp=inp, out=sout, clk=clk, VDD=VDD, VSS=VSS)
        sr = sr_latch(params.latch)(inp=sout, out=out, VDD=VDD, VSS=VSS)

    return Comparator
