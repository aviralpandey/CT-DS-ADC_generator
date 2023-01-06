import numpy as np
import matplotlib.pyplot as plt

import hdl21 as h
import hdl21.sim as hs
from hdl21 import Diff
from hdl21.prefix import m, f, n, PICO, G, KILO, Prefixed, FEMTO
from hdl21.primitives import Vdc, C, R, Vcvs

# PDK Imports
import sky130, sitepdks as _

# Local imports
from .shared import MosParams, nch, nch_lvt, pch, pch_lvt
from ..testutils import (
    Pvt,
    sim_options,
    DiffClkParams,
    DiffClkGen,
)
from .stage1 import (
    TelescopicAmpParams,
    generate_stage1_telescopic_amplifier,
    stage1_telescopic_amplifier,
)
from .stage2 import generate_stage2_common_source_amplifier


@h.paramclass
class TwoStageAmpParams:
    stage1_tail_params = h.Param(dtype=MosParams, desc="Stage 1 Tail MOS params")
    stage1_in_base_pair_params = h.Param(
        dtype=MosParams, desc="Stage 1 Input pair MOS params"
    )
    stage1_in_casc_pair_params = h.Param(
        dtype=MosParams, desc="Stage 1 Input cascode MOS params"
    )
    stage1_load_base_pair_params = h.Param(
        dtype=MosParams, desc="Stage 1 Load base MOS params"
    )
    stage1_load_casc_pair_params = h.Param(
        dtype=MosParams, desc="Stage 1 Load cascode MOS params"
    )
    stage1_in_type = h.Param(dtype=str, desc="Stage 1 Input MOS type")
    stage1_load_type = h.Param(dtype=str, desc="Stage 1 Load MOS type")
    stage1_tail_type = h.Param(dtype=str, desc="Stage 1 Tail MOS type")
    stage1_voutcm_ideal = h.Param(dtype=Prefixed, desc="Stage 1 Ideal Output CM")
    stage1_v_load = h.Param(dtype=Prefixed, desc="Stage 1 Load MOS Bias")
    stage1_v_pcasc = h.Param(dtype=Prefixed, desc="Stage 1 PMOS Cascode Device Bias")
    stage1_v_ncasc = h.Param(dtype=Prefixed, desc="Stage 1 NMOS Cascode Device Bias")
    stage1_v_tail = h.Param(dtype=Prefixed, desc="Stage 1 Tail MOS Bias")
    stage2_in_type = h.Param(dtype=str, desc="Stage 2 Input MOS type")
    stage2_load_type = h.Param(dtype=str, desc="Stage 2 Load MOS type")
    stage2_in_params = h.Param(dtype=MosParams, desc="Stage 2 Input MOS params")
    stage2_load_params = h.Param(dtype=MosParams, desc="Stage 2 Load MOS params")
    stage2_v_load = h.Param(dtype=Prefixed, desc="Stage 2 Load MOS Bias")
    stage2_voutcm_ideal = h.Param(dtype=Prefixed, desc="Stage 2 Ideal Output CM")
    c_comp = h.Param(dtype=Prefixed, desc="Compensation C Value")
    r_comp = h.Param(dtype=Prefixed, desc="Compensation R Value")


@h.generator
def two_stage_amplifier(params: TwoStageAmpParams) -> h.Module:

    if params.stage2_in_type == "nch":
        stage2_mos_in = nch
    elif params.stage2_in_type == "nch_lvt":
        stage2_mos_in = nch_lvt
    elif params.stage2_in_type == "pch":
        stage2_mos_in = pch
    elif params.stage2_in_type == "pch_lvt":
        stage2_mos_in = pch_lvt

    if params.stage2_load_type == "nch":
        stage2_mos_load = nch
    elif params.stage2_load_type == "nch_lvt":
        stage2_mos_load = nch_lvt
    elif params.stage2_load_type == "pch":
        stage2_mos_load = pch
    elif params.stage2_load_type == "pch_lvt":
        stage2_mos_load = pch_lvt

    stage1Params = TelescopicAmpParams(
        in_base_pair_params=params.stage1_in_base_pair_params,
        in_casc_pair_params=params.stage1_in_casc_pair_params,
        load_base_pair_params=params.stage1_load_base_pair_params,
        load_casc_pair_params=params.stage1_load_casc_pair_params,
        tail_params=params.stage1_tail_params,
        in_type=params.stage1_in_type,
        load_type=params.stage1_load_type,
        tail_type=params.stage1_tail_type,
        voutcm_ideal=params.stage1_voutcm_ideal,
        v_load=params.stage1_v_load,
        v_pcasc=params.stage1_v_pcasc,
        v_ncasc=params.stage1_v_ncasc,
        v_tail=params.stage1_v_tail,
    )

    @h.module
    class TwoStageAmplifier:

        VDD, VSS = h.Ports(2)
        v_in = Diff(port=True, role=Diff.Roles.SINK)
        v_out = Diff(port=True, role=Diff.Roles.SOURCE)
        v_out_stage1 = Diff(port=True, role=Diff.Roles.SOURCE)
        v_comp_mid_p = h.Signal()
        v_comp_mid_n = h.Signal()
        v_tail_stage1 = h.Input()
        v_load_stage1 = h.Input()
        v_pcasc_stage1 = h.Input()
        v_ncasc_stage1 = h.Input()
        v_load_stage2 = h.Input()

        x_stage1 = stage1_telescopic_amplifier(stage1Params)(
            v_in=v_in,
            v_out=v_out_stage1,
            v_tail=v_tail_stage1,
            v_ncasc=v_ncasc_stage1,
            v_pcasc=v_pcasc_stage1,
            v_load=v_load_stage1,
            VDD=VDD,
            VSS=VSS,
        )

        C_comp_p = C(C.Params(c=params.c_comp))(p=v_out.p, n=v_comp_mid_p)
        C_comp_n = C(C.Params(c=params.c_comp))(p=v_out.n, n=v_comp_mid_n)
        R_comp_p = R(R.Params(r=params.r_comp))(p=v_comp_mid_p, n=v_out_stage1.n)
        R_comp_n = R(R.Params(r=params.r_comp))(p=v_comp_mid_n, n=v_out_stage1.p)

        ## Stage 2 Input Devices
        m_in_stage2_p = stage2_mos_in(params.stage2_in_params)(
            g=v_out_stage1.n, s=VSS, d=v_out.p, b=VSS
        )
        m_in_stage2_n = stage2_mos_in(params.stage2_in_params)(
            g=v_out_stage1.p, s=VSS, d=v_out.n, b=VSS
        )

        ## Load Base Pair
        m_load_stage2_p = stage2_mos_load(params.stage2_load_params)(
            g=v_load_stage2, s=VDD, d=v_out.p, b=VDD
        )
        m_load_stage2_n = stage2_mos_load(params.stage2_load_params)(
            g=v_load_stage2, s=VDD, d=v_out.n, b=VDD
        )

    return TwoStageAmplifier


@h.paramclass
class TwoStageTbParams:
    dut = h.Param(dtype=TwoStageAmpParams, desc="DUT params")
    pvt = h.Param(
        dtype=Pvt, desc="Process, Voltage, and Temperature Parameters", default=Pvt()
    )
    vd = h.Param(dtype=h.Prefixed, desc="Differential Voltage (V)", default=1 * m)
    vc = h.Param(dtype=h.Prefixed, desc="Common-Mode Voltage (V)", default=1200 * m)
    cl = h.Param(dtype=h.Prefixed, desc="Load Cap (Single-Ended) (F)", default=100 * f)
    ccm = h.Param(
        dtype=h.Prefixed, desc="Common Mode Sensing Capacitor (F)", default=1 * f
    )
    rcm = h.Param(
        dtype=h.Prefixed, desc="Common Mode Sensing Resistor (Ω)", default=1 * G
    )
    CMFB_gain_stage1 = h.Param(
        dtype=h.Prefixed, desc="Common Mode Feedback Gain (V/V)", default=10 * KILO
    )
    CMFB_gain_stage2 = h.Param(
        dtype=h.Prefixed, desc="Common Mode Feedback Gain (V/V)", default=1 * KILO
    )


@h.generator
def TwoStageAmpTbTran(params: TwoStageTbParams) -> h.Module:

    tb = h.sim.tb("TwoStageAmplifierTbTran")
    tb.VDD = VDD = h.Signal()
    tb.vvdd = Vdc(Vdc.Params(dc=params.pvt.v, ac=(0 * m)))(p=VDD, n=tb.VSS)

    tb.v_tail_stage1 = h.Signal()
    tb.v_tail_stage1_src = Vdc(Vdc.Params(dc=(params.dut.stage1_v_tail), ac=(0 * m)))(
        p=tb.v_tail_stage1, n=tb.VSS
    )

    tb.v_pcasc_stage1 = h.Signal()
    tb.v_pcasc_stage1_src = Vdc(Vdc.Params(dc=(params.dut.stage1_v_pcasc), ac=(0 * m)))(
        p=tb.v_pcasc_stage1, n=tb.VSS
    )

    tb.v_ncasc_stage1 = h.Signal()
    tb.v_ncasc_stage1_src = Vdc(Vdc.Params(dc=(params.dut.stage1_v_ncasc), ac=(0 * m)))(
        p=tb.v_ncasc_stage1, n=tb.VSS
    )

    tb.v_load_stage1 = h.Signal()

    tb.v_load_stage2 = h.Signal()

    tb.voutcm_stage1_ideal = h.Signal()
    tb.v_outcm_stage1_ideal_src = Vdc(
        Vdc.Params(dc=(params.dut.stage1_voutcm_ideal), ac=(0 * m))
    )(p=tb.voutcm_stage1_ideal, n=tb.VSS)

    tb.voutcm_stage2_ideal = h.Signal()
    tb.v_outcm_stage2_ideal_src = Vdc(
        Vdc.Params(dc=(params.dut.stage2_voutcm_ideal), ac=(0 * m))
    )(p=tb.voutcm_stage2_ideal, n=tb.VSS)

    # Input-driving balun
    tb.inp = Diff()
    tb.inpgen = DiffClkGen(
        DiffClkParams(
            period=1000 * n, delay=1 * n, vc=params.vc, vd=params.vd, trf=800 * PICO
        )
    )(ck=tb.inp, VSS=tb.VSS)

    # Output & Load Caps
    tb.out = Diff()
    tb.out_stage1 = Diff()
    tb.CMSense_stage1 = h.Signal()
    tb.CMSense_stage2 = h.Signal()
    Cload = C(C.Params(c=params.cl))
    Ccmfb = C(C.Params(c=params.ccm))
    Rload = R(R.Params(r=params.rcm))
    tb.clp = Cload(p=tb.out.p, n=tb.VSS)
    tb.cln = Cload(p=tb.out.n, n=tb.VSS)
    tb.ccmfb_stage1 = Ccmfb(p=tb.CMSense_stage1, n=tb.VSS)
    tb.ccmfb_stage2 = Ccmfb(p=tb.CMSense_stage2, n=tb.VSS)
    tb.rcmp_stage1 = Rload(p=tb.out_stage1.p, n=tb.CMSense_stage1)
    tb.rcmn_stage1 = Rload(p=tb.out_stage1.n, n=tb.CMSense_stage1)
    tb.rcmp_stage2 = Rload(p=tb.out.p, n=tb.CMSense_stage2)
    tb.rcmn_stage2 = Rload(p=tb.out.n, n=tb.CMSense_stage2)
    tb.cmfb_stage1_src = Vcvs(Vcvs.Params(gain=params.CMFB_gain_stage1))(
        p=tb.v_load_stage1, n=tb.VSS, cp=tb.CMSense_stage1, cn=tb.voutcm_stage1_ideal
    )
    tb.cmfb_stage2_src = Vcvs(Vcvs.Params(gain=params.CMFB_gain_stage2))(
        p=tb.v_load_stage2, n=tb.VSS, cp=tb.CMSense_stage2, cn=tb.voutcm_stage2_ideal
    )

    # Create the Telescopic Amplifier DUT
    tb.dut = two_stage_amplifier(params.dut)(
        v_in=tb.inp,
        v_out=tb.out,
        v_out_stage1=tb.out_stage1,
        v_tail_stage1=tb.v_tail_stage1,
        v_ncasc_stage1=tb.v_ncasc_stage1,
        v_pcasc_stage1=tb.v_pcasc_stage1,
        v_load_stage1=tb.v_load_stage1,
        v_load_stage2=tb.v_load_stage2,
        VDD=VDD,
        VSS=tb.VSS,
    )
    return tb


@h.generator
def TwoStageAmpTbAc(params: TwoStageTbParams) -> h.Module:

    tb = h.sim.tb("TwoStageAmplifierTbTran")
    tb.VDD = VDD = h.Signal()
    tb.vvdd = Vdc(Vdc.Params(dc=params.pvt.v, ac=(0 * m)))(p=VDD, n=tb.VSS)

    tb.v_tail_stage1 = h.Signal()
    tb.v_tail_stage1_src = Vdc(Vdc.Params(dc=(params.dut.stage1_v_tail), ac=(0 * m)))(
        p=tb.v_tail_stage1, n=tb.VSS
    )

    tb.v_pcasc_stage1 = h.Signal()
    tb.v_pcasc_stage1_src = Vdc(Vdc.Params(dc=(params.dut.stage1_v_pcasc), ac=(0 * m)))(
        p=tb.v_pcasc_stage1, n=tb.VSS
    )

    tb.v_ncasc_stage1 = h.Signal()
    tb.v_ncasc_stage1_src = Vdc(Vdc.Params(dc=(params.dut.stage1_v_ncasc), ac=(0 * m)))(
        p=tb.v_ncasc_stage1, n=tb.VSS
    )

    tb.v_load_stage1 = h.Signal()

    tb.v_load_stage2 = h.Signal()

    tb.voutcm_stage1_ideal = h.Signal()
    tb.v_outcm_stage1_ideal_src = Vdc(
        Vdc.Params(dc=(params.dut.stage1_voutcm_ideal), ac=(0 * m))
    )(p=tb.voutcm_stage1_ideal, n=tb.VSS)

    tb.voutcm_stage2_ideal = h.Signal()
    tb.v_outcm_stage2_ideal_src = Vdc(
        Vdc.Params(dc=(params.dut.stage2_voutcm_ideal), ac=(0 * m))
    )(p=tb.voutcm_stage2_ideal, n=tb.VSS)

    # Input-driving balun
    tb.inp = Diff()
    tb.inpgen = Vdc(Vdc.Params(dc=(params.vc), ac=(500 * m)))(p=tb.inp.p, n=tb.VSS)
    tb.inngen = Vdc(Vdc.Params(dc=(params.vc), ac=(-500 * m)))(p=tb.inp.n, n=tb.VSS)

    # Output & Load Caps
    tb.out = Diff()
    tb.out_stage1 = Diff()
    tb.CMSense_stage1 = h.Signal()
    tb.CMSense_stage2 = h.Signal()
    Cload = C(C.Params(c=params.cl))
    Ccmfb = C(C.Params(c=params.ccm))
    Rload = R(R.Params(r=params.rcm))
    tb.clp = Cload(p=tb.out.p, n=tb.VSS)
    tb.cln = Cload(p=tb.out.n, n=tb.VSS)
    tb.ccmfb_stage1 = Ccmfb(p=tb.CMSense_stage1, n=tb.VSS)
    tb.ccmfb_stage2 = Ccmfb(p=tb.CMSense_stage2, n=tb.VSS)
    tb.rcmp_stage1 = Rload(p=tb.out_stage1.p, n=tb.CMSense_stage1)
    tb.rcmn_stage1 = Rload(p=tb.out_stage1.n, n=tb.CMSense_stage1)
    tb.rcmp_stage2 = Rload(p=tb.out.p, n=tb.CMSense_stage2)
    tb.rcmn_stage2 = Rload(p=tb.out.n, n=tb.CMSense_stage2)
    tb.cmfb_stage1_src = Vcvs(Vcvs.Params(gain=params.CMFB_gain_stage1))(
        p=tb.v_load_stage1, n=tb.VSS, cp=tb.CMSense_stage1, cn=tb.voutcm_stage1_ideal
    )
    tb.cmfb_stage2_src = Vcvs(Vcvs.Params(gain=params.CMFB_gain_stage2))(
        p=tb.v_load_stage2, n=tb.VSS, cp=tb.CMSense_stage2, cn=tb.voutcm_stage2_ideal
    )

    # Create the Telescopic Amplifier DUT
    tb.dut = two_stage_amplifier(params.dut)(
        v_in=tb.inp,
        v_out=tb.out,
        v_out_stage1=tb.out_stage1,
        v_tail_stage1=tb.v_tail_stage1,
        v_ncasc_stage1=tb.v_ncasc_stage1,
        v_pcasc_stage1=tb.v_pcasc_stage1,
        v_load_stage1=tb.v_load_stage1,
        v_load_stage2=tb.v_load_stage2,
        VDD=VDD,
        VSS=tb.VSS,
    )
    return tb


def generate_two_stage_ota():
    two_stage_amp_specs = dict(
        stage1_in_type="nch_lvt",
        stage1_tail_type="nch_lvt",
        stage1_load_type="pch",
        stage1_vstar_in=200e-3,
        stage1_vincm=1,
        stage1_voutcm=0.9,
        stage1_vds_tail_min=0.25,
        stage1_gain_min=100,
        stage1_input_stage_gain_min=200,
        stage1_bw_min=20e6,
        stage2_in_type="nch",
        stage2_load_type="pch",
        stage2_voutcm=1,
        bw_min=5e5,
        gain_min=500,
        vnoise_input_referred=10e-9,
        cload=1000e-15,
        rload=100e3,
        vdd=1.8,
    )

    stage1_telescopic_amp_specs = dict(
        in_type=two_stage_amp_specs["stage1_in_type"],
        tail_type=two_stage_amp_specs["stage1_tail_type"],
        load_type=two_stage_amp_specs["stage1_load_type"],
        cload=100e-15,
        vdd=two_stage_amp_specs["vdd"],
        vstar_in=two_stage_amp_specs["stage1_vstar_in"],
        vincm=two_stage_amp_specs["stage1_vincm"],
        voutcm=two_stage_amp_specs["stage1_voutcm"],
        vds_tail_min=two_stage_amp_specs["stage1_vds_tail_min"],
        gain_min=two_stage_amp_specs["stage1_gain_min"],
        input_stage_gain_min=two_stage_amp_specs["stage1_input_stage_gain_min"],
        selfbw_min=two_stage_amp_specs["stage1_bw_min"],
        bw_min=two_stage_amp_specs["stage1_bw_min"],
        vnoise_input_referred=two_stage_amp_specs["vnoise_input_referred"]
        * (
            two_stage_amp_specs["gain_min"]
            / (
                two_stage_amp_specs["stage1_input_stage_gain_min"]
                + two_stage_amp_specs["gain_min"]
            )
        ),
    )

    (
        telescopic_amp_params,
        telescopic_amp_results,
    ) = generate_stage1_telescopic_amplifier(stage1_telescopic_amp_specs)

    stage2_common_source_amp_specs = dict(
        in_type=two_stage_amp_specs["stage2_in_type"],
        load_type=two_stage_amp_specs["stage2_load_type"],
        cload=two_stage_amp_specs["cload"],
        rload=100e3,
        vdd=two_stage_amp_specs["vdd"],
        vincm=two_stage_amp_specs["stage1_voutcm"],
        voutcm=two_stage_amp_specs["stage2_voutcm"],
        stage1_gain=telescopic_amp_results["Av"],
        gain_min=two_stage_amp_specs["gain_min"] / telescopic_amp_results["Av"],
        amp_bw_min=two_stage_amp_specs["bw_min"],
        vnoise_input_referred=two_stage_amp_specs["vnoise_input_referred"]
        * (
            telescopic_amp_results["Av"]
            / (telescopic_amp_results["Av"] + two_stage_amp_specs["gain_min"])
        )
        * telescopic_amp_results["Av"],
    )

    stage2_params, stage2_op = generate_stage2_common_source_amplifier(
        stage2_common_source_amp_specs
    )

    twoStageAmpParams = TwoStageAmpParams(
        stage1_in_base_pair_params=telescopic_amp_params.in_base_pair_params,
        stage1_in_casc_pair_params=telescopic_amp_params.in_casc_pair_params,
        stage1_load_base_pair_params=telescopic_amp_params.load_base_pair_params,
        stage1_load_casc_pair_params=telescopic_amp_params.load_casc_pair_params,
        stage1_tail_params=telescopic_amp_params.tail_params,
        stage1_in_type=telescopic_amp_params.in_type,
        stage1_load_type=telescopic_amp_params.load_type,
        stage1_tail_type=telescopic_amp_params.tail_type,
        stage1_voutcm_ideal=telescopic_amp_params.voutcm_ideal,
        stage1_v_load=telescopic_amp_params.v_load,
        stage1_v_pcasc=telescopic_amp_params.v_pcasc,
        stage1_v_ncasc=telescopic_amp_params.v_ncasc,
        stage1_v_tail=telescopic_amp_params.v_tail,
        stage2_in_type=stage2_params.in_type,
        stage2_load_type=stage2_params.load_type,
        stage2_in_params=stage2_params.in_params,
        stage2_load_params=stage2_params.load_params,
        stage2_v_load=stage2_params.v_load,
        stage2_voutcm_ideal=stage2_params.voutcm_ideal,
        c_comp=250 * FEMTO,
        r_comp=1 * KILO,
    )

    params = TwoStageTbParams(
        pvt=Pvt(),
        vc=two_stage_amp_specs["stage1_vincm"] * 1000 * m,
        vd=1 * m,
        dut=twoStageAmpParams,
        cl=two_stage_amp_specs["cload"] * 1000 * m,
    )

    # Create our simulation input
    @hs.sim
    class TwoStageAmpTranSim:
        tb = TwoStageAmpTbTran(params)
        tr = hs.Tran(tstop=3000 * n, tstep=1 * n)

    # Add the PDK dependencies
    TwoStageAmpTranSim.lib(sky130.install.model_lib, "tt")
    TwoStageAmpTranSim.literal(".save all")
    results = TwoStageAmpTranSim.run(sim_options)
    tran_results = results.an[0].data
    t = tran_results["time"]
    v_out_diff = tran_results["v(xtop.out_p)"] - tran_results["v(xtop.out_n)"]
    v_out_cm = (tran_results["v(xtop.out_p)"] + tran_results["v(xtop.out_n)"]) / 2
    v_in_diff = tran_results["v(xtop.inp_p)"] - tran_results["v(xtop.inp_n)"]
    v_in_cm = (tran_results["v(xtop.inp_p)"] + tran_results["v(xtop.inp_n)"]) / 2

    v_out_stage1_diff = (
        tran_results["v(xtop.out_stage1_p)"] - tran_results["v(xtop.out_stage1_n)"]
    )
    v_out_stage1_cm = (
        tran_results["v(xtop.out_stage1_p)"] + tran_results["v(xtop.out_stage1_n)"]
    ) / 2

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_in_diff)
    ax[1].plot(t, v_out_diff)

    plt.show()

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_in_cm)
    ax[1].plot(t, v_out_cm)

    plt.show()

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_out_stage1_cm)
    ax[1].plot(t, v_out_stage1_diff)

    plt.show()

    # Create our simulation input
    @hs.sim
    class TwoStageAmpAcSim:
        tb = TwoStageAmpTbAc(params)
        myac = hs.Ac(sweep=hs.LogSweep(1e1, 1e11, 10))

    # Add the PDK dependencies
    TwoStageAmpAcSim.lib(sky130.install.model_lib, "tt")
    TwoStageAmpAcSim.literal(".save all")
    results = TwoStageAmpAcSim.run(sim_options)
    ac_results = results.an[0].data
    v_out_diff_ac = ac_results["v(xtop.out_p)"] - ac_results["v(xtop.out_n)"]
    f = np.logspace(start=1, stop=11, num=101, endpoint=True)
    f_ax = np.logspace(start=1, stop=11, num=11, endpoint=True)
    f3db_idx = np.squeeze(
        np.where(abs(v_out_diff_ac) < np.max(abs(v_out_diff_ac)) / np.sqrt(2))
    )[0]
    fgbw_idx = np.squeeze(np.where(abs(v_out_diff_ac) < 1))[0]
    f3db = f[f3db_idx]
    fgbw = f[fgbw_idx]
    Avdc = np.max(abs(v_out_diff_ac))
    v_out_diff_ac_dB = 20 * np.log10(abs(v_out_diff_ac))
    v_out_diff_ac_phase = (
        (np.angle(v_out_diff_ac) % (2 * np.pi) - 2 * np.pi) * 180 / np.pi
    )
    PM_ac = 180 + v_out_diff_ac_phase[fgbw_idx]
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].semilogx(f, v_out_diff_ac_dB)
    ax[0].grid(color="black", linestyle="--", linewidth=0.5)
    ax[0].grid(visible=True, which="both")
    ax[0].set_xticks(f_ax)
    ax[1].semilogx(f, v_out_diff_ac_phase)
    ax[1].grid(color="black", linestyle="--", linewidth=0.5)
    ax[1].grid(visible=True, which="both")
    ax[1].set_xticks(f_ax)
    plt.show()
    print("Av (AC Sim) = %f" % ((Avdc)))
    print("BW (AC Sim) = %f MHz" % (round(f3db / 1e6, 2)))
    print("GBW (AC Sim) = %f MHz" % (round(fgbw / 1e6, 2)))
    print("PM (AC Sim) = %f degrees" % PM_ac)

    ampResults = dict(
        Av=Avdc,
        bw=f3db,
        gbw=fgbw,
        pm=PM_ac,
    )


# if __name__ == "__main__":
#     generate_two_stage_ota()
#     breakpoint()
