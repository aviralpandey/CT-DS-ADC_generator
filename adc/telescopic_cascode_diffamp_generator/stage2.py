import pprint

import numpy as np
import matplotlib.pyplot as plt

import hdl21 as h
import hdl21.sim as hs
from hdl21 import Diff
from hdl21.prefix import m, f, n, PICO, G, KILO, Prefixed
from hdl21.primitives import Vdc, C, R, Vcvs

# Import the Hdl21 PDK package, and our "site" configuration of its installation
import sky130, sitepdks as _

# Local imports
from .dbstuff import query_db, query_db_for_vgs
from .shared import MosParams, nch, nch_lvt, pch, pch_lvt
from ..testutils import (
    Pvt,
    sim_options,
    DiffClkParams,
    DiffClkGen,
)


@h.paramclass
class Stage2AmpParams:
    in_params = h.Param(dtype=MosParams, desc="Input pair MOS params")
    load_params = h.Param(dtype=MosParams, desc="Load pair MOS params")
    in_type = h.Param(dtype=str, desc="Input MOS type")
    load_type = h.Param(dtype=str, desc="Load MOS type")
    voutcm_ideal = h.Param(dtype=Prefixed, desc="Ideal Output CM")
    v_load = h.Param(dtype=Prefixed, desc="Load MOS Bias")


@h.generator
def stage2_common_source_amplifier(params: Stage2AmpParams) -> h.Module:

    if params.in_type == "nch":
        mos_in = nch
    elif params.in_type == "nch_lvt":
        mos_in = nch_lvt
    elif params.in_type == "pch":
        mos_in = pch
    elif params.in_type == "pch_lvt":
        mos_in = pch_lvt

    if params.load_type == "nch":
        mos_load = nch
    elif params.load_type == "nch_lvt":
        mos_load = nch_lvt
    elif params.load_type == "pch":
        mos_load = pch
    elif params.load_type == "pch_lvt":
        mos_load = pch_lvt

    @h.module
    class Stage2Amplifier:

        VDD, VSS = h.Ports(2)
        v_in = Diff(port=True, role=Diff.Roles.SINK)
        v_out = Diff(port=True, role=Diff.Roles.SOURCE)
        v_load = h.Input()

        ## Stage 2 Input Devices
        m_in_p = mos_in(params.in_params)(g=v_in.n, s=VSS, d=v_out.p, b=VSS)
        m_in_n = mos_in(params.in_params)(g=v_in.p, s=VSS, d=v_out.n, b=VSS)

        ## Load Base Pair
        m_load_p = mos_load(params.load_params)(g=v_load, s=VDD, d=v_out.p, b=VDD)
        m_load_n = mos_load(params.load_params)(g=v_load, s=VDD, d=v_out.n, b=VDD)

    return Stage2Amplifier


@h.paramclass
class Stage2AmpTbParams:
    dut = h.Param(dtype=Stage2AmpParams, desc="DUT params")
    pvt = h.Param(
        dtype=Pvt, desc="Process, Voltage, and Temperature Parameters", default=Pvt()
    )
    vd = h.Param(dtype=h.Prefixed, desc="Differential Voltage (V)", default=1 * m)
    vc = h.Param(dtype=h.Prefixed, desc="Common-Mode Voltage (V)", default=1200 * m)
    cl = h.Param(dtype=h.Prefixed, desc="Load Cap (Single-Ended) (F)", default=100 * f)
    rcm = h.Param(
        dtype=h.Prefixed, desc="Common Mode Sensing Resistor (Ω)", default=1 * G
    )
    CMFB_gain = h.Param(
        dtype=h.Prefixed, desc="Common Mode Feedback Gain (V/V)", default=1 * KILO
    )


@h.generator
def Stage2AmpTbTran(params: Stage2AmpTbParams) -> h.Module:

    tb = h.sim.tb("Stage2AmplifierTbTran")
    tb.VDD = VDD = h.Signal()
    tb.vvdd = Vdc(Vdc.Params(dc=params.pvt.v, ac=(0 * m)))(p=VDD, n=tb.VSS)

    tb.v_load = h.Signal()

    tb.voutcm_ideal = h.Signal()
    tb.v_outcm_ideal_src = Vdc(Vdc.Params(dc=(params.dut.voutcm_ideal), ac=(0 * m)))(
        p=tb.voutcm_ideal, n=tb.VSS
    )

    # Input-driving balun
    tb.inp = Diff()
    tb.inpgen = DiffClkGen(
        DiffClkParams(
            period=1000 * n, delay=1 * n, vc=params.vc, vd=params.vd, trf=800 * PICO
        )
    )(ck=tb.inp, VSS=tb.VSS)

    # Output & Load Caps
    tb.out = Diff()
    tb.CMSense = h.Signal()
    Cload = C(C.Params(c=params.cl))
    Ccmfb = C(C.Params(c=100 * f))
    Rload = R(R.Params(r=params.rcm))
    tb.clp = Cload(p=tb.out.p, n=tb.VSS)
    tb.cln = Cload(p=tb.out.n, n=tb.VSS)
    tb.ccmfb = Ccmfb(p=tb.CMSense, n=tb.VSS)
    tb.rcmp = Rload(p=tb.out.p, n=tb.CMSense)
    tb.rcmn = Rload(p=tb.out.n, n=tb.CMSense)
    tb.cmfb_src = Vcvs(Vcvs.Params(gain=params.CMFB_gain))(
        p=tb.v_load, n=tb.VSS, cp=tb.CMSense, cn=tb.voutcm_ideal
    )

    # Create the Telescopic Amplifier DUT
    tb.dut = stage2_common_source_amplifier(params.dut)(
        v_in=tb.inp, v_out=tb.out, v_load=tb.v_load, VDD=VDD, VSS=tb.VSS
    )
    return tb


def stage2_common_source_amplifier_design_and_scale_amp(
    database_in, database_load, amp_specs, gen_params
):

    vdd = amp_specs["vdd"]
    vincm = amp_specs["vincm"]
    voutcm = amp_specs["voutcm"]
    vgs_res = gen_params["vgs_sweep_res"]
    gain_min = amp_specs["gain_min"]
    stage1_gain = amp_specs["stage1_gain"]
    bw_min = amp_specs["amp_bw_min"]
    in_type = amp_specs["in_type"]
    load_type = amp_specs["load_type"]
    lch_in = gen_params["lch_in"]
    lch_load = gen_params["lch_load"]
    load_scale_min = gen_params["load_scale_min"]
    load_scale_max = gen_params["load_scale_max"]
    load_scale_step = gen_params["load_scale_step"]
    rload = amp_specs["rload"]

    if in_type == "nch" or in_type == "nch_lvt":
        vs_in = 0
        vs_load = vdd
    else:
        vs_in = vdd
        vs_load = 0

    vbs_in = 0
    vbs_load = 0
    vgs_in = vincm - vs_in
    vds_in = voutcm - vs_in
    vds_load = voutcm - vs_load

    ids_in = query_db(database_in, "ids", in_type, lch_in, vbs_in, vgs_in, vds_in)
    gm_in = query_db(database_in, "gm", in_type, lch_in, vbs_in, vgs_in, vds_in)
    gds_in = query_db(database_in, "gds", in_type, lch_in, vbs_in, vgs_in, vds_in)
    cgg_in = query_db(database_in, "cgg", in_type, lch_in, vbs_in, vgs_in, vds_in)
    cdd_in = cgg_in * gen_params["cdd_cgg_ratio"]

    load_scale_count = int((load_scale_max - load_scale_min) / load_scale_step) + 1
    load_scale_list = np.linspace(
        load_scale_min, load_scale_max, load_scale_count, endpoint=True
    )

    metric_best = 0
    gain_max = 0
    bw_max = 0

    for lch_load in gen_params["lch_load"]:

        for load_scale in load_scale_list:

            ids_load = ids_in / load_scale
            try:
                vgs_load = query_db_for_vgs(
                    database_load,
                    load_type,
                    lch_load,
                    vbs_load,
                    vds_load,
                    ids=ids_load,
                    mode="ids",
                    scale=1,
                )
            except:
                continue

            gm_load = (
                query_db(
                    database_load,
                    "gm",
                    load_type,
                    lch_load,
                    vbs_load,
                    vgs_load,
                    vds_load,
                )
                * load_scale
            )
            gds_load = (
                query_db(
                    database_load,
                    "gds",
                    load_type,
                    lch_load,
                    vbs_load,
                    vgs_load,
                    vds_load,
                )
                * load_scale
            )
            cgg_load = (
                query_db(
                    database_load,
                    "cgg",
                    load_type,
                    lch_load,
                    vbs_load,
                    vgs_load,
                    vds_load,
                )
                * load_scale
            )
            cdd_load = cgg_load * gen_params["cdd_cgg_ratio"]

            gds = gds_load + gds_in + (1 / rload)
            cdd = cdd_in + cdd_load

            gain_cur = gm_in / gds
            bw_cur = gds / cdd / 2 / np.pi
            metric_cur = gain_cur * bw_cur / ids_in

            if load_type == "pch" or load_type == "pch_lvt":
                load_bias = vdd + vgs_load
            else:
                load_bias = vgs_load

            if gain_cur > gain_min and bw_cur > bw_min:
                if metric_cur > metric_best:
                    metric_best = metric_cur
                    best_op = dict(
                        Av=gain_cur,
                        bw=bw_cur,
                        load_scale=load_scale,
                        lch_load=lch_load,
                        gm_in=gm_in,
                        gm_load=gm_load,
                        gds=gds,
                        cdd=cdd,
                        metric=metric_best,
                        load_bias=load_bias,
                    )
                    print("New GBW/I Best = %f MHz/µA" % (round(metric_best / 1e12, 2)))
                    print("Updated Av Best = %f" % (gain_cur))
                    print("Updated BW Best = %f MHz" % (round(bw_cur / 1e6, 2)))
            if gain_cur > gain_max:
                gain_max = gain_cur
            if bw_cur > bw_max:
                bw_max = bw_cur
    print("2nd Stage Av Best = %f" % ((gain_max)))
    print("2nd Stage BW Best = %f MHz" % (round(bw_max / 1e6, 2)))

    vnoise_input_referred_max = amp_specs["vnoise_input_referred"]
    cload = amp_specs["cload"]
    vdd = amp_specs["vdd"]
    in_type = amp_specs["in_type"]
    k = 1.38e-23
    T = 300

    ibias = ids_in
    load_bias = best_op["load_bias"]
    gm_in = best_op["gm_in"]
    gamma_in = gen_params["gamma"]
    gamma_load = gen_params["gamma"]
    load_scale = best_op["load_scale"]
    Av = best_op["Av"]
    gds = best_op["gds"]
    cdd = best_op["cdd"]
    vnoise_squared_input_referred_max = vnoise_input_referred_max**2
    vnoise_squared_input_referred_per_scale = (
        4 * k * T * (gamma_in * gm_in + gamma_load * gm_load) / (gm_in**2)
    )
    scale_noise = max(
        1, vnoise_squared_input_referred_per_scale / vnoise_squared_input_referred_max
    )
    gbw_min = bw_min * Av * stage1_gain
    scale_bw = max(
        1, 2 * np.pi * gbw_min * cload / (gm_in - 2 * np.pi * gbw_min * (cdd + cgg_in))
    )
    print("2nd stage scale_noise:")
    pprint.pprint(scale_noise)
    print("2nd stage scale_bw:")
    pprint.pprint(scale_bw)

    scale_amp = max(scale_bw, scale_noise)

    vnoise_density_squared_input_referred = (
        vnoise_squared_input_referred_per_scale / scale_amp
    )
    vnoise_density_input_referred = np.sqrt(vnoise_density_squared_input_referred)

    amplifier_op = dict(
        gm=gm_in * scale_amp,
        gds=gds * scale_amp,
        cgg=cgg_in * scale_amp,
        cdd=cdd * scale_amp,
        vnoise_density_input=vnoise_density_input_referred,
        scale_in=scale_amp,
        scale_load=scale_amp * load_scale,
        load_bias=load_bias,
        lch_load=best_op["lch_load"],
        gain=Av,
        ibias=ibias * scale_amp,
        bw=(gds * scale_amp) / (2 * np.pi * (cload + (cdd * scale_amp))),
    )

    stage2AmpParams = Stage2AmpParams(
        in_params=MosParams(
            w=500 * m * int(np.round(amplifier_op["scale_in"])),
            l=gen_params["lch_in"],
            nf=int(np.round(amplifier_op["scale_in"])),
        ),
        load_params=MosParams(
            w=500 * m * int(np.round(amplifier_op["scale_load"])),
            l=amplifier_op["lch_load"],
            nf=int(np.round(amplifier_op["scale_load"])),
        ),
        in_type=amp_specs["in_type"],
        load_type=amp_specs["load_type"],
        voutcm_ideal=voutcm * 1000 * m,
        v_load=amplifier_op["load_bias"] * 1000 * m,
    )

    params = Stage2AmpTbParams(
        pvt=Pvt(),
        vc=vincm * 1000 * m,
        vd=1 * m,
        dut=stage2AmpParams,
        cl=amp_specs["cload"] * 1000 * m,
    )

    # Create our simulation input
    @hs.sim
    class Stage2AmpTranSim:
        tb = Stage2AmpTbTran(params)
        tr = hs.Tran(tstop=3000 * n, tstep=1 * n)

    # Add the PDK dependencies
    Stage2AmpTranSim.lib(sky130.install.model_lib, "tt")
    Stage2AmpTranSim.literal(".save all")
    results = Stage2AmpTranSim.run(sim_options)
    tran_results = results.an[0].data
    t = tran_results["time"]
    v_out_diff = tran_results["v(xtop.out_p)"] - tran_results["v(xtop.out_n)"]
    v_out_cm = (tran_results["v(xtop.out_p)"] + tran_results["v(xtop.out_n)"]) / 2
    v_in_diff = tran_results["v(xtop.inp_p)"] - tran_results["v(xtop.inp_n)"]
    v_in_cm = (tran_results["v(xtop.inp_p)"] + tran_results["v(xtop.inp_n)"]) / 2
    v_load = tran_results["v(xtop.v_load)"]

    # v_out_stage1_diff = tran_results['v(xtop.out_stage1_p)'] - tran_results['v(xtop.out_stage1_n)']

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_in_diff)
    ax[1].plot(t, v_out_diff)

    plt.show()

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_in_cm)
    ax[1].plot(t, v_out_cm)

    plt.show()

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_load)
    ax[1].plot(t, v_out_cm)

    plt.show()

    return stage2AmpParams, amplifier_op


def generate_stage2_common_source_amplifier(amp_specs):
    nch_db_filename = "database_nch.npy"
    nch_lvt_db_filename = "database_nch_lvt.npy"
    pch_db_filename = "database_pch.npy"
    pch_lvt_db_filename = "database_pch_lvt.npy"

    gen_params = dict(
        tail_vstar_vds_margin=50e-3,
        vgs_sweep_res=5e-3,
        vds_sweep_res=15e-3,
        load_scale_min=0.25,
        load_scale_max=2,
        load_scale_step=0.25,
        gamma=1,
        cdd_cgg_ratio=1.1,
        lch_in=0.5,
        lch_load=[0.35, 0.5, 1],
    )

    if amp_specs["in_type"] == "nch":
        in_db_filename = nch_db_filename
    elif amp_specs["in_type"] == "nch_lvt":
        in_db_filename = nch_lvt_db_filename
    elif amp_specs["in_type"] == "pch":
        in_db_filename = pch_db_filename
    elif amp_specs["in_type"] == "pch_lvt":
        in_db_filename = pch_lvt_db_filename

    if amp_specs["load_type"] == "nch":
        load_db_filename = nch_db_filename
    elif amp_specs["load_type"] == "nch_lvt":
        load_db_filename = nch_lvt_db_filename
    elif amp_specs["load_type"] == "pch":
        load_db_filename = pch_db_filename
    elif amp_specs["load_type"] == "pch_lvt":
        load_db_filename = pch_lvt_db_filename

    database_in = np.load(in_db_filename, allow_pickle=True).item()
    database_load = np.load(load_db_filename, allow_pickle=True).item()
    ampParams, amplifier_op = stage2_common_source_amplifier_design_and_scale_amp(
        database_in, database_load, amp_specs, gen_params
    )

    print("2nd stage op:")
    pprint.pprint(amplifier_op)

    return ampParams, amplifier_op
