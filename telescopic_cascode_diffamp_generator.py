# -*- coding: utf-8 -*-

import pprint

import numpy as np
import scipy.interpolate
import scipy.optimize as sciopt
import matplotlib.pyplot as plt

import os
from pathlib import Path

import hdl21 as h
import hdl21.sim as hs
from hdl21 import Diff
from hdl21.pdk import Corner
from hdl21.sim import Sim, LogSweep
from hdl21.prefix import m, µ, f, n, PICO, G, KILO, Prefixed
from hdl21.primitives import Vdc, Idc, C, Vpulse, R, Vcvs
from vlsirtools.spice import SimOptions, SupportedSimulators, ResultFormat

import nest_asyncio
nest_asyncio.apply()

@h.paramclass
class MosParams:
    w = h.Param(dtype=float, desc="Channel Width")
    l = h.Param(dtype=float, desc="Channel Length")
    nf = h.Param(dtype=int, desc="Number of fingers")

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", None)

sim_options = SimOptions(
    rundir=Path("./scratch2"),
    fmt=ResultFormat.SIM_DATA,
    simulator=SupportedSimulators.NGSPICE,
)

nch = h.ExternalModule(
    name="sky130_fd_pr__nfet_01v8", desc="Sky130 NMOS", 
    port_list=[h.Inout(name="d"), h.Inout(name="g"), h.Inout(name="s"), h.Inout(name="b")], 
    paramtype=MosParams)
nch_lvt = h.ExternalModule(
    name="sky130_fd_pr__nfet_01v8_lvt", desc="Sky130 NMOS LVT", 
    port_list=[h.Inout(name="d"), h.Inout(name="g"), h.Inout(name="s"), h.Inout(name="b")], 
    paramtype=MosParams)
pch = h.ExternalModule(
    name="sky130_fd_pr__pfet_01v8", desc="Sky130 PMOS", 
    port_list=[h.Inout(name="d"), h.Inout(name="g"), h.Inout(name="s"), h.Inout(name="b")], 
    paramtype=MosParams)
pch_lvt = h.ExternalModule(
    name="sky130_fd_pr__pfet_01v8_lvt", desc="Sky130 PMOS LVT", 
    port_list=[h.Inout(name="d"), h.Inout(name="g"), h.Inout(name="s"), h.Inout(name="b")], 
    paramtype=MosParams)

def query_db(database,varname,mos_type,lch,vbs,vgs,vds):
    
    mos_list = database['mos_list']
    mos_index = mos_list.index(mos_type)
    lch_list = database['lch_list']
    lch_index = lch_list.index(lch)
    var_raw = database[varname][mos_index,lch_index,:,:,:]
    if mos_type == "nch" or mos_type == "nch_lvt":
        vgs_raw = np.array(database['vgs_list'])
        vds_raw = np.array(database['vds_list'])
        vbs_raw = np.array(database['vbs_list'])
    else:
        vgs_raw = -np.array(database['vgs_list'])
        vds_raw = -np.array(database['vds_list'])
        vbs_raw = -np.array(database['vbs_list'])
    
    interp = scipy.interpolate.RegularGridInterpolator((vbs_raw,vgs_raw,vds_raw),var_raw)
    return interp([vbs,vgs,vds]).item()

def query_db_for_vgs(database,mos_type,lch,vbs,vds,vstar=200e-3,ids=0,mode='vstar',scale=1):
    mos_list = database['mos_list']
    mos_index = mos_list.index(mos_type)
    lch_list = database['lch_list']
    lch_index = lch_list.index(lch)
    if mos_type == "nch" or mos_type == "nch_lvt":
        vgs_raw = np.array(database['vgs_list'])
        vds_raw = np.array(database['vds_list'])
        vbs_raw = np.array(database['vbs_list'])
    else:
        vgs_raw = -np.array(database['vgs_list'])
        vds_raw = -np.array(database['vds_list'])
        vbs_raw = -np.array(database['vbs_list'])
        
    ids_raw = database['ids'][mos_index,lch_index,:,:,:]*scale
    gm_raw = database['gm'][mos_index,lch_index,:,:,:]*scale
    
    if mode == 'vstar':
        target_gm_id = 2/vstar
        gm_id_raw = gm_raw/ids_raw
        interp = scipy.interpolate.RegularGridInterpolator((vbs_raw,vgs_raw,vds_raw),gm_id_raw)
        
        resample_vector_vbs = [vbs]*(np.shape(vgs_raw)[0])
        resample_vector_vds = [vds]*(np.shape(vgs_raw)[0])
        resample_points = np.transpose(np.vstack((resample_vector_vbs,vgs_raw,resample_vector_vds)))
        gm_id_raw_1d = interp(resample_points)
        
        interp_vgs = scipy.interpolate.interp1d(gm_id_raw_1d,vgs_raw)
        
        try:
            return interp_vgs(target_gm_id).item()
        except:
            raise ValueError
    
    elif mode == 'ids':
        interp = scipy.interpolate.RegularGridInterpolator((vbs_raw,vgs_raw,vds_raw),ids_raw)
        
        resample_vector_vbs = [vbs]*(np.shape(vgs_raw)[0])
        resample_vector_vds = [vds]*(np.shape(vgs_raw)[0])
        resample_points = np.transpose(np.vstack((resample_vector_vbs,vgs_raw,resample_vector_vds)))
        ids_raw_1d = interp(resample_points)
        
        interp_vgs = scipy.interpolate.interp1d(ids_raw_1d,vgs_raw)
        
        try:
            return interp_vgs(ids).item()
        except:
            raise ValueError

def query_db_for_function(database,varname,mos_type,lch):
    
    mos_list = database['mos_list']
    mos_index = mos_list.index(mos_type)
    lch_list = database['lch_list']
    lch_index = lch_list.index(lch)
    
    var_raw = database[varname][mos_index,lch_index,:,:,:]
    if mos_type == "nch" or mos_type == "nch_lvt":
        vgs_raw = np.array(database['vgs_list'])
        vds_raw = np.array(database['vds_list'])
        vbs_raw = np.array(database['vbs_list'])
    else:
        vgs_raw = -np.array(database['vgs_list'])
        vds_raw = -np.array(database['vds_list'])
        vbs_raw = -np.array(database['vbs_list'])

    interp = scipy.interpolate.RegularGridInterpolator((vbs_raw,vgs_raw,vds_raw),var_raw)
    return interp
    

def stage1_telescopic_amplifier_design_input(database,amp_specs,gen_params):

    """Find operating point that meets the given vstar spec."""
    vdd = amp_specs['vdd']
    voutcm = amp_specs['voutcm']
    vincm = amp_specs['vincm']
    vstar_in = amp_specs['vstar_in']
    vdst_min = amp_specs['vds_tail_min']
    vds_sweep_res = gen_params['vds_sweep_res']
    casc_scale_max = gen_params['casc_scale_max']
    casc_scale_min = gen_params['casc_scale_min']
    casc_scale_step = gen_params['casc_scale_step']
    in_type = amp_specs['in_type']
    lch_in = gen_params['lch_in']

    num_casc_scale_points = int(np.ceil((casc_scale_max - casc_scale_min) / casc_scale_step)) + 1
    casc_scale_list = np.linspace(casc_scale_min, casc_scale_max, num_casc_scale_points, endpoint=True)
    
    if in_type == 'nch' or in_type == 'nch_lvt':
        vb = 0
        vtail_lim = vdst_min
        #vcasc_mid_lim = vtail_lim + vstar_in
        vds_in_lim_0 = vstar_in
        vds_in_lim_1 = (voutcm-vtail_lim)*2/3
        num_vds_points = int(np.ceil((vds_in_lim_1 - vds_in_lim_0) / vds_sweep_res)) + 1
        vds_in_val_list = np.linspace(vds_in_lim_0, vds_in_lim_1, num_vds_points, endpoint=True)
        vgs_lim_0  = 0.1
        vgs_lim_1  = vincm - vtail_lim
        
    else:
        vb = vdd
        vtail_lim = vb - vdst_min
        #vcasc_mid_lim = vtail_lim - vstar_in
        vds_in_lim_0 = -vstar_in
        vds_in_lim_1 = (voutcm-vtail_lim)*2/3
        num_vds_points = int(np.ceil((vds_in_lim_0 - vds_in_lim_1) / vds_sweep_res)) + 1
        vds_in_val_list = np.linspace(vds_in_lim_0, vds_in_lim_1, num_vds_points, endpoint=True)
        vgs_lim_0  = -0.1
        vgs_lim_1  = vincm - vtail_lim

    ib_fun = query_db_for_function(database,'ids',in_type,lch_in)
    gm_fun = query_db_for_function(database,'gm',in_type,lch_in)
   
    def vgs_in_search_fun(vgs_in, vcasc_mid, vsource):
        vds_in = vcasc_mid - vsource
        vbs_in = vb - vsource
        vstar_comp = (2 * ib_fun([vbs_in,vgs_in,vds_in])/gm_fun([vbs_in,vgs_in,vds_in]))
        return vstar_comp - vstar_in
    
    
    vgs_best = None
    casc_scale_best = None
    Av_best = 0 
    metric_best = 0
    
    for vds_in in vds_in_val_list:
        
        # First iteration to find approximate vgs (and hence vsource) to account for VBS later
        vcasc_mid = vdst_min + vds_in
        vsource = vdst_min
        try:
            vgs_in = sciopt.brentq(vgs_in_search_fun, vgs_lim_0, vgs_lim_1, args=(vcasc_mid,vsource,))
        except ValueError:
            continue 
        vsource = vincm - vgs_in
        vcasc_mid = vsource + vds_in
        
        # Second iteration, for more exact result with approximately the correct VBS
        try:
            vgs_in = sciopt.brentq(vgs_in_search_fun, vgs_lim_0, vgs_lim_1, args=(vcasc_mid,vsource,))
        except ValueError:
            continue
        vsource = vincm - vgs_in
        vcasc_mid = vsource + vds_in
        vbs_in = vb - vsource
        vds_in = vcasc_mid - vsource
        ids_in = ib_fun([vbs_in,vgs_in,vds_in])[0]
        vbs_casc = (vb - vcasc_mid)
        vds_casc = (voutcm - vcasc_mid)
        vds_in = (vcasc_mid - vsource)
        
        gm_in = query_db(database,'gm',in_type,lch_in,vbs_in,vgs_in,vds_in)
        for casc_scale in casc_scale_list:
            ids_casc = ids_in/casc_scale
            try: 
                vgs_casc = query_db_for_vgs(database=database, mos_type = in_type, lch = lch_in, vbs = vbs_casc, vds = vds_casc, ids=ids_casc, mode='ids')
            except:
                continue
            vg_casc = vcasc_mid + vgs_casc
            if vg_casc > vdd or vg_casc < 0:
                continue
            
            gm_casc = query_db(database,'gm',in_type,lch_in,vbs_casc,vgs_casc,vds_casc)*casc_scale
            gds_casc = query_db(database,'gds',in_type,lch_in,vbs_casc,vgs_casc,vds_casc)*casc_scale
            gds_base = query_db(database,'gds',in_type,lch_in,vbs_in,vgs_in,vds_in)
                
            gds_in = (gds_base * gds_casc / (gds_base + gds_casc + gm_casc))
            Av_cur = gm_in / gds_in
            
            cgg_casc = query_db(database, 'cgg',in_type,lch_in,vbs_casc,vgs_casc,vds_casc)*casc_scale
            cgg_in = query_db(database,'cgg',in_type,lch_in,vbs_in,vgs_in,vds_in)
            
            bw_cur = gm_in / (cgg_casc * gen_params['cdd_cgg_ratio'])
            metric_cur = Av_cur * bw_cur
            
            if Av_cur > (amp_specs['input_stage_gain_min']) and metric_cur > metric_best:
                metric_best = metric_cur
                Av_best = Av_cur
                vgs_best = vgs_in
                casc_scale_best = casc_scale
                input_op = dict(
                    Av = Av_cur,
                    vgs = vgs_in,
                    casc_scale = casc_scale,
                    casc_bias = vg_casc,
                    vin_mid = vcasc_mid,
                    ibias = ids_in,
                    gm_in = gm_in,
                    gm_casc = gm_casc,
                    gds_base = gds_in,
                    gds_casc = gds_casc,
                    gds_in = gds_in,
                    cdd_in = cgg_casc * gen_params['cdd_cgg_ratio'],
                    cgg_casc = cgg_casc,
                    cgg_base = cgg_in,
                    vtail = vsource,
                    )
                
                print("New Av Best = %f" % (Av_best))
                print("Updated VGS Best = %f" % (vgs_best))
                print("Updated Casc Scale Best = %f" % (casc_scale_best))
    
    print("--------------------------------")
    print("Input Stage Design:")
    print("Av Best = %f" % (Av_best))
    print("VGS Best = %f" % (vgs_best))
    print("Casc Scale Best = %f" % (casc_scale_best))
    print("VDS_in Best = %f" % (input_op['vin_mid'] - input_op['vtail']))
    print("VDS_casc Best = %f" % (voutcm - input_op['vin_mid']))
    print("--------------------------------")
    
    return input_op

def stage1_telescopic_amplifier_design_load(database,amp_specs,gen_params,input_op):
    """Design load.

    Sweep vgs.  For each vgs, compute gain and max bandwidth.  If
    both gain and BW specs are met, pick operating point that maximizes
    gamma_r * gm_r
    """
    vdd = amp_specs['vdd']
    vstar_in = amp_specs['vstar_in']
    voutcm = amp_specs['voutcm']
    vgs_res = gen_params['vgs_sweep_res']
    gain_min = amp_specs['gain_min']
    bw_min = max(amp_specs['selfbw_min'],amp_specs['bw_min'])
    casc_scale_max = gen_params['casc_scale_max']
    casc_scale_step = gen_params['casc_scale_step']
    casc_bias_step = gen_params['casc_bias_step']
    vds_sweep_res = gen_params['vds_sweep_res']
    in_type = amp_specs['in_type']
    load_type = amp_specs['load_type']
    lch_load = gen_params['lch_load']
    best_load_op = None
    metric_best = 0
    gain_max = 0
    bw_max = 0
    
    casc_scale_list = np.arange(1, casc_scale_max + casc_scale_step / 2, casc_scale_step)
    if in_type == 'nch' or in_type == 'nch_lvt':
        vs = vdd
        vb = vdd
        casc_bias_list = np.arange(0.5, voutcm + casc_bias_step / 2, casc_bias_step)
        vgs_base_max = -0.1
        vgs_base_min = -1.5
        vds_base_lim_0 = -vstar_in
        vds_base_lim_1 = (voutcm-vdd)*2/3
        num_vds_points = int(np.ceil((vds_base_lim_0 - vds_base_lim_1) / vds_sweep_res)) + 1
        vds_base_val_list = np.linspace(vds_base_lim_0, vds_base_lim_1, num_vds_points, endpoint=True)
    else:
        vs = 0
        vb = 0
        casc_bias_list = np.arange(voutcm, vdd - 0.5 + casc_bias_step / 2, casc_bias_step)
        vgs_base_min = 0.1
        vgs_base_max = 1.5
        vds_base_lim_0 = vstar_in
        vds_base_lim_1 = (voutcm)*2/3
        num_vds_points = int(np.ceil((vds_base_lim_1 - vds_base_lim_0) / vds_sweep_res)) + 1
        vds_base_val_list = np.linspace(vds_base_lim_0, vds_base_lim_1, num_vds_points, endpoint=True)
        
    for lch_load in gen_params['lch_load']:
        gm_fun = query_db_for_function(database,'gm',load_type,lch_load)
        gds_fun = query_db_for_function(database,'gds',load_type,lch_load)
        cgg_fun = query_db_for_function(database,'cgg',load_type,lch_load)
        gamma = gen_params['gamma']
        ib_fun = query_db_for_function(database,'ids',load_type,lch_load)
        
        num_points = int(np.ceil((vgs_base_max - vgs_base_min) / vgs_res)) + 1
        
        gm_in_base = input_op['gm_in']
        gm_in_casc = input_op['gm_casc']
        gds_in_base = input_op['gds_in']
        gds_in_casc = input_op['gds_casc']
        ibias_in = input_op['ibias']
        gds_in = input_op['gds_in']
        cgg_in = input_op['cgg_casc']
        cdd_in = cgg_in * gen_params['cdd_cgg_ratio']
        
        def vgs_base_search_fun(vgs_base, vcasc_mid, casc_scale, vg_casc, vsource):
            vgs_casc = vg_casc - vcasc_mid
            vds_casc = voutcm - vcasc_mid
            vbs_casc = vb - vcasc_mid
            vds_base = vcasc_mid - vsource
            vbs_base = 0
            
            ids_casc = ib_fun([vbs_casc,vgs_casc,vds_casc])[0]*casc_scale
            ids_base = ib_fun([vbs_base,vgs_base,vds_base])[0]
            
            return ids_casc - ids_base
        
        
        
        for vds_base in vds_base_val_list:
            vcasc_mid = vs + vds_base
            for casc_scale in casc_scale_list:
                for casc_bias in casc_bias_list:
                    try:
                        vgs_base = sciopt.brentq(vgs_base_search_fun, vgs_base_min, vgs_base_max, args=(vcasc_mid,casc_scale,casc_bias,vs,))
                    except ValueError:
                        continue
                    vgs_casc = casc_bias - vcasc_mid
                    vds_casc = voutcm - vcasc_mid
                    vbs_casc = vb - vcasc_mid
                    vds_base = vcasc_mid - vs
                    vbs_base = 0
                    
                    ibias_base = ib_fun([vbs_base,vgs_base,vds_base])[0]
                    load_scale = ibias_in/ibias_base
                    
                    gm_base = gm_fun([vbs_base,vgs_base,vds_base])[0]
                    gds_base = gds_fun([vbs_base,vgs_base,vds_base])[0]
                    cdd_base = cgg_fun([vbs_base,vgs_base,vds_base])[0] * gen_params['cdd_cgg_ratio']
                    
                    gm_casc = gm_fun([vbs_casc,vgs_casc,vds_casc])[0] * casc_scale
                    gds_casc = gds_fun([vbs_casc,vgs_casc,vds_casc])[0] * casc_scale
                    cdd_casc = cgg_fun([vbs_casc,vgs_casc,vds_casc])[0] * gen_params['cdd_cgg_ratio'] * casc_scale
                    
                    gm_load = gm_base * load_scale
                    gds_load = gds_base * gds_casc / (gds_base + gds_casc + gm_casc) * load_scale
                    cdd_load = cdd_casc * load_scale
                    
                    bw_cur = (gds_load + gds_in) / (cdd_load + cdd_in) / 2 / np.pi
                    gain_cur = gm_in_base / (gds_load + gds_in)
                    
                    metric_cur = gain_cur * bw_cur / ibias_in
                    
                    if load_type == 'pch':
                        base_bias = vdd + vgs_base
                    else:
                        base_bias = vgs_base
                    
                    if gain_cur > gain_min and bw_cur > bw_min:
                        if metric_cur > metric_best:
                            metric_best = metric_cur
                            best_load_op = dict(
                                Av = gain_cur,
                                bw = bw_cur,
                                casc_scale = casc_scale,
                                casc_bias = casc_bias,
                                base_bias = base_bias,
                                vload_mid = vcasc_mid,
                                load_scale = load_scale,
                                lch_load = lch_load,
                                gm_load = gm_load,
                                gds_load = gds_load,
                                cdd_load = cdd_load,
                                metric_load = metric_best,
                                )
                            print("New GBW/I Best = %f MHz/µA" % (round(metric_best/1e12,2)))
                            print("Updated Av Best = %f" % (gain_cur))
                            print("Updated BW Best = %f MHz" % (round(bw_cur/1e6,2)))
                    if gain_cur > gain_max:
                        gain_max = gain_cur
                    if bw_cur > bw_max:
                        bw_max = bw_cur
    print("Load Av Best = %f" % ((gain_max)))
    print("Load BW Best = %f MHz" % (round(bw_max/1e6,2)))
    return best_load_op

def stage1_telescopic_amplifier_design_amp(amp_specs, gen_params, input_op, load_op):

    vnoise_input_referred_max = amp_specs['vnoise_input_referred']
    bw_min = amp_specs['bw_min']
    cload = amp_specs['cload']
    vdd = amp_specs['vdd']
    in_type = amp_specs['in_type']
    k = 1.38e-23
    T = 300

    ibias = input_op['ibias']
    vtail = input_op['vtail']
    gm_in = input_op['gm_in']
    gds_in = input_op['gds_in']
    gds_in_casc = input_op['gds_casc']
    gds_in_base = input_op['gds_base']
    gm_in_casc = input_op['gm_casc']
    gamma_in = gen_params['gamma']
    cdd_in = input_op['cdd_in']
    cgg_in = input_op['cgg_base']
    gm_load = load_op['gm_load']
    gds_load = load_op['gds_load']
    cdd_load = load_op['cdd_load']
    gamma_load = gen_params['gamma']
    load_scale = load_op['load_scale']
    load_casc_scale = load_op['casc_scale']
    load_casc_bias = load_op['casc_bias']
    load_base_bias = load_op['base_bias']
    in_casc_scale = input_op['casc_scale']
    in_casc_bias = input_op['casc_bias']
    Av = load_op['Av']
    
    gds_tot = gds_in + gds_load
    cdd_tot = cdd_in + cdd_load
    vnoise_squared_input_referred_max = vnoise_input_referred_max ** 2
    vnoise_squared_input_referred_per_scale = (4*k*T*(gamma_in * gm_in + gamma_load * gm_load) / (gm_in**2))
    scale_noise = max(1, vnoise_squared_input_referred_per_scale/vnoise_squared_input_referred_max)
    scale_bw = max(1, 2 * np.pi * bw_min * cload / (gds_tot - 2 * np.pi * bw_min * cdd_tot))
    print('scale_noise:')
    pprint.pprint(scale_noise)
    print('scale_bw:')
    pprint.pprint(scale_bw)
    
    scale_amp = max(scale_bw,scale_noise)
    
    vnoise_density_squared_input_referred = vnoise_squared_input_referred_per_scale / scale_amp
    vnoise_density_input_referred = np.sqrt(vnoise_density_squared_input_referred)
    
    amplifier_op = dict(
            gm = gm_in * scale_amp,
            gds = gds_tot * scale_amp,
            cgg = cgg_in * scale_amp,
            cdd = cdd_tot * scale_amp,
            vnoise_density_input = vnoise_density_input_referred,
            scale_in_base = scale_amp,
            scale_in_casc = scale_amp * in_casc_scale,
            scale_load_base = scale_amp * load_scale,
            scale_load_casc = scale_amp * load_scale * load_casc_scale,
            vtail = vtail,
            load_base_bias = load_base_bias,
            load_casc_bias = load_casc_bias,
            in_casc_bias = in_casc_bias,
            ibias = ibias * scale_amp * 2,
            gain = Av,
            bw = (gds_tot * scale_amp) / (2 * np.pi * (cload + (cdd_tot * scale_amp))),
        )
    return amplifier_op
    
def stage1_telescopic_amplifier_design_tail(database, amp_specs, gen_params, amplifier_op):
    
    vdd = amp_specs['vdd']
    vtail = amplifier_op['vtail']
    vstar_tail = vtail - gen_params['tail_vstar_vds_margin']
    in_type = amp_specs['in_type']
    
    if in_type == 'nch' or in_type == 'nch_lvt':
        vds_tail = vtail
        vbs_tail = 0
    else:
        vds_tail = vtail - vdd
        vbs_tail = 0
    
    lch_tail = gen_params['lch_tail']
    
    ib_fun = query_db_for_function(database,'ids',in_type,lch_tail)
    itarg = amplifier_op['ibias']
    
    vgs_tail = query_db_for_vgs(database=database, mos_type = in_type, lch = lch_tail, vbs = vbs_tail, vds = vds_tail, vstar=vstar_tail, mode='vstar')

    ids_tail = ib_fun([vbs_tail,vgs_tail,vds_tail])[0]
    scale_tail = itarg/ids_tail
    
    gm_tail = query_db(database,'gm',in_type,lch_tail,vbs_tail,vgs_tail,vds_tail)
    gds_tail = query_db(database,'gds',in_type,lch_tail,vbs_tail,vgs_tail,vds_tail)
    cgg_tail = query_db(database,'cgg',in_type,lch_tail,vbs_tail,vgs_tail,vds_tail)
    cdd_tail = cgg_tail * gen_params['cdd_cgg_ratio']
    gmro_tail = gm_tail/gds_tail
    
    if in_type == 'nch' or in_type == 'nch_lvt':
        vbias_tail = vgs_tail
    else:
        vbias_tail = vdd + vgs_tail
        
    tail_op = dict(
        scale_tail = scale_tail,
        vbias_tail = vbias_tail,
        vgs = vgs_tail,
        vds = vds_tail,
        gm = gm_tail,
        gds = gds_tail,
        gmro = gmro_tail,
        cdd = cdd_tail,
        )
    return tail_op

@h.paramclass
class TelescopicAmpParams:
    tail_params = h.Param(dtype=MosParams, desc="Tail FET params")
    in_base_pair_params = h.Param(dtype=MosParams, desc="Input pair FET params")
    in_casc_pair_params = h.Param(dtype=MosParams, desc="Inverter NMos params")
    load_base_pair_params = h.Param(dtype=MosParams, desc="Inverter PMos params")
    load_casc_pair_params = h.Param(dtype=MosParams, desc="Reset Device params")
    in_type = h.Param(dtype=str, desc="Input MOS type")
    load_type = h.Param(dtype=str, desc="Load MOS type")
    tail_type = h.Param(dtype=str, desc="Tail MOS type")
    voutcm_ideal = h.Param(dtype=Prefixed, desc="Ideal Output CM")
    v_load = h.Param(dtype=Prefixed, desc="Load MOS Bias")
    v_pcasc = h.Param(dtype=Prefixed, desc="PMOS Cascode Device Bias")
    v_ncasc = h.Param(dtype=Prefixed, desc="NMOS Cascode Device Bias")
    v_tail = h.Param(dtype=Prefixed, desc="Tail MOS Bias")


@h.generator
def create_module(params: TelescopicAmpParams) -> h.Module:
    
    if params.in_type == 'nch':
        mos_in = nch
    elif params.in_type == 'nch_lvt':
        mos_in = nch_lvt
    elif params.in_type == 'pch':
        mos_in = pch
    elif params.in_type == 'pch_lvt':
        mos_in = pch_lvt
        
    if params.load_type == 'nch':
        mos_load = nch
    elif params.load_type == 'nch_lvt':
        mos_load = nch_lvt
    elif params.load_type == 'pch':
        mos_load = pch
    elif params.load_type == 'pch_lvt':
        mos_load = pch_lvt
        
    if params.tail_type == 'nch':
        mos_tail = nch
    elif params.tail_type == 'nch_lvt':
        mos_tail = nch_lvt
    elif params.tail_type == 'pch':
        mos_tail = pch
    elif params.tail_type == 'pch_lvt':
        mos_tail = pch_lvt
        
    if params.in_type == 'nch' or params.in_type == 'nch_lvt':
        @h.module
        class TelescopicAmp:
            
            VDD, VSS = h.Ports(2)
            v_in = Diff(port=True, role=Diff.Roles.SINK)
            v_out = Diff(port=True, role=Diff.Roles.SOURCE)
            v_tail = h.Input()
            v_load = h.Input()
            v_pcasc = h.Input()
            v_ncasc = h.Input()
            
            ## Tail Device
            m_tail = mos_tail(params.tail_params)(g=v_tail,s=VSS,b=VSS)
            
            ## Input Base Pair
            m_in_base_p = mos_in(params.in_base_pair_params)(g=v_in.p, s=m_tail.d , b=VSS)
            m_in_base_n = mos_in(params.in_base_pair_params)(g=v_in.n, s=m_tail.d , b=VSS)
            
            ## Input Cascode Pair
            m_in_casc_p = mos_in(params.in_casc_pair_params)(g=v_ncasc, s=m_in_base_p.d, d=v_out.n, b=VSS)
            m_in_casc_n = mos_in(params.in_casc_pair_params)(g=v_ncasc, s=m_in_base_n.d, d=v_out.p, b=VSS)
            
            ## Load Base Pair
            m_load_base_p = mos_load(params.load_base_pair_params)(g=v_load, s=VDD, b=VDD)
            m_load_base_n = mos_load(params.load_base_pair_params)(g=v_load, s=VDD, b=VDD)
            
            ## Load Cascode Pair
            m_load_casc_p = mos_load(params.load_casc_pair_params)(g=v_pcasc, s=m_load_base_p.d, d=v_out.n, b=VDD)
            m_load_casc_n = mos_load(params.load_casc_pair_params)(g=v_pcasc, s=m_load_base_n.d, d=v_out.p, b=VDD)
            
        return TelescopicAmp
    
    elif params.load_type == 'pch' or params.load_type == 'pch_lvt':
        @h.module
        class TelescopicAmp:
            VDD, VSS = h.Ports(2)
            v_in = Diff(port=True, role=Diff.Roles.SINK)
            v_out = Diff(port=True, role=Diff.Roles.SOURCE)
            v_tail = h.Input()
            v_load = h.Input()
            v_pcasc = h.Input()
            v_ncasc = h.Input()
            
            ## Tail Device
            m_tail = mos_tail(params.tail_params)(g=v_tail,s=VDD,b=VDD)
            
            ## Input Base Pair
            m_in_base_p = mos_in(params.in_base_pair_params)(g=v_in.p, s=m_tail.d , b=VDD)
            m_in_base_n = mos_in(params.in_base_pair_params)(g=v_in.n, s=m_tail.d , b=VDD)
            
            ## Input Cascode Pair
            m_in_casc_p = mos_in(params.in_casc_pair_params)(g=v_pcasc, s=m_in_base_p.d, d=v_out.n, b=VDD)
            m_in_casc_n = mos_in(params.in_casc_pair_params)(g=v_pcasc, s=m_in_base_n.d, d=v_out.p, b=VDD)
            
            ## Load Base Pair
            m_load_base_p = mos_load(params.load_base_pair_params)(g=v_load, s=VSS, b=VSS)
            m_load_base_n = mos_load(params.load_base_pair_params)(g=v_load, s=VSS, b=VSS)
            
            ## Load Cascode Pair
            m_load_casc_p = mos_load(params.load_casc_pair_params)(g=v_ncasc, s=m_load_base_p.d, d=v_out.n, b=VSS)
            m_load_casc_n = mos_load(params.load_casc_pair_params)(g=v_ncasc, s=m_load_base_n.d, d=v_out.p, b=VSS)
            
        return TelescopicAmp

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
    dut = h.Param(dtype=TelescopicAmpParams, desc="DUT params")
    pvt = h.Param(
        dtype=Pvt, desc="Process, Voltage, and Temperature Parameters", default=Pvt()
    )
    vd = h.Param(dtype=h.Prefixed, desc="Differential Voltage (V)", default=1 * m)
    vc = h.Param(dtype=h.Prefixed, desc="Common-Mode Voltage (V)", default=1200 * m)
    cl = h.Param(dtype=h.Prefixed, desc="Load Cap (Single-Ended) (F)", default=100 * f)
    rcm = h.Param(dtype=h.Prefixed, desc="Common Mode Sensing Resistor (Ω)", default = 1 * G)
    CMFB_gain = h.Param(dtype=h.Prefixed, desc="Common Mode Feedback Gain (V/V)", default = 1 * KILO)

@h.generator
def AmplifierTbTran(params: TbParams) -> h.Module:
    
    tb = h.sim.tb("TelescopicAmplifierTb")
    tb.VDD = VDD = h.Signal()
    tb.vvdd = Vdc(Vdc.Params(dc=params.pvt.v,ac=(0*m)))(p=VDD, n=tb.VSS)
    
    tb.v_tail = h.Signal()
    #tb.v_tail_src = Vdc(Vdc.Params(dc=(params.dut.v_tail),ac=(0*m)))(p=tb.v_tail, n=tb.VSS)
    
    tb.v_pcasc = h.Signal()
    tb.v_pcasc_src = Vdc(Vdc.Params(dc=(params.dut.v_pcasc),ac=(0*m)))(p=tb.v_pcasc, n=tb.VSS)
    
    tb.v_ncasc = h.Signal()
    tb.v_ncasc_src = Vdc(Vdc.Params(dc=(params.dut.v_ncasc),ac=(0*m)))(p=tb.v_ncasc, n=tb.VSS)
    
    tb.v_load = h.Signal()
    tb.voutcm_ideal = h.Signal()
    tb.v_load_src = Vdc(Vdc.Params(dc=(params.dut.v_load)))(p=tb.v_load, n=tb.VSS)
    tb.v_outcm_ideal_src = Vdc(Vdc.Params(dc=(params.dut.voutcm_ideal),ac=(0*m)))(p=tb.voutcm_ideal, n=tb.VSS)
    
    # Input-driving balun
    tb.inp = Diff()
    tb.inpgen = DiffClkGen(
        DiffClkParams(period= 100 * n, delay=1 * n, vc=params.vc, vd=params.vd, trf=800 * PICO)
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
    tb.cmfb_src = Vcvs(Vcvs.Params(gain=params.CMFB_gain))(p=tb.v_tail, n=tb.VSS, cp=tb.CMSense, cn=tb.voutcm_ideal)
    
    # Create the Telescopic Amplifier DUT
    tb.dut = create_module(params.dut)(
        v_in=tb.inp,
        v_out=tb.out,
        v_tail=tb.v_tail,
        v_ncasc=tb.v_ncasc,
        v_pcasc=tb.v_pcasc,
        v_load=tb.v_load,
        VDD=VDD,
        VSS=tb.VSS)
    return tb

@h.generator
def AmplifierTbAc(params: TbParams) -> h.Module:
    
    tb = h.sim.tb("TelescopicAmplifierTb")
    tb.VDD = VDD = h.Signal()
    tb.vvdd = Vdc(Vdc.Params(dc=params.pvt.v,ac=(0*m)))(p=VDD, n=tb.VSS)
    
    tb.v_tail = h.Signal()
    #tb.v_tail_src = Vdc(Vdc.Params(dc=(params.dut.v_tail),ac=(0*m)))(p=tb.v_tail, n=tb.VSS)
    
    tb.v_pcasc = h.Signal()
    tb.v_pcasc_src = Vdc(Vdc.Params(dc=(params.dut.v_pcasc),ac=(0*m)))(p=tb.v_pcasc, n=tb.VSS)
    
    tb.v_ncasc = h.Signal()
    tb.v_ncasc_src = Vdc(Vdc.Params(dc=(params.dut.v_ncasc),ac=(0*m)))(p=tb.v_ncasc, n=tb.VSS)
    
    tb.v_load = h.Signal()
    tb.voutcm_ideal = h.Signal()
    tb.v_load_src = Vdc(Vdc.Params(dc=(params.dut.v_load)))(p=tb.v_load, n=tb.VSS)
    tb.v_outcm_ideal_src = Vdc(Vdc.Params(dc=(params.dut.voutcm_ideal),ac=(0*m)))(p=tb.voutcm_ideal, n=tb.VSS)
    
    # Input-driving balun
    tb.inp = Diff()
    tb.inpgen = Vdc(Vdc.Params(dc=(params.vc),ac=(500*m)))(p=tb.inp.p, n=tb.VSS)
    tb.inngen = Vdc(Vdc.Params(dc=(params.vc),ac=(-500*m)))(p=tb.inp.n, n=tb.VSS)
    
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
    tb.cmfb_src = Vcvs(Vcvs.Params(gain=params.CMFB_gain))(p=tb.v_tail, n=tb.VSS, cp=tb.CMSense, cn=tb.voutcm_ideal)
    
    # Create the Telescopic Amplifier DUT
    tb.dut = create_module(params.dut)(
        v_in=tb.inp,
        v_out=tb.out,
        v_tail=tb.v_tail,
        v_ncasc=tb.v_ncasc,
        v_pcasc=tb.v_pcasc,
        v_load=tb.v_load,
        VDD=VDD,
        VSS=tb.VSS)
    return tb
    
    
def generate_stage1_telescopic_amplifier(amp_specs):

    nch_db_filename = "database_nch.npy"
    nch_lvt_db_filename = "database_nch_lvt.npy"
    pch_db_filename = "database_pch.npy"
    pch_lvt_db_filename = "database_pch_lvt.npy"
    
    gen_params = dict(
        tail_vstar_vds_margin = 50e-3,
        vgs_sweep_res = 5e-3,
        vds_sweep_res = 15e-3,
        casc_scale_min = 0.5,
        casc_scale_max = 3,
        casc_scale_step = 0.5,
        casc_bias_step = 10e-3,
        gamma = 1,
        cdd_cgg_ratio = 1.1,
        lch_in = 0.15,
        lch_tail = 0.5,
        lch_load = [0.15, 0.5, 1],
        )
    
    if amp_specs['in_type'] == 'nch':
        in_db_filename = nch_db_filename
    elif amp_specs['in_type'] == 'nch_lvt':
        in_db_filename = nch_lvt_db_filename
    elif amp_specs['in_type'] == 'pch':
        in_db_filename = pch_db_filename
    elif amp_specs['in_type'] == 'pch_lvt':
        in_db_filename = pch_lvt_db_filename
        
    if amp_specs['load_type'] == 'nch':
        load_db_filename = nch_db_filename
    elif amp_specs['load_type'] == 'nch_lvt':
        load_db_filename = nch_lvt_db_filename
    elif amp_specs['load_type'] == 'pch':
        load_db_filename = pch_db_filename
    elif amp_specs['load_type'] == 'pch_lvt':
        load_db_filename = pch_lvt_db_filename
        
    if amp_specs['tail_type'] == 'nch':
        tail_db_filename = nch_db_filename
    elif amp_specs['tail_type'] == 'nch_lvt':
        tail_db_filename = nch_lvt_db_filename
    elif amp_specs['tail_type'] == 'pch':
        tail_db_filename = pch_db_filename
    elif amp_specs['tail_type'] == 'pch_lvt':
        tail_db_filename = pch_lvt_db_filename
        
    database_in = np.load(in_db_filename,allow_pickle=True).item()
    database_tail = np.load(tail_db_filename,allow_pickle=True).item()
    database_load = np.load(load_db_filename,allow_pickle=True).item()
    
    amp_specs['gm_id_in'] = 2/amp_specs['vstar_in']

    input_op = stage1_telescopic_amplifier_design_input(database_in,amp_specs,gen_params)
    load_op = stage1_telescopic_amplifier_design_load(database_load,amp_specs,gen_params,input_op)
    print('input op:')
    pprint.pprint(input_op)
    print('load op:')
    pprint.pprint(load_op)
    amplifier_op = stage1_telescopic_amplifier_design_amp(amp_specs,gen_params,input_op,load_op)
    print('amplifier op:')
    pprint.pprint(amplifier_op)
    tail_op = stage1_telescopic_amplifier_design_tail(database_tail,amp_specs, gen_params, amplifier_op)
    print('tail op:')
    pprint.pprint(tail_op)
    
    if amp_specs['tail_type'] == 'nch' or amp_specs['tail_type'] == 'nch_lvt':
        v_load = amplifier_op['load_base_bias']
        v_pcasc = amplifier_op['load_casc_bias']
        v_ncasc = amplifier_op['in_casc_bias']
        v_tail = tail_op['vbias_tail']
    else:
        v_load = amplifier_op['load_base_bias']
        v_ncasc = amplifier_op['load_casc_bias']
        v_pcasc = amplifier_op['in_casc_bias']
        v_tail = tail_op['vbias_tail']
    
    ampParams=TelescopicAmpParams(
        in_base_pair_params = MosParams(w= 500 * m * int(np.round(amplifier_op['scale_in_base'])), l=gen_params['lch_in'], nf=int(np.round(amplifier_op['scale_in_base']))),
        in_casc_pair_params = MosParams(w= 500 * m * int(np.round(amplifier_op['scale_in_casc'])), l=gen_params['lch_in'], nf=int(np.round(amplifier_op['scale_in_casc']))),
        load_base_pair_params = MosParams(w= 500 * m * int(np.round(amplifier_op['scale_load_base'])), l=load_op['lch_load'], nf=int(np.round(amplifier_op['scale_load_base']))),
        load_casc_pair_params = MosParams(w= 500 * m * int(np.round(amplifier_op['scale_load_casc'])), l=load_op['lch_load'], nf=int(np.round(amplifier_op['scale_load_casc']))),
        tail_params = MosParams(w= 500 * m * int(np.round(tail_op['scale_tail'])), l=gen_params['lch_tail'], nf=int(np.round(tail_op['scale_tail']))),
        in_type = amp_specs['in_type'],
        load_type = amp_specs['load_type'],
        tail_type = amp_specs['tail_type'],
        voutcm_ideal = amp_specs['voutcm'] * 1000 * m,
        v_load = v_load * 1000 * m,
        v_pcasc = v_pcasc * 1000 * m,
        v_ncasc = v_ncasc * 1000 * m,
        v_tail = v_tail * 1000 * m,
    )
    

    params = TbParams(pvt=Pvt(), vc=amp_specs['vincm'] * 1000 * m, vd=1 * m, dut=ampParams, cl = amp_specs['cload'] * 1000 * m)

    # Create our simulation input
    @hs.sim
    class TelescopicAmplifierTranSim:
        tb = AmplifierTbTran(params)
        tr = hs.Tran(tstop=300 * n, tstep=1*n)

    # Add the PDK dependencies
    TelescopicAmplifierTranSim.lib(f"{CONDA_PREFIX}/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice", 'tt')
    TelescopicAmplifierTranSim.literal(".save all")
    results = TelescopicAmplifierTranSim.run(sim_options)
    tran_results = results.an[0].data
    t = tran_results['time']
    v_out_diff = tran_results['v(xtop.out_p)'] - tran_results['v(xtop.out_n)']
    v_out_cm = (tran_results['v(xtop.out_p)'] + tran_results['v(xtop.out_n)'])/2
    v_in_diff = tran_results['v(xtop.inp_p)'] - tran_results['v(xtop.inp_n)']
    v_in_cm = (tran_results['v(xtop.inp_p)'] + tran_results['v(xtop.inp_n)'])/2
    
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_in_diff)
    ax[1].plot(t, v_out_diff)
    
    plt.show()
    
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(t, v_in_cm)
    ax[1].plot(t, v_out_cm)
    
    # Create our simulation input
    @hs.sim
    class TelescopicAmplifierAcSim:
        tb = AmplifierTbAc(params)
        myac = hs.Ac(sweep=LogSweep(1e1, 1e11, 10))
    
    # Add the PDK dependencies
    TelescopicAmplifierAcSim.lib(f"{CONDA_PREFIX}/share/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice", 'tt')
    TelescopicAmplifierAcSim.literal(".save all")
    results = TelescopicAmplifierAcSim.run(sim_options)
    ac_results = results.an[0].data
    v_out_diff_ac = ac_results['v(xtop.out_p)'] - ac_results['v(xtop.out_n)']
    f = np.logspace(start=1, stop=11, num=101, endpoint=True)
    f_ax = np.logspace(start=1, stop=11, num=11, endpoint=True)
    f3db_idx = np.squeeze(np.where(abs(v_out_diff_ac) < np.max(abs(v_out_diff_ac))/np.sqrt(2)))[0]
    fgbw_idx = np.squeeze(np.where(abs(v_out_diff_ac) < 1))[0]
    f3db = f[f3db_idx]
    fgbw = f[fgbw_idx]
    Avdc = np.max(abs(v_out_diff_ac))
    v_out_diff_ac_dB = 20 * np.log10(abs(v_out_diff_ac))
    v_out_diff_ac_phase = (np.angle(v_out_diff_ac) % (2*np.pi) - 2*np.pi) * 180 / np.pi
    PM_ac = 180 + v_out_diff_ac_phase[fgbw_idx]
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].semilogx(f, v_out_diff_ac_dB)
    ax[0].grid(color='black', linestyle='--', linewidth=0.5)
    ax[0].grid(visible=True, which="both")
    ax[0].set_xticks(f_ax)
    ax[1].semilogx(f, v_out_diff_ac_phase)
    ax[1].grid(color='black', linestyle='--', linewidth=0.5)
    ax[1].grid(visible=True, which="both")
    ax[1].set_xticks(f_ax)
    plt.show()
    print("Av (AC Sim) = %f" % ((Avdc)))
    print("BW (AC Sim) = %f MHz" % (round(f3db/1e6,2)))
    print("GBW (AC Sim) = %f MHz" % (round(fgbw/1e6,2)))
    print("PM (AC Sim) = %f degrees" % PM_ac)
    
    ampResults = dict(
        Av = Avdc,
        bw = f3db,
        gbw = fgbw,
        pm = PM_ac,
        )
    
    return ampParams, ampResults
   
def generate_telescopic_amp():
    two_stage_amp_specs = dict(
        stage1_in_type='nch_lvt',
        stage1_tail_type='nch_lvt',
        stage1_load_type='pch',
        stage1_vstar_in = 200e-3,
        stage1_vincm=1.1,
        stage1_voutcm=1,
        stage1_vds_tail_min=0.25,
        stage1_gain_min=90,
        stage1_input_stage_gain_min=200,
        stage1_bw_min=60e6,
        
        stage2_in_type='nch_lvt',
        stage2_load_type='pch',
        stage2_voutcm=1.1,
        
        bw_min = 10e6,
        gain_min = 1000,
        vnoise_input_referred=10e-9,
        
        cload=1000e-15,
        rload=100e3,
        vdd=1.8,
        )
    
    stage1_telescopic_amp_specs = dict(
        in_type=two_stage_amp_specs['stage1_in_type'],
        tail_type=two_stage_amp_specs['stage1_tail_type'],
        load_type=two_stage_amp_specs['stage1_load_type'],
        cload=100e-15,
        vdd=two_stage_amp_specs['vdd'],
        vstar_in = two_stage_amp_specs['stage1_vstar_in'],
        vincm=two_stage_amp_specs['stage1_vincm'],
        voutcm=two_stage_amp_specs['stage1_voutcm'],
        vds_tail_min=two_stage_amp_specs['stage1_vds_tail_min'],
        gain_min=two_stage_amp_specs['stage1_gain_min'],
        input_stage_gain_min=two_stage_amp_specs['stage1_input_stage_gain_min'],
        selfbw_min=two_stage_amp_specs['stage1_bw_min'],
        bw_min=two_stage_amp_specs['stage1_bw_min'],
        vnoise_input_referred=two_stage_amp_specs['vnoise_input_referred'] * (two_stage_amp_specs['gain_min']/(two_stage_amp_specs['stage1_input_stage_gain_min'] + two_stage_amp_specs['gain_min'])),
        )
    
    telescopic_amp_params, telescopic_amp_results = generate_stage1_telescopic_amplifier(stage1_telescopic_amp_specs)
    
if __name__ == '__main__':
    generate_telescopic_amp()
    breakpoint()



























