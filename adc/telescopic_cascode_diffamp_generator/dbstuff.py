import numpy as np
import scipy.interpolate


# Query the given database for the given variable name, for the given lch, vbs, vgs, vds
def query_db(database, varname, mos_type, lch, vbs, vgs, vds):

    mos_list = database["mos_list"]
    mos_index = mos_list.index(mos_type)
    lch_list = database["lch_list"]
    lch_index = lch_list.index(lch)
    var_raw = database[varname][mos_index, lch_index, :, :, :]
    if mos_type == "nch" or mos_type == "nch_lvt":
        vgs_raw = np.array(database["vgs_list"])
        vds_raw = np.array(database["vds_list"])
        vbs_raw = np.array(database["vbs_list"])
    else:
        vgs_raw = -np.array(database["vgs_list"])
        vds_raw = -np.array(database["vds_list"])
        vbs_raw = -np.array(database["vbs_list"])

    interp = scipy.interpolate.RegularGridInterpolator(
        (vbs_raw, vgs_raw, vds_raw), var_raw
    )
    return interp([vbs, vgs, vds]).item()


# Specialized query function to find the vgs of a transistor if all other OP conditions are known.
# It has two operation modes:
# 'vstar' mode (where vstar = 2 * ID / gm): Find the vgs of the device if its lch, vbs, vds and gm/ID are known
# 'ids' mode: Find the vgs of the device if its lch, vbs, vds and ids are known.
def query_db_for_vgs(
    database, mos_type, lch, vbs, vds, vstar=200e-3, ids=0, mode="vstar", scale=1
):
    mos_list = database["mos_list"]
    mos_index = mos_list.index(mos_type)
    lch_list = database["lch_list"]
    lch_index = lch_list.index(lch)
    if mos_type == "nch" or mos_type == "nch_lvt":
        vgs_raw = np.array(database["vgs_list"])
        vds_raw = np.array(database["vds_list"])
        vbs_raw = np.array(database["vbs_list"])
    else:
        vgs_raw = -np.array(database["vgs_list"])
        vds_raw = -np.array(database["vds_list"])
        vbs_raw = -np.array(database["vbs_list"])

    ids_raw = database["ids"][mos_index, lch_index, :, :, :] * scale
    gm_raw = database["gm"][mos_index, lch_index, :, :, :] * scale

    if mode == "vstar":
        target_gm_id = 2 / vstar
        gm_id_raw = gm_raw / ids_raw
        interp = scipy.interpolate.RegularGridInterpolator(
            (vbs_raw, vgs_raw, vds_raw), gm_id_raw
        )

        resample_vector_vbs = [vbs] * (np.shape(vgs_raw)[0])
        resample_vector_vds = [vds] * (np.shape(vgs_raw)[0])
        resample_points = np.transpose(
            np.vstack((resample_vector_vbs, vgs_raw, resample_vector_vds))
        )
        gm_id_raw_1d = interp(resample_points)

        interp_vgs = scipy.interpolate.interp1d(gm_id_raw_1d, vgs_raw)

        try:
            return interp_vgs(target_gm_id).item()
        except:
            raise ValueError

    elif mode == "ids":
        interp = scipy.interpolate.RegularGridInterpolator(
            (vbs_raw, vgs_raw, vds_raw), ids_raw
        )

        resample_vector_vbs = [vbs] * (np.shape(vgs_raw)[0])
        resample_vector_vds = [vds] * (np.shape(vgs_raw)[0])
        resample_points = np.transpose(
            np.vstack((resample_vector_vbs, vgs_raw, resample_vector_vds))
        )
        ids_raw_1d = interp(resample_points)

        interp_vgs = scipy.interpolate.interp1d(ids_raw_1d, vgs_raw)

        try:
            return interp_vgs(ids).item()
        except:
            raise ValueError


# Query function that returns the function for the given variable instead of a single datapoint.
# Useful for optimization kinda stuff where you iterate over functions.
def query_db_for_function(database, varname, mos_type, lch):

    mos_list = database["mos_list"]
    mos_index = mos_list.index(mos_type)
    lch_list = database["lch_list"]
    lch_index = lch_list.index(lch)

    var_raw = database[varname][mos_index, lch_index, :, :, :]
    if mos_type == "nch" or mos_type == "nch_lvt":
        vgs_raw = np.array(database["vgs_list"])
        vds_raw = np.array(database["vds_list"])
        vbs_raw = np.array(database["vbs_list"])
    else:
        vgs_raw = -np.array(database["vgs_list"])
        vds_raw = -np.array(database["vds_list"])
        vbs_raw = -np.array(database["vbs_list"])

    interp = scipy.interpolate.RegularGridInterpolator(
        (vbs_raw, vgs_raw, vds_raw), var_raw
    )
    return interp
