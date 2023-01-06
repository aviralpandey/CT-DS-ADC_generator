import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt


class MosDB:

    _vars = ["gm", "ids", "gds", "cgg", "cdd"]

    def __init__(self):
        self._mos_list = []
        self._mos_to_lch = {}
        self._map = {}

    def build(self, filename):
        """Given the filename, build the MOS Database"""
        data = np.load(filename, allow_pickle=True)[()]
        self._mos_list.extend(data["mos_list"])
        mos = data["mos_list"][0]
        lch_list = data["lch_list"]
        self._mos_to_lch[mos] = lch_list
        for lch_idx, lch in enumerate(lch_list):
            # Regardless of how the MOS was characterized
            # We make all of VGS/VDS/VBS positive
            vgs_raw = np.abs(np.array(data["vgs_list"]))
            vds_raw = np.abs(np.array(data["vds_list"]))
            vbs_raw = np.abs(np.array(data["vbs_list"]))
            for var in self._vars:
                var_raw = data[var][0, lch_idx, :, :, :]
                self._map[(mos, lch, var)] = interp.RegularGridInterpolator(
                    (vbs_raw, vgs_raw, vds_raw), var_raw
                )
            self._map[(mos, lch, "vgs_list")] = vgs_raw
            self._map[(mos, lch, "vds_list")] = vds_raw
            self._map[(mos, lch, "vbs_list")] = vbs_raw

    def query_db(self, varname, mos_type, lch):
        """Query the database for the given variable name"""
        return self._map[(mos_type, lch, varname)]

    def query_db_for_vgs(
        self, mos_type, lch, vbs, vds, vstar=200e-3, ids=0, mode="vstar", scale=1
    ):
        vgs_vals = self._map[(mos_type, lch, "vgs_list")]
        vgs_min = min(vgs_vals)
        vgs_max = max(vgs_vals)
        gm_func = self._map[(mos_type, lch, "gm")]
        id_func = self._map[(mos_type, lch, "ids")]
        if mode == "vstar":
            # Only search where the VGS is greater than the VGS
            # at which vstar min was achieved
            vstar_min_vgs_idx = np.argmin(
                2 * id_func((vbs, vgs_vals, vds)) / gm_func((vbs, vgs_vals, vds))
            )
            vgs_min = vgs_vals[vstar_min_vgs_idx]

            def vstar_func(vgs):
                return vstar - 2 * id_func((vbs, vgs, vds)) / gm_func((vbs, vgs, vds))

            return opt.brentq(vstar_func, vgs_min, vgs_max)

        elif mode == "ids":

            def ids_func(vgs):
                return ids - scale * id_func((vbs, vgs, vds))

            return opt.brentq(ids_func, vgs_min, vgs_max)


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
        vbs_raw = -np.array(database["vbs_list"])
    else:
        vgs_raw = -np.array(database["vgs_list"])
        vds_raw = -np.array(database["vds_list"])
        vbs_raw = np.array(database["vbs_list"])

    interp = scipy.interpolate.RegularGridInterpolator(
        (vbs_raw, vgs_raw, vds_raw), var_raw
    )
    return interp([-vbs, vgs, vds]).item()


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
        vbs_raw = -np.array(database["vbs_list"])
    else:
        vgs_raw = -np.array(database["vgs_list"])
        vds_raw = -np.array(database["vds_list"])
        vbs_raw = np.array(database["vbs_list"])

    ids_raw = database["ids"][mos_index, lch_index, :, :, :] * scale
    gm_raw = database["gm"][mos_index, lch_index, :, :, :] * scale

    if mode == "vstar":
        target_gm_id = 2 / vstar
        gm_id_raw = gm_raw / ids_raw
        interp = scipy.interpolate.RegularGridInterpolator(
            (vbs_raw, vgs_raw, vds_raw), gm_id_raw
        )

        resample_vector_vbs = [-vbs] * (np.shape(vgs_raw)[0])
        resample_vector_vds = [vds] * (np.shape(vgs_raw)[0])
        resample_points = np.transpose(
            np.vstack((resample_vector_vbs, vgs_raw, resample_vector_vds))
        )
        gm_id_raw_1d = interp(resample_points)

        interp_vgs = scipy.interpolate.interp1d(gm_id_raw_1d, vgs_raw)

        return interp_vgs(target_gm_id).item()

    elif mode == "ids":
        interp = scipy.interpolate.RegularGridInterpolator(
            (vbs_raw, vgs_raw, vds_raw), ids_raw
        )

        resample_vector_vbs = [-vbs] * (np.shape(vgs_raw)[0])
        resample_vector_vds = [vds] * (np.shape(vgs_raw)[0])
        resample_points = np.transpose(
            np.vstack((resample_vector_vbs, vgs_raw, resample_vector_vds))
        )
        ids_raw_1d = interp(resample_points)

        interp_vgs = scipy.interpolate.interp1d(ids_raw_1d, vgs_raw)

        return interp_vgs(ids).item()


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


nch_db_filename = "database_nch.npy"
nch_lvt_db_filename = "database_nch_lvt.npy"
pch_db_filename = "database_pch.npy"
pch_lvt_db_filename = "database_pch_lvt.npy"
