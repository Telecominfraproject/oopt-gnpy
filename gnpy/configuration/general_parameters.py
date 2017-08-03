# -*- coding: utf-8 -*
"""general_parameters.py contains the general configuration settings

    The sectings are subdivided in two dictionaries:
        sys_param: a dictionary containing the general system parameters:
            f0: the starting frequency of the laser grid used to describe the WDM system [THz]
            ns: the number of 6.25 GHz slots in the grid

        control_param:
            save_each_comp: a boolean flag. If true, it saves in output folder one spectrum file at the output of each
                            component, otherwise it saves just the last spectrum
            is_linear: a bool flag. If true, is doesn't compute NLI, if false, OLE will consider NLI
            is_analytic: a boolean flag. If true, the NLI is computed through the analytic formula, otherwise it uses
                the double integral. Warning: the double integral is very slow.
            points_not_interp: if the double integral is used, it indicates how much points are calculated, others will
                be interpolated
            kind_interp: a string indicating the interpolation method for the double integral
            th_fwm: the threshold of the four wave mixing efficiency for the double integral
            n_points: number of points in which the double integral is computed in the high FWM efficiency region
            n_points_min: number of points in which the double integral is computed in the low FWM efficiency region
            n_cores: number of cores for parallel computation [not yet implemented]
"""
# System parameters
sys_param = {
    'f0': 190.603,
    'ns': 800
}

# control parameters
control_param = {
    'save_each_comp': True,
    'is_linear': False,
    'is_analytic': True,
    'points_not_interp': 2,
    'kind_interp': 'linear',
    'th_fwm': 50,
    'n_points': 500,
    'n_points_min': 4,
    'n_cores': 1
}
