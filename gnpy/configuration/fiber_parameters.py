# coding=utf-8
""" fiber_parameters.py describes the fiber parameters.
    fibers is a dictionary containing a dictionary for each kind of fiber
    each dictionary has to report:
        reference_frequency: the frequency at which the parameters are evaluated [THz]
        alpha: the attenuation coefficient [dB/km]
        alpha_1st: the first derivative of alpha indicating the alpha slope [dB/km/THz]
                if you assume a flat attenuation with respect to the frequency you put it as zero
        beta_2: the dispersion coefficient [ps^2/km]
        n_2: second-order nonlinear refractive index [m^2/W]
            a typical value is 2.5E-20 m^2/W
        a_eff: the effective area of the fiber [um^2]
"""

fibers = {
    'SMF': {
        'reference_frequency': 193.5,
        'alpha': 0.21,
        'alpha_1st': 0,
        'beta_2': 21.27,
        'n_2': 2.5E-20,
        'a_eff': 80,
    },
    'NZDF': {
        'reference_frequency': 193.5,
        'alpha': 0.22,
        'alpha_1st': 0,
        'beta_2': 21,
        'n_2': 2.5E-20,
        'a_eff': 70,
    }
}
