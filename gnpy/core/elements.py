#!/usr/bin/env python3
from gnpy.core.node import Node
from gnpy.core.units import UNITS
import numpy as np
from scipy.constants import c


# network elements

class Transceiver(Node):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, spectral_info):
        return spectral_info

class Fiber(Node):
    def __init__(self, config):
        super().__init__(config)
        metadata = self.config.metadata
        self.length = metadata.length * UNITS[metadata.units]

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid}, length={self.length})'

    def effective_length(self, loss_coef):
        alpha_dict = self.dbkm_2_lin(loss_coef)
        alpha = alpha_dict['alpha_acoef']
        leff = 1 - np.exp(-2 * alpha * self.span_length)
        return leff

    def asymptotic_length(self, loss_coef):
        alpha_dict = self.dbkm_2_lin(loss_coef)
        alpha = alpha_dict['alpha_acoef']
        aleff = 1/(2 * alpha)
        return aleff

    def dbkm_2_lin(self, loss_coef):
        """ calculates the linear loss coefficient
        """
        alpha_pcoef = loss_coef
        alpha_acoef = alpha_pcoef/(2*4.3429448190325184)
        s = 'alpha_pcoef is linear loss coefficient in [dB/km^-1] units'
        s = ''.join([s, "alpha_acoef is linear loss field amplitude \
                     coefficient in [km^-1] units"])
        d = {'alpha_pcoef': alpha_pcoef, 'alpha_acoef': alpha_acoef,
             'description:': s}
        return d

    def beta2(self, dispersion, ref_wavelength=None):
        """ Returns beta2 from dispersion parameter.  Dispersion is entered in
        ps/nm/km.  Disperion can be a numpy array or a single value.  If a
        value ref_wavelength is not entered 1550e-9m will be assumed.
        ref_wavelength can be a numpy array.
        """
        if ref_wavelength is None:
            ref_wavelength = 1550e-9
        wl = ref_wavelength
        D = np.abs(dispersion)
        b2 = (10**21) * (wl**2) * D / (2 * np.pi * c)
#       10^21 scales to ps^2/km
        return b2

    # convenience access
    loss = property(lambda self: self.loss_.value)
    length = property(lambda self: self.length_.value)
    loc  = property(lambda self: self.location)
    lat  = property(lambda self: self.location.latitude)
    long = property(lambda self: self.location.longitude)

    def propagate(self, *carriers):
        for carrier in carriers:
            power = carrier.power
            power = power._replace(signal=0.5 * power.signal * .5,
                                   nonlinear_interference=2 * power.nli,
                                   amplified_spontaneous_emission=2 * power.ase)
            yield carrier._replace(power=power)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        return spectral_info.update(carriers=carriers)
