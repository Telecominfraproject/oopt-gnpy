#!/usr/bin/env python3

from collections import namedtuple
import numpy as np
from utilities import c

# helpers

class Coords(namedtuple('Coords', 'latitude longitude')):
    lat  = property(lambda self: self.latitude)
    long = property(lambda self: self.longitude)

class Location(namedtuple('Location', 'city region coords')):
    def __new__(cls, city, region, **kwargs):
        return super().__new__(cls, city, region, Coords(**kwargs))

class Length(namedtuple('Length', 'quantity units')):
    UNITS = {'m': 1, 'km': 1e3}

    @property
    def value(self):
        return self.quantity * self.UNITS[self.units]
    val = value

class Loss(namedtuple('Loss', 'quantity units')):
    UNITS = {'1/m': 0.0002, '1/km': 0.2, 'dB/m': 0.0002, 'dB/km': 0.2}

    @property
    def value(self):
        return self.quantity * self.UNITS[self.units]
    val = value


# network elements

class Transceiver(namedtuple('Transceiver', 'uid location')):
    def __new__(cls, uid, location):
        location = Location(**location)
        return super().__new__(cls, uid, location)

    def __call__(self, spectral_info):
        return spectral_info

    # convenience access
    loc  = property(lambda self: self.location)
    lat  = property(lambda self: self.location.coords.latitude)
    long = property(lambda self: self.location.coords.longitude)

class Fiber(namedtuple('Fiber', 'uid length_ location')):
    def __new__(cls, uid, length, units, location):
        length = Length(length, units)
        location = Coords(**location)
        loss = Loss(loss, units)
        return super().__new__(cls, uid, length, location)

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid}, length={self.length})'

    def propagate(self, *carriers):
        for carrier in carriers:
            power = carrier.power
            power = power._replace(signal = power.signal * .5,
                                   nonlinear_interference = power.nli * 2,
                                   amplified_spontaneous_emission = power.ase * 2)
            yield carrier._replace(power=power)

    def __call__(self, spectral_info):
        return spectral_info.update(carriers=tuple(self.propagate(*spectral_info.carriers)))
    
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
        b2 = (10**21) * (wl**2) * D / (2 * np.pi * c())
#       10^21 scales to ps^2/km
        return b2

    # convenience access
    loss = property(lambda self: self.loss_.value)
    length = property(lambda self: self.length_.value)
    loc  = property(lambda self: self.location)
    lat  = property(lambda self: self.location.latitude)
    long = property(lambda self: self.location.longitude)
