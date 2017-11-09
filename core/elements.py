#!/usr/bin/env python
from collections import namedtuple

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

# network elements

class Transceiver(namedtuple('Transceiver', 'uid location')):
    def __new__(cls, uid, location):
        location = Location(**location)
        return super().__new__(cls, uid, location)

    def __call__(self, *spectral_infos):
        for si in spectral_infos:
            yield si

    # convenience access
    loc  = property(lambda self: self.location)
    lat  = property(lambda self: self.location.coords.latitude)
    long = property(lambda self: self.location.coords.longitude)

class Fiber(namedtuple('Fiber', 'uid length_ location')):
    def __new__(cls, uid, length, units, location):
        length = Length(length, units)
        location = Coords(**location)
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

    def __call__(self, *spectral_infos):
        for si in spectral_infos:
            yield si._replace(carriers=tuple(self.propagate(*si.carriers)))

    # convenience access
    length = property(lambda self: self.length_.value)
    loc  = property(lambda self: self.location)
    lat  = property(lambda self: self.location.latitude)
    long = property(lambda self: self.location.longitude)
