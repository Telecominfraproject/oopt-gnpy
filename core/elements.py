#!/usr/bin/env python
from collections import namedtuple

class Coords(namedtuple('Coords', 'latitude longitude')):
    lat  = property(lambda self: self.latitude)
    long = property(lambda self: self.longitude)

class Location(namedtuple('Location', 'city region coords')):
    def __new__(cls, city, region, **kwargs):
        return super().__new__(cls, city, region, Coords(**kwargs))

class Transceiver(namedtuple('Transceiver', 'uid location')):
    def __new__(cls, uid, location):
        return super().__new__(cls, uid, Location(**location))

    def __call__(self, *spectral_infos):
        return spectral_info.copy()

    # convenience access
    loc  = property(lambda self: self.location)
    lat  = property(lambda self: self.location.coords.latitude)
    long = property(lambda self: self.location.coords.longitude)

class Length(namedtuple('Length', 'quantity units')):
    UNITS = {'m': 1, 'km': 1e3}

    @property
    def value(self):
        return self.quantity * self.UNITS[self.units]
    val = value

class Fiber(namedtuple('Fiber', 'uid length_ location')):
    def __new__(cls, uid, length, units, location):
        return super().__new__(cls, uid, Length(length, units), Coords(**location))

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid}, length={self.length})'

    def __call__(self, *spectral_infos):
        return spectral_info.copy()

    # convenience access
    length = property(lambda self: self.length_.value)
    loc  = property(lambda self: self.location)
    lat  = property(lambda self: self.location.latitude)
    long = property(lambda self: self.location.longitude)
