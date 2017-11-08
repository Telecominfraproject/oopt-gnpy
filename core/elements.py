#!/usr/bin/env python
from collections import namedtuple

class City(namedtuple('City', 'id city region latitude longitude')):
    def __call__(self, spectral_info):
        return spectral_info.copy()

class Fiber:
    UNITS = {'m': 1, 'km': 1e3}
    def __init__(self, id, length, units, latitude, longitude):
        self.id = id
        self._length = length
        self._units = units
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self):
        return f'{type(self).__name__}(id={self.id}, length={self.length})'

    @property
    def length(self):
        return self._length * self.UNITS[self._units]

    def __call__(self, spectral_info):
        return spectral_info.copy()
