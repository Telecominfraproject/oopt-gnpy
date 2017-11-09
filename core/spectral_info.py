#!/usr/bin/env python3

from collections import namedtuple


si_tuple = namedtuple('SpectralInfo',
                      'center_freq power osnr')


class SpectralInfo(si_tuple):

    def __init__(self, center_freq, power, osnr):
        self._center_freq = center_freq
        self._power = power
        self._osnr = osnr
