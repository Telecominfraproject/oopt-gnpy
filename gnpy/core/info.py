#!/usr/bin/env python3

'''
gnpy.core.info
==============

This module contains classes for modelling SpectralInformation.
'''


from collections import namedtuple


class ConvenienceAccess:

    def __init_subclass__(cls):
        for abbrev, field in getattr(cls, '_ABBREVS', {}).items():
            setattr(cls, abbrev, property(lambda self, f=field: getattr(self, f)))

    def update(self, **kwargs):
        for abbrev, field in getattr(self, '_ABBREVS', {}).items():
            if abbrev in kwargs:
                kwargs[field] = kwargs.pop(abbrev)
        return self._replace(**kwargs)


class Power(namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission'), ConvenienceAccess):

    _ABBREVS = {'nli': 'nonlinear_interference',
                'ase': 'amplified_spontaneous_emission',}


class Channel(namedtuple('Channel', 'channel_number frequency baud_rate roll_off power'), ConvenienceAccess):

    _ABBREVS = {'channel':  'channel_number',
                'num_chan': 'channel_number',
                'ffs':      'frequency',
                'freq':     'frequency',}


class SpectralInformation(namedtuple('SpectralInformation', 'carriers'), ConvenienceAccess):

    def __new__(cls, *carriers):
        return super().__new__(cls, carriers)


if __name__ == '__main__':
    si = SpectralInformation(
        Channel(1, 193.95e12, 32e9, 0.15,  # 193.95 THz, 32 Gbaud
            Power(1e-3, 1e-6, 1e-6)),             # 1 mW, 1uW, 1uW
        Channel(1, 195.95e12, 32e9, 0.15,  # 195.95 THz, 32 Gbaud
            Power(1.2e-3, 1e-6, 1e-6)),           # 1.2 mW, 1uW, 1uW
    )

    si = SpectralInformation()
    spacing = 0.05 #THz

    si = si.update(carriers=tuple(Channel(f+1, 191.3+spacing*(f+1), 32e9, 0.15, Power(1e-3, f, 1)) for f in range(96)))

    print(f'si = {si}')
    print(f'si = {si.carriers[0].power.nli}')
    print(f'si = {si.carriers[20].power.nli}')
    si2 = si.update(carriers=tuple(c.update(power = c.power.update(nli = c.power.nli * 1e5))
                              for c in si.carriers))
    print(f'si2 = {si2}')
