#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.info
==============

This module contains classes for modelling :class:`SpectralInformation`.
'''


from collections import namedtuple
from numpy import array
from gnpy.core.utils import lin2db, db2lin
from json import loads
from gnpy.core.utils import load_json
from gnpy.core.equipment import automatic_nch, automatic_spacing

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
    """carriers power in W"""
    _ABBREVS = {'nli': 'nonlinear_interference',
                'ase': 'amplified_spontaneous_emission',}


class Channel(namedtuple('Channel', 'channel_number frequency baud_rate roll_off power'), ConvenienceAccess):

    _ABBREVS = {'channel':  'channel_number',
                'num_chan': 'channel_number',
                'ffs':      'frequency',
                'freq':     'frequency',}

class Pref(namedtuple('Pref', 'p_span0, p_spani, neq_ch '), ConvenienceAccess):
    """noiseless reference power in dBm: 
    p0: inital target carrier power
    pi: carrier power after element i
    neqch: equivalent channel count in dB"""

    _ABBREVS = {'p0' :  'p_span0',
                'pi' :  'p_spani'}

class SpectralInformation(namedtuple('SpectralInformation', 'pref carriers'), ConvenienceAccess):

    def __new__(cls, pref, carriers):
        return super().__new__(cls, pref, carriers)

def merge_input_spectral_information(*si):
    """mix channel combs of different baud rates and power"""
    #TODO
    pass

def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, power, spacing):
    # pref in dB : convert power lin into power in dB
    pref = lin2db(power * 1e3)
    nb_channel = automatic_nch(f_min, f_max, spacing)
    si = SpectralInformation(
        pref=Pref(pref, pref, lin2db(nb_channel)),
        carriers=[
            Channel(f, (f_min+spacing*f),
            baud_rate, roll_off, Power(power, 0, 0)) for f in range(1,nb_channel+1)
            ])
    return si

if __name__ == '__main__':
    pref = lin2db(power * 1e3)
    si = SpectralInformation(
        Pref(pref, pref),
        Channel(1, 193.95e12, 32e9, 0.15,  # 193.95 THz, 32 Gbaud
            Power(1e-3, 1e-6, 1e-6)),             # 1 mW, 1uW, 1uW
        Channel(1, 195.95e12, 32e9, 0.15,  # 195.95 THz, 32 Gbaud
            Power(1.2e-3, 1e-6, 1e-6)),           # 1.2 mW, 1uW, 1uW
    )

    si = SpectralInformation()
    spacing = 0.05 # THz

    si = si.update(carriers=tuple(Channel(f+1, 191.3+spacing*(f+1), 32e9, 0.15, Power(1e-3, f, 1)) for f in range(96)))

    print(f'si = {si}')
    print(f'si = {si.carriers[0].power.nli}')
    print(f'si = {si.carriers[20].power.nli}')
    si2 = si.update(carriers=tuple(c.update(power = c.power.update(nli = c.power.nli * 1e5))
                              for c in si.carriers))
    print(f'si2 = {si2}')
