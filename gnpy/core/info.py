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

class Power(namedtuple('Power', 'signal nli ase')):
    """carriers power in W"""


class Channel(namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')):
    pass


class Pref(namedtuple('Pref', 'p_span0, p_spani, neq_ch ')):
    """noiseless reference power in dBm: 
    p_span0: inital target carrier power
    p_spani: carrier power after element i
    neq_ch: equivalent channel count in dB"""


class SpectralInformation(namedtuple('SpectralInformation', 'pref carriers')):

    def __new__(cls, pref, carriers):
        return super().__new__(cls, pref, carriers)


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

    si = si._replace(carriers=tuple(Channel(f+1, 191.3+spacing*(f+1), 32e9, 0.15, Power(1e-3, f, 1)) for f in range(96)))

    print(f'si = {si}')
    print(f'si = {si.carriers[0].power.nli}')
    print(f'si = {si.carriers[20].power.nli}')
    si2 = si._replace(carriers=tuple(c._replace(power = c.power._replace(nli = c.power.nli * 1e5))
                              for c in si.carriers))
    print(f'si2 = {si2}')
