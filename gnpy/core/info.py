#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.info
==============

This module contains classes for modelling :class:`SpectralInformation`.
'''


from collections import namedtuple
from typing import List
import gnpy.core.exceptions as exceptions
from gnpy.core.utils import automatic_nch, lin2db, pairwise


class Power(namedtuple('Power', 'signal nli ase')):
    """carriers power in W"""


class Channel(namedtuple('Channel', 'channel_number frequency baud_rate roll_off power chromatic_dispersion pmd')):
    """ Class containing the parameters of a WDM signal.

        :param channel_number: channel number in the WDM grid
        :param frequency: central frequency of the signal (Hz)
        :param baud_rate: the symbol rate of the signal (Baud)
        :param roll_off: the roll off of the signal. It is a pure number between 0 and 1
        :param power (gnpy.core.info.Power): power of signal, ASE noise and NLI (W)
        :param chromatic_dispersion: chromatic dispersion (s/m)
        :param pmd: polarization mode dispersion (s)
    """


class Pref(namedtuple('Pref', 'p_span0, p_spani, neq_ch ')):
    """noiseless reference power in dBm:
    p_span0: inital target carrier power
    p_spani: carrier power after element i
    neq_ch: equivalent channel count in dB"""


class SpectralInformation(namedtuple('SpectralInformation', 'pref carriers')):
    '''
    >>> pref = Pref(0, 0, lin2db(10))
    >>> c_20 = Channel(0, 192e12, 33e9, 0.15, Power(.001, 0, 0), 0, 0)
    >>> c_20_5 = Channel(1, 192.05e12, 33e9, 0.15, Power(.001, 0, 0), 0, 0)
    >>> c_21 = Channel(2, 192.1e12, 33e9, 0.15, Power(.001, 0, 0), 0, 0)
    >>> c_21_100 = Channel(2, 192.1e12, 100e9, 0.15, Power(.001, 0, 0), 0, 0)
    >>> _ = SpectralInformation(pref, [c_20])
    >>> _ = SpectralInformation(pref, [c_20, c_20_5, c_21])
    >>> _ = SpectralInformation(pref, [c_21, c_20])
    Traceback (most recent call last):
    ...
    gnpy.core.exceptions.SpectrumError: Channels in SpectralInformation are not sorted: 192.1 THz >= 192.0 THz
    >>> _ = SpectralInformation(pref, [c_20, c_20_5, c_21_100])
    Traceback (most recent call last):
    ...
    gnpy.core.exceptions.SpectrumError: Channel overlap between 192.05 (+0.0165) THz and 192.1 (-0.05) THz
    '''

    def __new__(cls, pref: Pref, carriers: List[Channel]):
        res = super().__new__(cls, pref, carriers)
        for (a, b) in pairwise(carriers):
            if a.frequency >= b.frequency:
                raise exceptions.SpectrumError('Channels in SpectralInformation are not sorted: '
                                               f'{a.frequency / 1e12} THz >= {b.frequency / 1e12} THz')
            # assume that the baud_rate is also the effective channel width
            if a.frequency + a.baud_rate / 2 > b.frequency - b.baud_rate / 2:
                raise exceptions.SpectrumError('Channel overlap between '
                                               f'{a.frequency / 1e12} (+{a.baud_rate / 2e12}) THz and '
                                               f'{b.frequency / 1e12} (-{b.baud_rate / 2e12}) THz')
        return res


def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, power, spacing):
    # pref in dB : convert power lin into power in dB
    pref = lin2db(power * 1e3)
    nb_channel = automatic_nch(f_min, f_max, spacing)
    si = SpectralInformation(
        pref=Pref(pref, pref, lin2db(nb_channel)),
        carriers=[
            Channel(f, (f_min + spacing * f),
                    baud_rate, roll_off, Power(power, 0, 0), 0, 0) for f in range(1, nb_channel + 1)
        ]
    )
    return si
