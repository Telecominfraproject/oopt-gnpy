#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.info
==============

This module contains classes for modelling :class:`SpectralInformation`.
'''


from collections import namedtuple
from numpy import argsort, squeeze, min, mean, append
from gnpy.core.utils import automatic_nch, lin2db, dimension_reshape

from numpy import array


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


class SpectralInformation(object):
    def __init__(self, pref, frequency, grid, signal, nli, ase, baud_rate,
                 roll_off, chromatic_dispersion, pmd):
        self._pref = pref
        indices = argsort(frequency)
        self._frequency = frequency[indices]
        self._number_of_channels = len(self._frequency)
        self._grid = grid[indices]
        self._signal = signal[indices]
        self._nli = nli[indices]
        self._ase = ase[indices]
        self._baud_rate = baud_rate[indices]
        self._roll_off = roll_off[indices]
        self._chromatic_dispersion = chromatic_dispersion[indices]
        self._pmd = pmd[indices]

    @property
    def pref(self):
        return self._pref

    @pref.setter
    def pref(self, pref):
        self._pref = pref

    @property
    def frequency(self):
        return self._frequency

    @property
    def grid(self):
        return self._grid

    @property
    def baud_rate(self):
        return self._baud_rate

    @property
    def number_of_channels(self):
        return self._number_of_channels

    @property
    def powers(self):
        powers = zip(self.signal, self.nli, self.ase)
        return [Power(*p) for p in powers]

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        self._signal = signal

    @property
    def nli(self):
        return self._nli

    @nli.setter
    def nli(self, nli):
        self._nli = nli

    @property
    def ase(self):
        return self._ase

    @ase.setter
    def ase(self, ase):
        self._ase = ase

    @property
    def roll_off(self):
        return self._roll_off

    @property
    def chromatic_dispersion(self):
        return self._chromatic_dispersion

    @chromatic_dispersion.setter
    def chromatic_dispersion(self, chromatic_dispersion):
        self._chromatic_dispersion = chromatic_dispersion

    @property
    def pmd(self):
        return self._pmd

    @pmd.setter
    def pmd(self, pmd):
        self._pmd = pmd

    @property
    def carriers(self):
        channel_numbers = [n for n in range(1, self.number_of_channels + 1)]
        entries = zip(channel_numbers, self.frequency, self.baud_rate,
                      self.roll_off, self.powers, self.chromatic_dispersion, self.pmd)
        return [Channel(*entry) for entry in entries]

    def __add__(self, si):
        frequency = append(self.frequency, si.frequency)
        grid = append(self.grid, si.grid)
        signal = append(self.signal, si.signal)
        nli = append(self.nli, si.nli)
        ase = append(self.ase, si.ase)
        baud_rate = append(self.baud_rate, si.baud_rate)
        roll_off = append(self.roll_off, si.roll_off)
        chromatic_dispersion = append(self.chromatic_dispersion, si.chromatic_dispersion)
        pmd = append(self.pmd, si.pmd)
        number_of_channels = self.number_of_channels + si.number_of_channels
        pref = lin2db(mean(signal) * 1e3)
        pref = Pref(pref, pref, lin2db(number_of_channels))
        si = SpectralInformation(pref=pref, frequency=frequency, grid=grid,
                                 signal=signal, nli=nli, ase=ase,
                                 baud_rate=baud_rate, roll_off=roll_off,
                                 chromatic_dispersion=chromatic_dispersion, pmd=pmd)
        return si

    def _replace(self, carriers, pref):
        self.chromatic_dispersion = array([c.chromatic_dispersion for c in carriers])
        self.pmd = array([c.pmd for c in carriers])
        self.signal = array([c.power.signal for c in carriers])
        self.nli = array([c.power.nli for c in carriers])
        self.ase = array([c.power.ase for c in carriers])
        self.pref = pref
        return self



def create_arbitrary_spectral_information(frequency, grid=None, signal=None, baud_rate=None,
                                          roll_off=None, chromatic_dispersion=None, pmd=None):
    """ Creates an arbitrary spectral information """
    frequency = array(frequency)
    order = argsort(frequency)
    frequency = squeeze(frequency[order])
    number_of_channels = len(frequency)
    grid = dimension_reshape(grid, number_of_channels, min((frequency[1:] - frequency[:-1])), order)
    signal = dimension_reshape(signal, number_of_channels, 0, order)
    nli = dimension_reshape(0, number_of_channels)
    ase = dimension_reshape(0, number_of_channels)
    baud_rate = dimension_reshape(baud_rate, number_of_channels, grid, order)
    roll_off = dimension_reshape(roll_off, number_of_channels, 0, order)
    chromatic_dispersion = dimension_reshape(chromatic_dispersion, number_of_channels, 0, order)
    pmd = dimension_reshape(pmd, number_of_channels, 0, order)
    pref = lin2db(mean(signal) * 1e3)
    pref = Pref(pref, pref, lin2db(number_of_channels))
    si = SpectralInformation(pref=pref, frequency=frequency, grid=grid,
                             signal=signal, nli=nli, ase=ase,
                             baud_rate=baud_rate, roll_off=roll_off,
                             chromatic_dispersion=chromatic_dispersion, pmd=pmd)
    return si


def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, power, spacing):
    """ Creates a fixed grid spectral information with flat power """
    nb_channel = automatic_nch(f_min, f_max, spacing)
    frequency = [(f_min + spacing * i) for i in range(1, nb_channel + 1)]
    si = create_arbitrary_spectral_information(
        frequency, grid=spacing, signal=power, baud_rate=baud_rate, roll_off=roll_off
    )
    return si
