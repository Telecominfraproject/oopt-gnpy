#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.info
==============

This module contains classes for modelling :class:`SpectralInformation`.
"""

from collections import namedtuple
from collections.abc import Sized
from numpy import argsort, mean, array, squeeze, append, ones, ceil, any, zeros, outer

from gnpy.core.utils import automatic_nch, lin2db
from gnpy.core.exceptions import SpectrumError

DEFAULT_SLOT_WIDTH_STEP = 12.5e9  # Hz
"""Channels with unspecified slot width will have their slot width evaluated as the baud rate rounded up to the minimum
multiple of the DEFAULT_SLOT_WIDTH_STEP (the baud rate is extended including the roll off in this evaluation)"""


class Power(namedtuple('Power', 'signal nli ase')):
    """carriers power in W"""


class Channel(namedtuple('Channel', 'channel_number frequency baud_rate slot_width ' +
                         'roll_off power chromatic_dispersion pmd')):
    """ Class containing the parameters of a WDM signal.
        :param channel_number: channel number in the WDM grid
        :param frequency: central frequency of the signal (Hz)
        :param baud_rate: the symbol rate of the signal (Baud)
        :param slot_width: the slot width (Hz)
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
    """ Class containing the parameters of the entire WDM comb."""
    def __init__(self, frequency: array, baud_rate: array, slot_width: array, signal: array, nli: array, ase: array,
                 roll_off: array, chromatic_dispersion: array, pmd: array):
        indices = argsort(frequency)
        self._frequency = frequency[indices]
        self._df = outer(ones(frequency.shape), frequency) - outer(frequency, ones(frequency.shape))
        self._number_of_channels = len(self._frequency)
        self._slot_width = slot_width[indices]
        self._baud_rate = baud_rate[indices]
        if any(self._frequency[:-1] + self._slot_width[:-1] / 2 > self._frequency[1:] - self._slot_width[1:] / 2):
            raise SpectrumError('Spectrum required slot widths larger than the frequency spectral distances.')
        elif any(self._baud_rate > self._slot_width):
            raise SpectrumError('Spectrum baud rate larger than the slot width.')
        self._signal = signal[indices]
        self._nli = nli[indices]
        self._ase = ase[indices]
        self._roll_off = roll_off[indices]
        self._chromatic_dispersion = chromatic_dispersion[indices]
        self._pmd = pmd[indices]
        self._channel_number = [*range(1, self._number_of_channels + 1)]
        pref = lin2db(mean(signal) * 1e3)
        self._pref = Pref(pref, pref, lin2db(self._number_of_channels))

    @property
    def pref(self):
        """Instance of gnpy.info.Pref"""
        return self._pref

    @pref.setter
    def pref(self, pref: Pref):
        self._pref = pref

    @property
    def frequency(self):
        return self._frequency

    @property
    def df(self):
        """Matrix of relative frequency distances between all channels. Positive elements in the upper right side."""
        return self._df

    @property
    def slot_width(self):
        return self._slot_width

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
    def channel_number(self):
        return self._channel_number

    @property
    def carriers(self):
        entries = zip(self.channel_number, self.frequency, self.baud_rate, self.slot_width,
                      self.roll_off, self.powers, self.chromatic_dispersion, self.pmd)
        return [Channel(*entry) for entry in entries]

    def __add__(self, si):
        try:
            return SpectralInformation(frequency=append(self.frequency, si.frequency),
                                       slot_width=append(self.slot_width, si.slot_width),
                                       signal=append(self.signal, si.signal), nli=append(self.nli, si.nli),
                                       ase=append(self.ase, si.ase), baud_rate=append(self.baud_rate, si.baud_rate),
                                       roll_off=append(self.roll_off, si.roll_off),
                                       chromatic_dispersion=append(self.chromatic_dispersion, si.chromatic_dispersion),
                                       pmd=append(self.pmd, si.pmd))
        except SpectrumError:
            raise SpectrumError('Spectra cannot be summed: channels overlapping.')

    def _replace(self, carriers, pref):
        self.chromatic_dispersion = array([c.chromatic_dispersion for c in carriers])
        self.pmd = array([c.pmd for c in carriers])
        self.signal = array([c.power.signal for c in carriers])
        self.nli = array([c.power.nli for c in carriers])
        self.ase = array([c.power.ase for c in carriers])
        self.pref = pref
        return self


def create_arbitrary_spectral_information(frequency, signal, baud_rate, slot_width=None,
                                          roll_off=0, chromatic_dispersion=0, pmd=0):
    """ Creates an arbitrary spectral information """
    if isinstance(frequency, Sized):
        frequency = squeeze(frequency)
    else:
        frequency = array([frequency])
    number_of_channels = frequency.size
    try:
        signal *= ones(number_of_channels)
        baud_rate *= ones(number_of_channels)
        roll_off *= ones(number_of_channels)
        slot_width = slot_width*ones(number_of_channels) if slot_width else \
            ceil((1+2*roll_off) * baud_rate / DEFAULT_SLOT_WIDTH_STEP) * DEFAULT_SLOT_WIDTH_STEP
        chromatic_dispersion *= ones(number_of_channels)
        pmd *= ones(number_of_channels)
        nli = zeros(number_of_channels)
        ase = zeros(number_of_channels)
        return SpectralInformation(frequency=frequency, slot_width=slot_width,
                                   signal=signal, nli=nli, ase=ase,
                                   baud_rate=baud_rate, roll_off=roll_off,
                                   chromatic_dispersion=chromatic_dispersion, pmd=pmd)
    except ValueError:
        raise SpectrumError('Dimension mismatch in input fields.')


def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, power, spacing):
    """ Creates a fixed slot width spectral information with flat power """
    nb_channel = automatic_nch(f_min, f_max, spacing)
    frequency = [(f_min + spacing * i) for i in range(1, nb_channel + 1)]
    return create_arbitrary_spectral_information(frequency, slot_width=spacing, signal=power, baud_rate=baud_rate,
                                                 roll_off=roll_off)
