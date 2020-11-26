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
from gnpy.core.exceptions import InfoError

BASE_SLOT_WIDTH = 12.5e9  # Hz


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


class Pref(namedtuple('Pref', 'p_span0, p_spani, p_span0_per_channel ')):
    """noiseless reference power in dBm:
    p_span0: inital target carrier power for a reference channel defined by user
    p_spani: carrier power after element i for a reference channel defined by user
    neq_ch: equivalent channel count in dB -> almost never used : suppress it
    p_span0_per_channel: per frequency target per channel power"""


class SpectralInformation(object):
    """ Class containing the parameters of the entire WDM comb."""
    def __init__(self, frequency, baud_rate, slot_width, signal, nli, ase,
                 roll_off, chromatic_dispersion, pmd, ref_power):
        indices = argsort(frequency)
        self._frequency = frequency[indices]
        self._df = outer(ones(frequency.shape), frequency) - outer(frequency, ones(frequency.shape))
        self._number_of_channels = len(self._frequency)
        self._slot_width = slot_width[indices]
        self._baud_rate = baud_rate[indices]
        if any(self._frequency[:-1] + self._slot_width[:-1] / 2 > self._frequency[1:] - self._slot_width[1:] / 2):
            raise InfoError('Spectrum required slot widths larger than the frequency spectral distances.')
        elif any(self._baud_rate > self._slot_width):
            raise InfoError('Spectrum baud rate larger than the slot width.')
        self._signal = signal[indices]
        self._nli = nli[indices]
        self._ase = ase[indices]
        self._roll_off = roll_off[indices]
        self._chromatic_dispersion = chromatic_dispersion[indices]
        self._pmd = pmd[indices]
        self._channel_number = [*range(1, self._number_of_channels + 1)]
        self._pref = ref_power

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
    def df(self):
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
    def gsnr(self):
        return lin2db(self.signal/(self.ase+self.nli))

    @property
    def osnr(self):
        return lin2db(self.signal/self.ase)

    @property
    def snr(self):
        return lin2db(self.signal/self.nli)

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
        frequency = append(self.frequency, si.frequency)
        baud_rate = append(self.baud_rate, si.baud_rate)
        slot_width = append(self.slot_width, si.slot_width)
        signal = append(self.signal, si.signal)
        nli = append(self.nli, si.nli)
        ase = append(self.ase, si.ase)
        roll_off = append(self.roll_off, si.roll_off)
        chromatic_dispersion = append(self.chromatic_dispersion, si.chromatic_dispersion)
        pmd = append(self.pmd, si.pmd)
        pref = Pref(self.pref.p_span0, self.pref.p_spani,
            {**self.pref.p_span0_per_channel, **si.pref.p_span0_per_channel})
        try:
            si = SpectralInformation(frequency=frequency, slot_width=slot_width,
                                     signal=signal, nli=nli, ase=ase,
                                     baud_rate=baud_rate, roll_off=roll_off,
                                     chromatic_dispersion=chromatic_dispersion, pmd=pmd,
                                     ref_power=pref)
        except InfoError:
            raise InfoError('Spectra cannot be summed: channels overlapping.')
        return si

    def _replace(self, carriers, pref):
        self.chromatic_dispersion = array([c.chromatic_dispersion for c in carriers])
        self.pmd = array([c.pmd for c in carriers])
        self.signal = array([c.power.signal for c in carriers])
        self.nli = array([c.power.nli for c in carriers])
        self.ase = array([c.power.ase for c in carriers])
        self.pref = pref
        return self


def dimension_reshape(value, dimension, default=None, name=None):
    """Check or reshape value over the given dimension. Apply the default value if given and value is None.

    @param value: array or single value
    @param dimension: required dimension
    @param default: given default value
    @param name: entry name, used to exception management
    @return: checked or reshaped array with given dimension
    @raise: InfoError instatnce of gnpy.core.exceptions.InfoError
            A mandatory entry must be provided and no default is given, thus, if value is None
            >>> dimension_reshape(None, dimension=3, default=None, name='mandatory_entry')
            Traceback (most recent call last):
            gnpy.core.exceptions.InfoError: Missing mandatory field: mandatory_entry.

            If value is an array with the wrong dimension it cannot be reshaped, thus, if value.size != dimension
            >>> dimension_reshape(value=[1,2], dimension=3, default=None, name='entry')
            Traceback (most recent call last):
            gnpy.core.exceptions.InfoError: Dimension mismatch field: entry.
    """
    if value is None:
        if default is not None:
            value = dimension_reshape(value=default, dimension=dimension, name=name)
        else:
            raise InfoError(f'Missing mandatory field: {name}.')
    elif not isinstance(value, Sized):
        value = value * ones(dimension)
    else:
        if len(value) == 1:
            value = value[0] * ones(dimension)
        elif len(value) == dimension:
            value = squeeze(value)
        else:
            raise InfoError(f'Dimension mismatch field: {name}.')
    return value


def create_arbitrary_spectral_information(frequency, slot_width=None, signal=None, baud_rate=None,
                                          roll_off=None, chromatic_dispersion=None, pmd=None, ref_power=None):
    """ Creates an arbitrary spectral information """
    if isinstance(frequency, Sized):
        frequency = squeeze(frequency)
        number_of_channels = len(frequency)
    else:
        frequency = array([frequency])
        number_of_channels = 1
    baud_rate = dimension_reshape(baud_rate, number_of_channels, None, 'baud rate')
    slot_width = dimension_reshape(slot_width, number_of_channels, ceil(baud_rate / BASE_SLOT_WIDTH) * BASE_SLOT_WIDTH,
                                   'slot_width')
    signal = dimension_reshape(signal, number_of_channels, 0, 'signal')
    nli = zeros(number_of_channels)
    ase = zeros(number_of_channels)
    roll_off = dimension_reshape(roll_off, number_of_channels, 0, 'roll_off')
    chromatic_dispersion = dimension_reshape(chromatic_dispersion, number_of_channels, 0, 'chromatic dispersion')
    pmd = dimension_reshape(pmd, number_of_channels, 0, 'pmd')
    si = SpectralInformation(frequency=frequency, slot_width=slot_width,
                             signal=signal, nli=nli, ase=ase,
                             baud_rate=baud_rate, roll_off=roll_off,
                             chromatic_dispersion=chromatic_dispersion, pmd=pmd,
                             ref_power=ref_power)
    return si


def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, power, spacing):
    """ Creates a fixed slot width spectral information with flat power.
    all arguments are scalar values"""
    nb_channel = automatic_nch(f_min, f_max, spacing)
    frequency = [(f_min + spacing * i) for i in range(1, nb_channel + 1)]
    p_span0 = lin2db(power * 1e3)
    p_spani = lin2db(power * 1e3)
    p_span0_per_channel = {f: lin2db(power * 1e3) for f in frequency}
    si = create_arbitrary_spectral_information(
        frequency, slot_width=spacing, signal=power, baud_rate=baud_rate,
        roll_off=roll_off, ref_power=Pref(p_span0, p_spani, p_span0_per_channel))
    return si


def use_initial_spectrum(initial_spectrum, ref_power):
    """ initial spectrum is a dict with key = carrier frequency, and value a dict with power,
    baudrate and roll off for this carrier. ref_power is a Pref object with the power used for the reference channel
    """
    frequency = [f for f in initial_spectrum.keys()]
    signal = [s['power'] for s in initial_spectrum.values()]
    roll_off = [s['roll_off'] for s in initial_spectrum.values()]
    baud_rate = [s['baud_rate'] for s in initial_spectrum.values()]
    p_span0 = lin2db(ref_power * 1e3)
    p_spani = lin2db(ref_power * 1e3)
    p_span0_per_channel = {f: lin2db(s['power'] * 1e3) for f, s in initial_spectrum.items()}
    si = create_arbitrary_spectral_information(frequency=frequency, signal=signal, baud_rate=baud_rate,
                                               roll_off=roll_off, ref_power=Pref(p_span0, p_spani, p_span0_per_channel))
    return si
