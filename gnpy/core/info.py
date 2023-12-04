#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.info
==============

This module contains classes for modelling :class:`SpectralInformation`.
"""

from __future__ import annotations
from collections import namedtuple
from collections.abc import Iterable
from typing import Union
from dataclasses import dataclass
from numpy import argsort, mean, array, append, ones, ceil, any, zeros, outer, full, ndarray, asarray

from gnpy.core.utils import automatic_nch, db2lin, watt2dbm
from gnpy.core.exceptions import SpectrumError

DEFAULT_SLOT_WIDTH_STEP = 12.5e9  # Hz
"""Channels with unspecified slot width will have their slot width evaluated as the baud rate rounded up to the minimum
multiple of the DEFAULT_SLOT_WIDTH_STEP (the baud rate is extended including the roll off in this evaluation)"""


class Power(namedtuple('Power', 'signal nli ase')):
    """carriers power in W"""


class Channel(
    namedtuple('Channel',
               'channel_number frequency baud_rate slot_width roll_off power chromatic_dispersion pmd pdl latency')):
    """Class containing the parameters of a WDM signal.

    :param channel_number: channel number in the WDM grid
    :param frequency: central frequency of the signal (Hz)
    :param baud_rate: the symbol rate of the signal (Baud)
    :param slot_width: the slot width (Hz)
    :param roll_off: the roll off of the signal. It is a pure number between 0 and 1
    :param power (gnpy.core.info.Power): power of signal, ASE noise and NLI (W)
    :param chromatic_dispersion: chromatic dispersion (s/m)
    :param pmd: polarization mode dispersion (s)
    :param pdl: polarization dependent loss (dB)
    :param latency: propagation latency (s)
    """


class SpectralInformation(object):
    """Class containing the parameters of the entire WDM comb.

    delta_pdb_per_channel: (per frequency) per channel delta power in dbm for the actual mix of channels"""

    def __init__(self, frequency: array, baud_rate: array, slot_width: array, signal: array, nli: array, ase: array,
                 roll_off: array, chromatic_dispersion: array, pmd: array, pdl: array, latency: array,
                 delta_pdb_per_channel: array, tx_osnr: array, label: array):
        indices = argsort(frequency)
        self._frequency = frequency[indices]
        self._df = outer(ones(frequency.shape), frequency) - outer(frequency, ones(frequency.shape))
        self._number_of_channels = len(self._frequency)
        self._channel_number = [*range(1, self._number_of_channels + 1)]
        self._slot_width = slot_width[indices]
        self._baud_rate = baud_rate[indices]
        overlap = self._frequency[:-1] + self._slot_width[:-1] / 2 > self._frequency[1:] - self._slot_width[1:] / 2
        if any(overlap):
            overlap = [pair for pair in zip(overlap * self._channel_number[:-1], overlap * self._channel_number[1:])
                       if pair != (0, 0)]
            raise SpectrumError(f'Spectrum required slot widths larger than the frequency spectral distances '
                                f'between channels: {overlap}.')
        exceed = self._baud_rate > self._slot_width
        if any(exceed):
            raise SpectrumError(f'Spectrum baud rate, including the roll off, larger than the slot width for channels: '
                                f'{[ch for ch in exceed * self._channel_number if ch]}.')
        self._signal = signal[indices]
        self._nli = nli[indices]
        self._ase = ase[indices]
        self._roll_off = roll_off[indices]
        self._chromatic_dispersion = chromatic_dispersion[indices]
        self._pmd = pmd[indices]
        self._pdl = pdl[indices]
        self._latency = latency[indices]
        self._delta_pdb_per_channel = delta_pdb_per_channel[indices]
        self._tx_osnr = tx_osnr[indices]
        self._label = label[indices]

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

    @property
    def label(self):
        return self._label

    @pmd.setter
    def pmd(self, pmd):
        self._pmd = pmd

    @property
    def pdl(self):
        return self._pdl

    @pdl.setter
    def pdl(self, pdl):
        self._pdl = pdl

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def delta_pdb_per_channel(self):
        return self._delta_pdb_per_channel

    @delta_pdb_per_channel.setter
    def delta_pdb_per_channel(self, delta_pdb_per_channel):
        self._delta_pdb_per_channel = delta_pdb_per_channel

    @property
    def tx_osnr(self):
        return self._tx_osnr

    @tx_osnr.setter
    def tx_osnr(self, tx_osnr):
        self._tx_osnr = tx_osnr

    @property
    def channel_number(self):
        return self._channel_number

    @property
    def carriers(self):
        entries = zip(self.channel_number, self.frequency, self.baud_rate, self.slot_width,
                      self.roll_off, self.powers, self.chromatic_dispersion, self.pmd, self.pdl, self.latency)
        return [Channel(*entry) for entry in entries]

    def apply_attenuation_lin(self, attenuation_lin):
        self.signal *= attenuation_lin
        self.nli *= attenuation_lin
        self.ase *= attenuation_lin

    def apply_attenuation_db(self, attenuation_db):
        attenuation_lin = 1 / db2lin(attenuation_db)
        self.apply_attenuation_lin(attenuation_lin)

    def apply_gain_lin(self, gain_lin):
        self.signal *= gain_lin
        self.nli *= gain_lin
        self.ase *= gain_lin

    def apply_gain_db(self, gain_db):
        gain_lin = db2lin(gain_db)
        self.apply_gain_lin(gain_lin)

    def __add__(self, other: SpectralInformation):
        try:
            return SpectralInformation(frequency=append(self.frequency, other.frequency),
                                       slot_width=append(self.slot_width, other.slot_width),
                                       signal=append(self.signal, other.signal), nli=append(self.nli, other.nli),
                                       ase=append(self.ase, other.ase),
                                       baud_rate=append(self.baud_rate, other.baud_rate),
                                       roll_off=append(self.roll_off, other.roll_off),
                                       chromatic_dispersion=append(self.chromatic_dispersion,
                                                                   other.chromatic_dispersion),
                                       pmd=append(self.pmd, other.pmd),
                                       pdl=append(self.pdl, other.pdl),
                                       latency=append(self.latency, other.latency),
                                       delta_pdb_per_channel=append(self.delta_pdb_per_channel,
                                                                    other.delta_pdb_per_channel),
                                       tx_osnr=append(self.tx_osnr, other.tx_osnr),
                                       label=append(self.label, other.label))
        except SpectrumError:
            raise SpectrumError('Spectra cannot be summed: channels overlapping.')

    def _replace(self, carriers):
        self.chromatic_dispersion = array([c.chromatic_dispersion for c in carriers])
        self.pmd = array([c.pmd for c in carriers])
        self.pdl = array([c.pdl for c in carriers])
        self.latency = array([c.latency for c in carriers])
        self.signal = array([c.power.signal for c in carriers])
        self.nli = array([c.power.nli for c in carriers])
        self.ase = array([c.power.ase for c in carriers])
        return self


def create_arbitrary_spectral_information(frequency: Union[ndarray, Iterable, float],
                                          signal: Union[float, ndarray, Iterable],
                                          baud_rate: Union[float, ndarray, Iterable],
                                          tx_osnr: Union[float, ndarray, Iterable],
                                          delta_pdb_per_channel: Union[float, ndarray, Iterable] = 0.,
                                          slot_width: Union[float, ndarray, Iterable] = None,
                                          roll_off: Union[float, ndarray, Iterable] = 0.,
                                          chromatic_dispersion: Union[float, ndarray, Iterable] = 0.,
                                          pmd: Union[float, ndarray, Iterable] = 0.,
                                          pdl: Union[float, ndarray, Iterable] = 0.,
                                          latency: Union[float, ndarray, Iterable] = 0.,
                                          label: Union[str, ndarray, Iterable] = None):
    """This is just a wrapper around the SpectralInformation.__init__() that simplifies the creation of
    a non-uniform spectral information with NLI and ASE powers set to zero."""
    frequency = asarray(frequency)
    number_of_channels = frequency.size
    try:
        signal = full(number_of_channels, signal)
        baud_rate = full(number_of_channels, baud_rate)
        roll_off = full(number_of_channels, roll_off)
        slot_width = full(number_of_channels, slot_width) if slot_width is not None else \
            ceil((1 + roll_off) * baud_rate / DEFAULT_SLOT_WIDTH_STEP) * DEFAULT_SLOT_WIDTH_STEP
        chromatic_dispersion = full(number_of_channels, chromatic_dispersion)
        pmd = full(number_of_channels, pmd)
        pdl = full(number_of_channels, pdl)
        latency = full(number_of_channels, latency)
        nli = zeros(number_of_channels)
        ase = zeros(number_of_channels)
        delta_pdb_per_channel = full(number_of_channels, delta_pdb_per_channel)
        tx_osnr = full(number_of_channels, tx_osnr)
        label = full(number_of_channels, label)
        return SpectralInformation(frequency=frequency, slot_width=slot_width,
                                   signal=signal, nli=nli, ase=ase,
                                   baud_rate=baud_rate, roll_off=roll_off,
                                   chromatic_dispersion=chromatic_dispersion,
                                   pmd=pmd, pdl=pdl, latency=latency,
                                   delta_pdb_per_channel=delta_pdb_per_channel,
                                   tx_osnr=tx_osnr, label=label)
    except ValueError as e:
        if 'could not broadcast' in str(e):
            raise SpectrumError('Dimension mismatch in input fields.')
        else:
            raise


def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, power, spacing, tx_osnr, delta_pdb=0):
    """Creates a fixed slot width spectral information with flat power.
    all arguments are scalar values"""
    number_of_channels = automatic_nch(f_min, f_max, spacing)
    frequency = [(f_min + spacing * i) for i in range(1, number_of_channels + 1)]
    delta_pdb_per_channel = delta_pdb * ones(number_of_channels)
    label = [f'{baud_rate * 1e-9 :.2f}G' for i in range(number_of_channels)]
    return create_arbitrary_spectral_information(frequency, slot_width=spacing, signal=power, baud_rate=baud_rate,
                                                 roll_off=roll_off, delta_pdb_per_channel=delta_pdb_per_channel,
                                                 tx_osnr=tx_osnr, label=label)


def carriers_to_spectral_information(initial_spectrum: dict[float, Carrier],
                                     power: float) -> SpectralInformation:
    """Initial spectrum is a dict with key = carrier frequency, and value a Carrier object.
    :param initial_spectrum: indexed by frequency in Hz, with power offset (delta_pdb), baudrate, slot width,
    tx_osnr and roll off.
    :param power: power of the request
    """
    frequency = list(initial_spectrum.keys())
    signal = [power * db2lin(c.delta_pdb) for c in initial_spectrum.values()]
    roll_off = [c.roll_off for c in initial_spectrum.values()]
    baud_rate = [c.baud_rate for c in initial_spectrum.values()]
    delta_pdb_per_channel = [c.delta_pdb for c in initial_spectrum.values()]
    slot_width = [c.slot_width for c in initial_spectrum.values()]
    tx_osnr = [c.tx_osnr for c in initial_spectrum.values()]
    label = [c.label for c in initial_spectrum.values()]
    p_span0 = watt2dbm(power)
    return create_arbitrary_spectral_information(frequency=frequency, signal=signal, baud_rate=baud_rate,
                                                 slot_width=slot_width, roll_off=roll_off,
                                                 delta_pdb_per_channel=delta_pdb_per_channel, tx_osnr=tx_osnr,
                                                 label=label)


@dataclass
class Carrier:
    """One channel in the initial mixed-type spectrum definition, each type being defined by
    its delta_pdb (power offset with respect to reference power), baud rate, slot_width, roll_off
    and tx_osnr. delta_pdb offset is applied to target power out of Roadm.
    Label is used to group carriers which belong to the same partition when printing results.
    """
    delta_pdb: float
    baud_rate: float
    slot_width: float
    roll_off: float
    tx_osnr: float
    label: str


@dataclass
class ReferenceCarrier:
    """Reference channel type is used to determine target power out of ROADM for the reference channel when
    constant power spectral density (PSD) equalization is set. Reference channel is the type that has been defined
    in SI block and used for the initial design of the network.
    Computing the power out of ROADM for the reference channel is required to correctly compute the loss
    experienced by p_span_i in Roadm element.

    Baud rate is required to find the target power in constant PSD: power = PSD_target * baud_rate.
    For example, if target PSD is 3.125e4mW/GHz and reference carrier type a 32 GBaud channel then
    output power should be -20 dBm and for a 64 GBaud channel power target would need 3 dB more: -17 dBm.

    Slot width is required to find the target power in constant PSW (constant power per slot width equalization):
    power = PSW_target * slot_width.
    For example, if target PSW is 2e4mW/GHz and reference carrier type a 32 GBaud channel in a 50GHz slot width then
    output power should be -20 dBm and for a 64 GBaud channel  in a 75 GHz slot width, power target would be -18.24 dBm.

    Other attributes (like slot_width or roll-off) may be added there for future equalization purpose.
    """
    baud_rate: float
    slot_width: float
