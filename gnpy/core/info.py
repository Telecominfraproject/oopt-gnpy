#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.core.info: classes for modelling Spectral Information
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.core.info
==============

This module contains classes for modelling :class:`SpectralInformation`.
"""

from __future__ import annotations
from collections import namedtuple
from collections.abc import Iterable
from typing import Union, List, Optional
from dataclasses import dataclass
from numpy import argsort, array, append, ones, ceil, any, zeros, outer, full, ndarray, \
    asarray

from gnpy.core.utils import automatic_nch, db2lin, watt2dbm, lin2db
from gnpy.core.exceptions import SpectrumError

DEFAULT_SLOT_WIDTH_STEP = 12.5e9  # Hz
"""Channels with unspecified slot width will have their slot width evaluated as the baud rate rounded up to the minimum
multiple of the DEFAULT_SLOT_WIDTH_STEP (the baud rate is extended including the roll off in this evaluation)"""


class Channel(
    namedtuple('Channel',
               'channel_number frequency baud_rate slot_width roll_off signal ase nli chromatic_dispersion '
               'pmd pdl latency')):
    """Class containing the parameters of a WDM signal.

    :param channel_number: channel number in the WDM grid
    :param frequency: central frequency of the signal (Hz)
    :param baud_rate: the symbol rate of the signal (Baud)
    :param slot_width: the slot width (Hz)
    :param roll_off: the roll off of the signal. It is a pure number between 0 and 1
    :param signal: signal power (dBm)
    :param ase: ASE noise power (dBm)
    :param nli: NLI noise power (dBm)
    :param chromatic_dispersion: chromatic dispersion (s/m)
    :param pmd: polarization mode dispersion (s)
    :param pdl: polarization dependent loss (dB)
    :param latency: propagation latency (s)
    """


class SpectralInformation(object):
    """Class containing the parameters of the entire WDM comb.

    delta_pdb_per_channel: (per frequency) per channel delta power in dbm for the actual mix of channels"""

    def __init__(self, frequency: array, baud_rate: array, slot_width: array, pch: array,
                 signal_ratio: array, ase_ratio: array, nli_ratio: array,
                 roll_off: array, chromatic_dispersion: array, pmd: array, pdl: array, latency: array,
                 delta_pdb_per_channel: array, tx_osnr: array, tx_power: array, label: array):
        indices = argsort(frequency)
        self._frequency = frequency[indices]
        self._df = outer(ones(frequency.shape), self._frequency) - outer(self._frequency, ones(frequency.shape))
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
        self._pch = pch[indices]
        self._signal_ratio = signal_ratio[indices]
        self._nli_ratio = nli_ratio[indices]
        self._ase_ratio = ase_ratio[indices]
        self._roll_off = roll_off[indices]
        self._chromatic_dispersion = chromatic_dispersion[indices]
        self._pmd = pmd[indices]
        self._pdl = pdl[indices]
        self._latency = latency[indices]
        self._delta_pdb_per_channel = delta_pdb_per_channel[indices]
        self._tx_osnr = tx_osnr[indices]
        self._tx_power = tx_power[indices]
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
    def pch(self):
        return array(self._pch)

    @property
    def pch_dbm(self):
        return watt2dbm(self.pch)

    @property
    def ptot(self):
        return sum(self._pch)

    @property
    def ptot_dbm(self):
        return watt2dbm(self.ptot)

    @pch.setter
    def pch(self, pch):
        self._pch = pch

    @property
    def signal(self):
        return self._signal_ratio * self._pch

    @property
    def signal_dbm(self):
        return watt2dbm(self.signal)

    @property
    def nli(self):
        return self._nli_ratio * self._pch

    @property
    def nli_dbm(self):
        return watt2dbm(self.nli)

    def add_nli(self, nli):
        # NLI power is interpreted exclusively as a power transfer from pch to nli thus it only affects the ratios
        nli_ratio = nli / self.pch
        self._signal_ratio *= (1 - nli_ratio)
        self._ase_ratio *= (1 - nli_ratio)
        self._nli_ratio = (self._nli_ratio * (1 - nli_ratio) + nli_ratio)

    @property
    def ase(self):
        return self._ase_ratio * self._pch

    @property
    def ase_dbm(self):
        return watt2dbm(self.ase)

    def add_ase(self, ase):
        pch = self.pch + ase
        self._signal_ratio *= self.pch / pch
        self._nli_ratio *= self.pch / pch
        self._ase_ratio = (self._ase_ratio * self.pch + ase) / pch
        self.pch = pch

    @property
    def snr_lin(self):
        return self._signal_ratio / self._ase_ratio

    @property
    def snr_lin_db(self):
        return lin2db(self.snr_lin)

    @property
    def opt_snr_lin_db(self):
        return self.snr_lin_db - lin2db(12.5e9/self.baud_rate)

    @property
    def snr_nli(self):
        return self._signal_ratio / self._nli_ratio

    @property
    def snr_nli_db(self):
        return lin2db(self.snr_nli)

    @property
    def opt_snr_nli_db(self):
        return self.snr_nli_db - lin2db(12.5e9/self.baud_rate)

    @property
    def gsnr(self):
        return self._signal_ratio / (self._ase_ratio + self._nli_ratio)

    @property
    def gsnr_db(self):
        return lin2db(self.gsnr)

    @property
    def opt_gsnr_db(self):
        return self.gsnr_db - lin2db(12.5e9/self.baud_rate)

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
    def tx_power(self):
        return self._tx_power

    @tx_power.setter
    def tx_power(self, tx_power):
        self._tx_power = tx_power

    @property
    def channel_number(self):
        return self._channel_number

    @property
    def carriers(self):
        entries = (zip(self.channel_number, self.frequency, self.baud_rate, self.slot_width, self.roll_off,
                       self.signal, self.ase, self.nli, self.chromatic_dispersion, self.pmd, self.pdl, self.latency))
        return [Channel(*entry) for entry in entries]

    def apply_attenuation_lin(self, attenuation_lin):
        self.pch *= attenuation_lin

    def apply_attenuation_db(self, attenuation_db):
        attenuation_lin = 1 / db2lin(attenuation_db)
        self.apply_attenuation_lin(attenuation_lin)

    def apply_gain_lin(self, gain_lin):
        self.pch *= gain_lin

    def apply_gain_db(self, gain_db):
        gain_lin = db2lin(gain_db)
        self.apply_gain_lin(gain_lin)

    def __add__(self, other: SpectralInformation):
        try:
            return SpectralInformation(frequency=append(self.frequency, other.frequency),
                                       slot_width=append(self.slot_width, other.slot_width),
                                       pch=append(self.pch, other.pch),
                                       signal_ratio=append(self._signal_ratio, other._signal_ratio),
                                       nli_ratio=append(self._nli_ratio, other._nli_ratio),
                                       ase_ratio=append(self._ase_ratio, other._ase_ratio),
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
                                       tx_power=append(self.tx_power, other.tx_power),
                                       label=append(self.label, other.label))
        except SpectrumError:
            raise SpectrumError('Spectra cannot be summed: channels overlapping.')


def create_arbitrary_spectral_information(frequency: Union[ndarray, Iterable, float],
                                          pch: Union[float, ndarray, Iterable],
                                          baud_rate: Union[float, ndarray, Iterable],
                                          tx_osnr: Union[float, ndarray, Iterable],
                                          tx_power: Union[float, ndarray, Iterable] = None,
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
        pch = full(number_of_channels, pch)
        baud_rate = full(number_of_channels, baud_rate)
        roll_off = full(number_of_channels, roll_off)
        slot_width = full(number_of_channels, slot_width) if slot_width is not None else \
            ceil((1 + roll_off) * baud_rate / DEFAULT_SLOT_WIDTH_STEP) * DEFAULT_SLOT_WIDTH_STEP
        chromatic_dispersion = full(number_of_channels, chromatic_dispersion)
        pmd = full(number_of_channels, pmd)
        pdl = full(number_of_channels, pdl)
        latency = full(number_of_channels, latency)
        signal_ratio = ones(number_of_channels)
        nli_ratio = zeros(number_of_channels)
        ase_ratio = zeros(number_of_channels)
        delta_pdb_per_channel = full(number_of_channels, delta_pdb_per_channel)
        tx_osnr = full(number_of_channels, tx_osnr)
        tx_power = full(number_of_channels, tx_power)
        label = full(number_of_channels, label)
        return SpectralInformation(frequency=frequency, slot_width=slot_width, pch=pch,
                                   signal_ratio=signal_ratio, nli_ratio=nli_ratio, ase_ratio=ase_ratio,
                                   baud_rate=baud_rate, roll_off=roll_off,
                                   chromatic_dispersion=chromatic_dispersion,
                                   pmd=pmd, pdl=pdl, latency=latency,
                                   delta_pdb_per_channel=delta_pdb_per_channel,
                                   tx_osnr=tx_osnr, tx_power=tx_power, label=label)
    except ValueError as e:
        if 'could not broadcast' in str(e):
            raise SpectrumError('Dimension mismatch in input fields.')
        raise e


def create_input_spectral_information(f_min, f_max, roll_off, baud_rate, spacing, tx_osnr, tx_power,
                                      delta_pdb=0):
    """Creates a fixed slot width spectral information with flat power.
    all arguments are scalar values"""
    number_of_channels = automatic_nch(f_min, f_max, spacing)
    frequency = [(f_min + spacing * i) for i in range(1, number_of_channels + 1)]
    delta_pdb_per_channel = delta_pdb * ones(number_of_channels)
    label = [f'{baud_rate * 1e-9 :.2f}G' for i in range(number_of_channels)]
    return create_arbitrary_spectral_information(frequency, slot_width=spacing, pch=tx_power, baud_rate=baud_rate,
                                                 roll_off=roll_off, delta_pdb_per_channel=delta_pdb_per_channel,
                                                 tx_osnr=tx_osnr, tx_power=tx_power, label=label)


def select_channels(spectrum: SpectralInformation, select: array) -> SpectralInformation:
    """
    select: boolean array of indices to keep
    """
    return SpectralInformation(frequency=spectrum.frequency[select], baud_rate=spectrum.baud_rate[select],
                               slot_width=spectrum.slot_width[select], pch=spectrum.pch[select],
                               signal_ratio=spectrum._signal_ratio[select], nli_ratio=spectrum._nli_ratio[select],
                               ase_ratio=spectrum._ase_ratio[select],
                               roll_off=spectrum.roll_off[select],
                               chromatic_dispersion=spectrum.chromatic_dispersion[select],
                               pmd=spectrum.pmd[select], pdl=spectrum.pdl[select], latency=spectrum.latency[select],
                               delta_pdb_per_channel=spectrum.delta_pdb_per_channel[select],
                               tx_osnr=spectrum.tx_osnr[select], tx_power=spectrum.tx_power[select],
                               label=spectrum.label[select])


def is_in_band(frequency: array, slot_width: array, band: dict) -> array:
    """band has {"f_min": value, "f_max": value} format
    """
    return (frequency - slot_width / 2 >= band['f_min']) * (frequency + slot_width / 2 <= band['f_max']) == 1


def demuxed_spectral_information(input_si: SpectralInformation, band: dict) -> Optional[SpectralInformation]:
    """extract a si based on band
    """

    select = is_in_band(input_si.frequency, input_si.slot_width, band)

    if any(select):
        spectrum = select_channels(input_si, select)
    else:
        spectrum = None
    return spectrum


def muxed_spectral_information(input_si_list: List[SpectralInformation]) -> SpectralInformation:
    """return the assembled spectrum
    """
    if input_si_list and len(input_si_list) > 1:
        si = input_si_list[0] + muxed_spectral_information(input_si_list[1:])
        return si
    if input_si_list and len(input_si_list) == 1:
        return input_si_list[0]
    raise ValueError('liste vide')


def carriers_to_spectral_information(initial_spectrum: dict[float, Carrier],
                                     power: float) -> SpectralInformation:
    """Initial spectrum is a dict with key = carrier frequency, and value a Carrier object.
    :param initial_spectrum: indexed by frequency in Hz, with power offset (delta_pdb), baudrate, slot width,
    tx_osnr, tx_power and roll off.
    :param power: power of the request
    """
    frequency = list(initial_spectrum.keys())
    pch = [c.tx_power for c in initial_spectrum.values()]
    roll_off = [c.roll_off for c in initial_spectrum.values()]
    baud_rate = [c.baud_rate for c in initial_spectrum.values()]
    delta_pdb_per_channel = [c.delta_pdb for c in initial_spectrum.values()]
    slot_width = [c.slot_width for c in initial_spectrum.values()]
    tx_osnr = [c.tx_osnr for c in initial_spectrum.values()]
    tx_power = [c.tx_power for c in initial_spectrum.values()]
    label = [c.label for c in initial_spectrum.values()]
    return create_arbitrary_spectral_information(frequency=frequency, pch=pch, baud_rate=baud_rate,
                                                 slot_width=slot_width, roll_off=roll_off,
                                                 delta_pdb_per_channel=delta_pdb_per_channel, tx_osnr=tx_osnr,
                                                 tx_power=tx_power, label=label)


@dataclass
class Carrier:
    """One channel in the initial mixed-type spectrum definition, each type being defined by
    its delta_pdb (power offset with respect to reference power), baud rate, slot_width, roll_off
    tx_power, and tx_osnr. delta_pdb offset is applied to target power out of Roadm.
    Label is used to group carriers which belong to the same partition when printing results.
    """
    delta_pdb: float
    baud_rate: float
    slot_width: float
    roll_off: float
    tx_osnr: float
    tx_power: float
    label: str


@dataclass
class ReferenceCarrier:
    """Reference channel type is used to determine target power out of ROADM for the reference channel when
    constant power spectral density (PSD) equalization is set. Reference channel is the type that has been defined
    in SI block and used for the initial design of the network.
    Computing the power out of ROADM for the reference channel is required to correctly compute the loss
    experienced by reference channel in Roadm element.

    Baud rate is required to find the target power in constant PSD: power = PSD_target * baud_rate.
    For example, if target PSD is 3.125e4mW/GHz and reference carrier type a 32 GBaud channel then
    output power should be -20 dBm and for a 64 GBaud channel power target would need 3 dB more: -17 dBm.

    Slot width is required to find the target power in constant PSW (constant power per slot width equalization):
    power = PSW_target * slot_width.
    For example, if target PSW is 2e4mW/GHz and reference carrier type a 32 GBaud channel in a 50GHz slot width then
    output power should be -20 dBm and for a 64 GBaud channel  in a 75 GHz slot width, power target would be -18.24 dBm.

    Other attributes (like roll-off) may be added there for future equalization purpose.
    """
    baud_rate: float
    slot_width: float
