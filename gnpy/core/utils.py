#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.utils
===============

This module contains utility functions that are used with gnpy.
"""

from csv import writer
from numpy import pi, cos, sqrt, log10, linspace, zeros, shape, where, logical_and
from scipy import constants

from gnpy.core.exceptions import ConfigurationError


def write_csv(obj, filename):
    """
    Convert dictionary items to a CSV file the dictionary format:
    ::

        {'result category 1':
                        [
                        # 1st line of results
                        {'header 1' : value_xxx,
                         'header 2' : value_yyy},
                         # 2nd line of results: same headers, different results
                        {'header 1' : value_www,
                         'header 2' : value_zzz}
                        ],
        'result_category 2':
                        [
                        {},{}
                        ]
        }

    The generated csv file will be:
    ::

        result_category 1
        header 1    header 2
        value_xxx   value_yyy
        value_www   value_zzz
        result_category 2
        ...
    """
    with open(filename, 'w', encoding='utf-8') as f:
        w = writer(f)
        for data_key, data_list in obj.items():
            # main header
            w.writerow([data_key])
            # sub headers:
            headers = [_ for _ in data_list[0].keys()]
            w.writerow(headers)
            for data_dict in data_list:
                w.writerow([_ for _ in data_dict.values()])


def arrange_frequencies(length, start, stop):
    """Create an array of frequencies

    :param length: number of elements
    :param start: Start frequency in THz
    :param stop: Stop frequency in THz
    :type length: integer
    :type start: float
    :type stop: float
    :return: an array of frequencies determined by the spacing parameter
    :rtype: numpy.ndarray
    """
    return linspace(start, stop, length)


def lin2db(value):
    """Convert linear unit to logarithmic (dB)

    >>> lin2db(0.001)
    -30.0
    >>> round(lin2db(1.0), 2)
    0.0
    >>> round(lin2db(1.26), 2)
    1.0
    >>> round(lin2db(10.0), 2)
    10.0
    >>> round(lin2db(100.0), 2)
    20.0
    """
    return 10 * log10(value)


def db2lin(value):
    """Convert logarithimic units to linear

    >>> round(db2lin(10.0), 2)
    10.0
    >>> round(db2lin(20.0), 2)
    100.0
    >>> round(db2lin(1.0), 2)
    1.26
    >>> round(db2lin(0.0), 2)
    1.0
    >>> round(db2lin(-10.0), 2)
    0.1
    """
    return 10**(value / 10)


def round2float(number, step):
    step = round(step, 1)
    if step >= 0.01:
        number = round(number / step, 0)
        number = round(number * step, 1)
    else:
        number = round(number, 2)
    return number


wavelength2freq = constants.lambda2nu
freq2wavelength = constants.nu2lambda


def freq2wavelength(value):
    """ Converts frequency units to wavelength units.

    >>> round(freq2wavelength(191.35e12) * 1e9, 3)
    1566.723
    >>> round(freq2wavelength(196.1e12) * 1e9, 3)
    1528.773
    """
    return constants.c / value


def snr_sum(snr, bw, snr_added, bw_added=12.5e9):
    snr_added = snr_added - lin2db(bw / bw_added)
    snr = -lin2db(db2lin(-snr) + db2lin(-snr_added))
    return snr


def deltawl2deltaf(delta_wl, wavelength):
    """ deltawl2deltaf(delta_wl, wavelength):
    delta_wl is BW in wavelength units
    wavelength is the center wl
    units for delta_wl and wavelength must be same

    :param delta_wl: delta wavelength BW in same units as wavelength
    :param wavelength: wavelength BW is relevant for
    :type delta_wl: float or numpy.ndarray
    :type wavelength: float
    :return: The BW in frequency units
    :rtype: float or ndarray

    """
    f = wavelength2freq(wavelength)
    return delta_wl * f / wavelength


def deltaf2deltawl(delta_f, frequency):
    """ deltawl2deltaf(delta_f, frequency):
        converts delta frequency to delta wavelength
        units for delta_wl and wavelength must be same

    :param delta_f: delta frequency in same units as frequency
    :param frequency: frequency BW is relevant for
    :type delta_f: float or numpy.ndarray
    :type frequency: float
    :return: The BW in wavelength units
    :rtype: float or ndarray

    """
    wl = freq2wavelength(frequency)
    return delta_f * wl / frequency


def rrc(ffs, baud_rate, alpha):
    """ rrc(ffs, baud_rate, alpha): computes the root-raised cosine filter
    function.

    :param ffs: A numpy array of frequencies
    :param baud_rate: The Baud Rate of the System
    :param alpha: The roll-off factor of the filter
    :type ffs: numpy.ndarray
    :type baud_rate: float
    :type alpha: float
    :return: hf a numpy array of the filter shape
    :rtype: numpy.ndarray

    """
    Ts = 1 / baud_rate
    l_lim = (1 - alpha) / (2 * Ts)
    r_lim = (1 + alpha) / (2 * Ts)
    hf = zeros(shape(ffs))
    slope_inds = where(
        logical_and(abs(ffs) > l_lim, abs(ffs) < r_lim))
    hf[slope_inds] = 0.5 * (1 + cos((pi * Ts / alpha) *
                                    (abs(ffs[slope_inds]) - l_lim)))
    p_inds = where(logical_and(abs(ffs) > 0, abs(ffs) < l_lim))
    hf[p_inds] = 1
    return sqrt(hf)


def merge_amplifier_restrictions(dict1, dict2):
    """Updates contents of dicts recursively

    >>> d1 = {'params': {'restrictions': {'preamp_variety_list': [], 'booster_variety_list': []}}}
    >>> d2 = {'params': {'target_pch_out_db': -20}}
    >>> merge_amplifier_restrictions(d1, d2)
    {'params': {'restrictions': {'preamp_variety_list': [], 'booster_variety_list': []}, 'target_pch_out_db': -20}}

    >>> d3 = {'params': {'restrictions': {'preamp_variety_list': ['foo'], 'booster_variety_list': ['bar']}}}
    >>> merge_amplifier_restrictions(d1, d3)
    {'params': {'restrictions': {'preamp_variety_list': [], 'booster_variety_list': []}}}
    """

    copy_dict1 = dict1.copy()
    for key in dict2:
        if key in dict1:
            if isinstance(dict1[key], dict):
                copy_dict1[key] = merge_amplifier_restrictions(copy_dict1[key], dict2[key])
        else:
            copy_dict1[key] = dict2[key]
    return copy_dict1


def silent_remove(this_list, elem):
    """Remove matching elements from a list without raising ValueError

    >>> li = [0, 1]
    >>> li = silent_remove(li, 1)
    >>> li
    [0]
    >>> li = silent_remove(li, 1)
    >>> li
    [0]
    """

    try:
        this_list.remove(elem)
    except ValueError:
        pass
    return this_list


def automatic_nch(f_min, f_max, spacing):
    """How many channels are available in the spectrum

    :param f_min Lowest frequenecy [Hz]
    :param f_max Highest frequency [Hz]
    :param spacing Channel width [Hz]
    :return Number of uniform channels

    >>> automatic_nch(191.325e12, 196.125e12, 50e9)
    96
    >>> automatic_nch(193.475e12, 193.525e12, 50e9)
    1
    """
    return int((f_max - f_min) // spacing)


def automatic_fmax(f_min, spacing, nch):
    """Find the high-frequenecy boundary of a spectrum

    :param f_min Start of the spectrum (lowest frequency edge) [Hz]
    :param spacing Grid/channel spacing [Hz]
    :param nch Number of channels
    :return End of the spectrum (highest frequency) [Hz]

    >>> automatic_fmax(191.325e12, 50e9, 96)
    196125000000000.0
    """
    return f_min + spacing * nch


def convert_length(value, units):
    """Convert length into basic SI units

    >>> convert_length(1, 'km')
    1000.0
    >>> convert_length(2.0, 'km')
    2000.0
    >>> convert_length(123, 'm')
    123.0
    >>> convert_length(123.0, 'm')
    123.0
    >>> convert_length(42.1, 'km')
    42100.0
    >>> convert_length(666, 'yards')
    Traceback (most recent call last):
        ...
    gnpy.core.exceptions.ConfigurationError: Cannot convert length in "yards" into meters
    """
    if units == 'm':
        return value * 1e0
    elif units == 'km':
        return value * 1e3
    else:
        raise ConfigurationError(f'Cannot convert length in "{units}" into meters')
