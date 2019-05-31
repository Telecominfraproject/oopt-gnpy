#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.utils
===============

This module contains utility functions that are used with gnpy.
'''


import json

from csv import writer
import numpy as np
from numpy import pi, cos, sqrt, log10
from scipy import constants


def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_csv(obj, filename):
    """
    convert dictionary items to a csv file
    the dictionary format :

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

    the generated csv file will be:
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
            #main header
            w.writerow([data_key])
            #sub headers:
            headers = [_ for _ in data_list[0].keys()]
            w.writerow(headers)
            for data_dict in data_list:
                w.writerow([_ for _ in data_dict.values()])

def c():
    """
    Returns the speed of light in meters per second
    """
    return constants.c


def itufs(spacing, startf=191.35, stopf=196.10):
    """Creates an array of frequencies whose default range is
    191.35-196.10 THz

    :param spacing: Frequency spacing in THz
    :param starf: Start frequency in THz
    :param stopf: Stop frequency in THz
    :type spacing: float
    :type startf: float
    :type stopf: float
    :return an array of frequnecies determined by the spacing parameter
    :rtype: numpy.ndarray
    """
    return np.arange(startf, stopf + spacing / 2, spacing)

def itufl(length, startf=191.35, stopf=196.10):
    """Creates an array of frequencies whose default range is
    191.35-196.10 THz

    :param length: number of elements
    :param starf: Start frequency in THz
    :param stopf: Stop frequency in THz
    :type length: integer
    :type startf: float
    :type stopf: float
    :return an array of frequnecies determined by the spacing parameter
    :rtype: numpy.ndarray
    """
    return np.linspace(startf, stopf, length)

def h():
    """
    Returns plank's constant in J*s
    """
    return constants.h


def lin2db(value):
    return 10 * log10(value)


def db2lin(value):
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
    """
    return c() / value

def snr_sum(snr, bw, snr_added, bw_added=12.5e9):
    snr_added = snr_added - lin2db(bw/bw_added)
    snr = -lin2db(db2lin(-snr)+db2lin(-snr_added))
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
    hf = np.zeros(np.shape(ffs))
    slope_inds = np.where(
        np.logical_and(np.abs(ffs) > l_lim, np.abs(ffs) < r_lim))
    hf[slope_inds] = 0.5 * (1 + cos((pi * Ts / alpha) *
                                    (np.abs(ffs[slope_inds]) - l_lim)))
    p_inds = np.where(np.logical_and(np.abs(ffs) > 0, np.abs(ffs) < l_lim))
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
