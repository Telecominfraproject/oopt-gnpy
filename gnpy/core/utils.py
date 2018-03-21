#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.utils
===============

This module contains utility functions that are used with gnpy.
'''


import json

import numpy as np
from numpy import pi, cos, sqrt, log10


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def save_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def c():
    """
    Returns the speed of light in meters per second
    """
    return 299792458.0


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


def h():
    """
    Returns plank's constant in J*s
    """
    return 6.62607004e-34


def lin2db(value):
    return 10 * log10(value)


def db2lin(value):
    return 10**(value / 10)


def wavelength2freq(value):
    """ Converts wavelength units to frequeuncy units.
    """
    return c() / value


def freq2wavelength(value):
    """ Converts frequency units to wavelength units.
    """
    return c() / value


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
