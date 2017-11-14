#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:50:46 2017

@author: briantaylor
"""
import numpy as np
from numpy import pi, cos, sqrt

def c():
    """
    Returns the speed of light in meters per second
    """
    return 299792458


def wavelength2freq(value):
    return c()/value


def freq2wavelength(value):
    return c()/value


def deltawl2deltaf(delta_wl, wavelength):
    '''deltawl2deltaf(delta_wl, wavelength):
    delta_wl is BW in wavelength units
    wavelength is the center wl
    units for delta_wl and wavelength must be same
    '''
    f = wavelength2freq(wavelength)
    return delta_wl*f/wavelength


def deltaf2deltawl(delta_f, frequency):
    '''deltawl2deltaf(delta_f, frequency):
    delta_f is BW in HZ
    frequency is in HZ
    units for delta_wl and wavelength must be same
    '''
    wl = freq2wavelength(frequency)
    return delta_f*wl/frequency


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
    Ts = 1/baud_rate
    l_lim = (1 - alpha)/(2 * Ts)
    r_lim = (1 + alpha)/(2 * Ts)
    hf = np.zeros(np.shape(ffs))
    slope_inds = np.where(
            np.logical_and(np.abs(ffs) > l_lim, np.abs(ffs) < r_lim))
    hf[slope_inds] = 0.5 * (1 + cos((pi * Ts / alpha) * 
            (np.abs(ffs[slope_inds]) - l_lim)))
    p_inds = np.where(np.logical_and(np.abs(ffs) > 0, np.abs(ffs) < l_lim))
    hf[p_inds] =  1
    return sqrt(hf)