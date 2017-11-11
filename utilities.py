#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:50:46 2017

@author: briantaylor
"""

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