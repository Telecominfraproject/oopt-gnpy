# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:46:19 2017

@author: briantaylor
"""
import numpy as np


def generic_box_psd():
    """
    creates a generic rectangular PSD at the channel spacing and baud rate
    TODO: convert input to kwargs
    Input is in THz (for now).  Also normalizes the total power to 1 over the
    band of interest.
    """
    baud = 0.034
    ffs = np.arange(193.95, 194.5, 0.05)
    zffs = 1e-6
    grid = []
    power = []
    """
    TODO: The implementation below is awful. Please fix.
    """
    for ea in ffs:
        fl1 = ea - baud/2 - zffs
        fl = ea - baud/2
        fr = ea + baud/2
        fr1 = ea + baud/2 + zffs
        grid = grid + [fl1, fl, ea, fr, fr1]
        power = power + [0, 1, 1, 1, 0]
    grid = np.array(grid)
    power = np.power(power)/np.sum(power)
    data = np.hstack(grid, power)
    return data
