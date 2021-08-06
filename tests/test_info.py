#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from numpy import array, zeros, ones
from numpy.testing import assert_array_equal
from gnpy.core.info import create_arbitrary_spectral_information, Pref
from gnpy.core.exceptions import SpectrumError


def test_create_arbitrary_spectral_information():
    si = create_arbitrary_spectral_information(frequency=[193.25e12, 193.3e12, 193.35e12],
                                               baud_rate=32e9, signal=[1, 1, 1],
                                               delta_pdb_per_channel=[1, 1, 1],
                                               tx_osnr=40.0,
                                               ref_power=Pref(1, 1, None))
    assert_array_equal(si.baud_rate, array([32e9, 32e9, 32e9]))
    assert_array_equal(si.slot_width, array([37.5e9, 37.5e9, 37.5e9]))
    assert_array_equal(si.signal, ones(3))
    assert_array_equal(si.nli, zeros(3))
    assert_array_equal(si.ase, zeros(3))
    assert_array_equal(si.delta_pdb_per_channel, ones(3))
    assert_array_equal(si.roll_off, zeros(3))
    assert_array_equal(si.chromatic_dispersion, zeros(3))
    assert_array_equal(si.pmd, zeros(3))
    assert_array_equal(si.channel_number, array([1, 2, 3]))
    assert_array_equal(si.number_of_channels, 3)
    assert_array_equal(si.df, array([[0, 50e9, 100e9], [-50e9, 0, 50e9], [-100e9, -50e9, 0]]))
    assert_array_equal(si.tx_osnr, array([40.0, 40.0, 40.0]))

    with pytest.raises(SpectrumError, match='Spectra cannot be summed: channels overlapping.'):
        si += si

    si = create_arbitrary_spectral_information(frequency=array([193.35e12, 193.3e12, 193.25e12]),
                                               slot_width=array([50e9, 50e9, 50e9]),
                                               baud_rate=32e9, signal=array([1, 2, 3]),
                                               tx_osnr=40.0,
                                               ref_power=Pref(1, 1, None))

    assert_array_equal(si.signal, array([3, 2, 1]))

    with pytest.raises(SpectrumError, match='Spectrum baud rate, including the roll off, '
                                            r'larger than the slot width for channels: \[1, 3\].'):
        create_arbitrary_spectral_information(frequency=[193.25e12, 193.3e12, 193.35e12], signal=1,
                                              baud_rate=[64e9, 32e9, 64e9], slot_width=50e9,
                                              tx_osnr=40.0,
                                              ref_power=Pref(1, 1, None))
    with pytest.raises(SpectrumError, match='Spectrum required slot widths larger than the frequency spectral '
                                            r'distances between channels: \[\(1, 2\), \(3, 4\)\].'):
        create_arbitrary_spectral_information(frequency=[193.26e12, 193.3e12, 193.35e12, 193.39e12], signal=1,
                                              tx_osnr=40.0, baud_rate=32e9, slot_width=50e9, ref_power=Pref(1, 1, None))
    with pytest.raises(SpectrumError, match='Spectrum required slot widths larger than the frequency spectral '
                                            r'distances between channels: \[\(1, 2\), \(2, 3\)\].'):
        create_arbitrary_spectral_information(frequency=[193.25e12, 193.3e12, 193.35e12], signal=1, baud_rate=49e9,
                                              tx_osnr=40.0, roll_off=0.1, ref_power=Pref(1, 1, None))
    with pytest.raises(SpectrumError,
                       match='Dimension mismatch in input fields.'):
        create_arbitrary_spectral_information(frequency=[193.25e12, 193.3e12, 193.35e12], signal=[1, 2], baud_rate=49e9,
                                              tx_osnr=40.0, ref_power=Pref(1, 1, None))
