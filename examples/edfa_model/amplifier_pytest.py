#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Jean-Luc Auge
# @Date:   2018-02-02 14:06:55

from amplifier21 import *
from numpy import polyval
import pytest

@pytest.fixture
def nf_fitco_values():
    return [1e-04, 4e-02, 3e-02, 6]

@pytest.fixture
def amp_parameters1():
    """generates parameters to test the nf model"""
    return ('true', 20, 30, 5.5, 8)

@pytest.fixture
def amp_parameters2(nf_fitco_values):
    """generates parameters for both fitco and nf models
    the nf model nf_min and nf_max parameters are claculated with the nf_fitco_values
    this way a comparison of nf_fitco vs nf_model is possible"""
    gain_flat = 25
    gain_min = 15
    #nf_min:
    gain = gain_flat
    voa = gain_flat - gain
    nf_min = get_nf(gain, voa, 'false', gain_min, gain_flat, 0, 0, nf_fitco_values)
    #nf_max:
    gain = gain_min
    voa = gain_flat - gain
    nf_max = get_nf(gain, voa, 'false', gain_min, gain_flat, 0, 0, nf_fitco_values)
    return (gain_min, gain_flat, nf_min, nf_max, nf_fitco_values)

@pytest.mark.parametrize("gain", [13, 15, 17, 19, 21, 23, 25, 27])
def test_nf_fitco(gain, amp_parameters2):   
    (gain_min, gain_flat, nf_min, nf_max, nf_fitco_values) = amp_parameters2
    voa = gain_flat - gain
    pad = 0
    if gain < gain_min:
        pad = gain_min - gain
    nf_expected = polyval(nf_fitco_values, -(voa-pad)) + pad
    dif = abs(get_nf(gain, voa, 'false', gain_min, gain_flat, 0, 0, nf_fitco_values) - nf_expected)
    assert dif < 0.01

@pytest.mark.parametrize("gain, nf_expected",[(20, 8), (30, 5.5)])
def test_nf_model(gain, nf_expected, amp_parameters1):
    (nf_model_enabled, gain_min, gain_flat, nf_min, nf_max) = amp_parameters1
    voa = gain_flat - gain
    nf_fitco = [0, 0, 0, 0]
    dif = abs(get_nf(gain, voa, nf_model_enabled, gain_min, gain_flat, nf_min, nf_max, nf_fitco) - nf_expected)
    assert dif < 0.01

@pytest.mark.parametrize("gain", [15, 25])
def test_compare_nf_models1(gain, amp_parameters2):
    (gain_min, gain_flat, nf_min, nf_max, nf_fitco_values) = amp_parameters2
    voa = gain_flat - gain
    nf_fitco = get_nf(gain, voa, 'false', gain_min, gain_flat, 0, 0, nf_fitco_values)
    nf_model = get_nf(gain, voa, 'true', gain_min, gain_flat, nf_min, nf_max, [0, 0, 0, 0])
    dif = abs(nf_model - nf_fitco)
    assert dif < 0.01

@pytest.mark.parametrize("gain", [17, 19, 21, 23])
def test_compare_nf_models2(gain, amp_parameters2):
    (gain_min, gain_flat, nf_min, nf_max, nf_fitco_values) = amp_parameters2
    voa = gain_flat - gain
    nf_fitco = get_nf(gain, voa, 'false', gain_min, gain_flat, 0, 0, nf_fitco_values)
    nf_model = get_nf(gain, voa, 'true', gain_min, gain_flat, nf_min, nf_max, [0, 0, 0, 0])
    dif = abs(nf_model - nf_fitco)
    assert dif < 0.5



