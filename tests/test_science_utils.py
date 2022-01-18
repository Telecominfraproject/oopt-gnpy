#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checks that RamanFiber propagates properly the spectral information. In this way, also the RamanSolver and the NliSolver
are tested.
"""

from pathlib import Path
from pandas import read_csv
from numpy.testing import assert_allclose
from numpy import array
import pytest

from gnpy.core.info import create_input_spectral_information, create_arbitrary_spectral_information
from gnpy.core.elements import Fiber, RamanFiber
from gnpy.core.parameters import SimParams
from gnpy.tools.json_io import load_json

TEST_DIR = Path(__file__).parent


def test_fiber():
    """ Test the accuracy of propagating the Fiber."""
    fiber = Fiber(**load_json(TEST_DIR / 'data' / 'test_science_utils_fiber_config.json'))

    # fix grid spectral information generation
    spectral_info_input = create_input_spectral_information(f_min=191.3e12, f_max=196.1e12, roll_off=0.15,
                                                            baud_rate=32e9, power=1e-3, spacing=50e9)
    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = spectral_info_out.signal
    p_nli = spectral_info_out.nli

    expected_results = read_csv(TEST_DIR / 'data' / 'test_fiber_fix_expected_results.csv')
    assert_allclose(p_signal, expected_results['signal'], rtol=1e-3)
    assert_allclose(p_nli, expected_results['nli'], rtol=1e-3)

    # flex grid spectral information generation
    frequency = 191e12 + array([0, 50e9, 150e9, 225e9, 275e9])
    slot_width = array([37.5e9, 50e9, 75e9, 50e9, 37.5e9])
    baud_rate = array([32e9, 42e9, 64e9, 42e9, 32e9])
    signal = 1e-3 + array([0, -1e-4, 3e-4, -2e-4, +2e-4])
    spectral_info_input = create_arbitrary_spectral_information(frequency=frequency, slot_width=slot_width,
                                                                signal=signal, baud_rate=baud_rate, roll_off=0.15)

    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = spectral_info_out.signal
    p_nli = spectral_info_out.nli

    expected_results = read_csv(TEST_DIR / 'data' / 'test_fiber_flex_expected_results.csv')
    assert_allclose(p_signal, expected_results['signal'], rtol=1e-3)
    assert_allclose(p_nli, expected_results['nli'], rtol=1e-3)


@pytest.mark.usefixtures('set_sim_params')
def test_raman_fiber():
    """ Test the accuracy of propagating the RamanFiber."""
    # spectral information generation
    spectral_info_input = create_input_spectral_information(f_min=191.3e12, f_max=196.1e12, roll_off=0.15,
                                                            baud_rate=32e9, power=1e-3, spacing=50e9)
    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    fiber = RamanFiber(**load_json(TEST_DIR / 'data' / 'test_science_utils_fiber_config.json'))

    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = spectral_info_out.signal
    p_ase = spectral_info_out.ase
    p_nli = spectral_info_out.nli

    expected_results = read_csv(TEST_DIR / 'data' / 'test_raman_fiber_expected_results.csv')
    assert_allclose(p_signal, expected_results['signal'], rtol=1e-3)
    assert_allclose(p_ase, expected_results['ase'], rtol=1e-3)
    assert_allclose(p_nli, expected_results['nli'], rtol=1e-3)
