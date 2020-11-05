#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Alessio Ferrari
"""
Checks that RamanFiber propagates properly the spectral information. In this way, also the RamanSolver and the NliSolver
are tested.
"""

from pathlib import Path
from pandas import read_csv
from numpy.testing import assert_allclose

from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import RamanFiber
from gnpy.core.parameters import SimParams
from gnpy.tools.json_io import load_json

TEST_DIR = Path(__file__).parent


def test_raman_fiber():
    """ Test the accuracy of propagating the RamanFiber."""
    # spectral information generation
    power = 1e-3
    eqpt_params = load_json(TEST_DIR / 'data' / 'eqpt_config.json')
    spectral_info_params = eqpt_params['SI'][0]
    spectral_info_params.pop('power_dbm')
    spectral_info_params.pop('power_range_db')
    spectral_info_params.pop('tx_osnr')
    spectral_info_params.pop('sys_margins')
    spectral_info_input = create_input_spectral_information(power=power, **spectral_info_params)

    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    fiber = RamanFiber(**load_json(TEST_DIR / 'data' / 'raman_fiber_config.json'))

    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = [carrier.power.signal for carrier in spectral_info_out.carriers]
    p_ase = [carrier.power.ase for carrier in spectral_info_out.carriers]
    p_nli = [carrier.power.nli for carrier in spectral_info_out.carriers]

    expected_results = read_csv(TEST_DIR / 'data' / 'test_science_utils_expected_results.csv')
    assert_allclose(p_signal, expected_results['signal'], rtol=1e-3)
    assert_allclose(p_ase, expected_results['ase'], rtol=1e-3)
    assert_allclose(p_nli, expected_results['nli'], rtol=1e-3)
    SimParams.reset()
