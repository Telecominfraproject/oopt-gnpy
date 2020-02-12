#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Alessio Ferrari
"""
checks that RamanFiber propagates properly the spectral information. In this way, also the RamanSolver and the NliSolver
 are tested.
"""

import json
from pandas import read_csv
from numpy.testing import assert_allclose
from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import RamanFiber
from gnpy.core.network import load_sim_params
from pathlib import Path
TEST_DIR = Path(__file__).parent

def test_raman_fiber():
    """ Test the accuracy of propagating the RamanFiber.
    """
    # spectral information generation
    power = 1e-3
    with open(TEST_DIR / 'data' / 'eqpt_config.json', 'r') as file:
        eqpt_params = json.load(file)
    spectral_info_params = eqpt_params['SI'][0]
    spectral_info_params.pop('power_dbm')
    spectral_info_params.pop('power_range_db')
    spectral_info_params.pop('tx_osnr')
    spectral_info_params.pop('sys_margins')
    spectral_info_input = create_input_spectral_information(power=power, **spectral_info_params)

    # RamanFiber
    with open(TEST_DIR / 'data' / 'raman_fiber_config.json', 'r') as file:
        raman_fiber_params = json.load(file)
    sim_params = load_sim_params(TEST_DIR / 'data' / 'sim_params.json')
    fiber = RamanFiber(**raman_fiber_params)
    fiber.sim_params = sim_params

    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = [carrier.power.signal for carrier in spectral_info_out.carriers]
    p_ase = [carrier.power.ase for carrier in spectral_info_out.carriers]
    p_nli = [carrier.power.nli for carrier in spectral_info_out.carriers]

    expected_results = read_csv(TEST_DIR / 'data' / 'expected_results_science_utils.csv')
    assert_allclose(p_signal, expected_results['signal'], rtol=1e-3)
    assert_allclose(p_ase, expected_results['ase'], rtol=1e-3)
    assert_allclose(p_nli, expected_results['nli'], rtol=1e-3)
