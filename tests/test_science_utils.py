#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Alessio Ferrari
"""
Checks that RamanFiber propagates properly the spectral information. In this way, also the RamanSolver and the NliSolver
are tested.
"""

import pytest
from pathlib import Path
from pandas import read_csv
from numpy.testing import assert_allclose
from numpy import load

from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, RamanFiber
from gnpy.core.parameters import SimParams
from gnpy.tools.json_io import load_json
from gnpy.core.exceptions import NetworkTopologyError
from gnpy.core.science_utils import RamanSolver

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


@pytest.mark.parametrize("loss, position, errmsg", (
        (0.5, -2, f"Lumped loss positions must be between 0 and the fiber length (80.0 km), boundaries excluded."),
        (0.5, 81, f"Lumped loss positions must be between 0 and the fiber length (80.0 km), boundaries excluded.")))
def test_fiber_lumped_losses(loss, position, errmsg):
    """ Check on Fiber with lumped losses."""
    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    fiber_dict = load_json(TEST_DIR / 'data' / 'test_lumped_losses_fiber_config.json')
    fiber_dict['params']['lumped_losses'] = [{'position': position, 'loss': loss}]
    with pytest.raises(NetworkTopologyError) as e:
        Fiber(**fiber_dict)
    assert str(e.value) == errmsg
    SimParams.reset()


def test_fiber_lumped_losses_srs():
    """ Test the accuracy of Fiber with lumped losses propagation."""
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
    sim_params = SimParams.get()
    fiber = Fiber(**load_json(TEST_DIR / 'data' / 'test_lumped_losses_fiber_config.json'))
    raman_fiber = RamanFiber(**load_json(TEST_DIR / 'data' / 'test_lumped_losses_raman_fiber_config.json'))

    # propagation
    # without Raman pumps
    expected_power_profile = load(TEST_DIR / 'data' / 'test_lumped_losses_fiber_no_pumps.npy')
    stimulated_raman_scattering = RamanSolver.calculate_stimulated_raman_scattering(
        spectral_info_input, fiber, sim_params)
    power_profile = stimulated_raman_scattering.power_profile
    assert_allclose(power_profile, expected_power_profile, rtol=1e-3)

    # with Raman pumps
    expected_power_profile = load(TEST_DIR / 'data' / 'test_lumped_losses_raman_fiber.npy')
    stimulated_raman_scattering = RamanSolver.calculate_stimulated_raman_scattering(
        spectral_info_input, raman_fiber, sim_params)
    power_profile = stimulated_raman_scattering.power_profile
    assert_allclose(power_profile, expected_power_profile, rtol=1e-3)

    # without Stimulated Raman Scattering
    expected_power_profile = load(TEST_DIR / 'data' / 'test_lumped_losses_fiber_no_raman.npy')
    stimulated_raman_scattering = RamanSolver.calculate_attenuation_profile(
        spectral_info_input, fiber, sim_params)
    power_profile = stimulated_raman_scattering.power_profile
    assert_allclose(power_profile, expected_power_profile, rtol=1e-3)
    SimParams.reset()
