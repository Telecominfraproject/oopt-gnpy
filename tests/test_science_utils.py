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

from gnpy.core.info import create_input_spectral_information, create_arbitrary_spectral_information, Pref, \
    ReferenceCarrier
from gnpy.core.elements import Fiber, RamanFiber
from gnpy.core.parameters import SimParams
from gnpy.tools.json_io import load_json
from gnpy.core.exceptions import NetworkTopologyError
from gnpy.core.science_utils import RamanSolver

TEST_DIR = Path(__file__).parent


def test_fiber():
    """Test the accuracy of propagating the Fiber."""
    fiber = Fiber(**load_json(TEST_DIR / 'data' / 'test_science_utils_fiber_config.json'))

    # fix grid spectral information generation
    spectral_info_input = create_input_spectral_information(f_min=191.3e12, f_max=196.1e12, roll_off=0.15,
                                                            baud_rate=32e9, power=1e-3, spacing=50e9, tx_osnr=40.0,
                                                            ref_carrier=ReferenceCarrier(baud_rate=32e9, slot_width=50e9))
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
    delta_pdb_per_channel = [0, 0, 0, 0, 0]
    pref = Pref(p_span0=0, p_spani=0, ref_carrier=None)
    spectral_info_input = create_arbitrary_spectral_information(frequency=frequency, slot_width=slot_width,
                                                                signal=signal, baud_rate=baud_rate, roll_off=0.15,
                                                                delta_pdb_per_channel=delta_pdb_per_channel,
                                                                tx_osnr=40.0, ref_power=pref)

    # propagation
    spectral_info_out = fiber(spectral_info_input)

    p_signal = spectral_info_out.signal
    p_nli = spectral_info_out.nli

    expected_results = read_csv(TEST_DIR / 'data' / 'test_fiber_flex_expected_results.csv')
    assert_allclose(p_signal, expected_results['signal'], rtol=1e-3)
    assert_allclose(p_nli, expected_results['nli'], rtol=1e-3)


@pytest.mark.usefixtures('set_sim_params')
def test_raman_fiber():
    """Test the accuracy of propagating the RamanFiber."""
    # spectral information generation
    spectral_info_input = create_input_spectral_information(f_min=191.3e12, f_max=196.1e12, roll_off=0.15,
                                                            baud_rate=32e9, power=1e-3, spacing=50e9, tx_osnr=40.0,
                                                            ref_carrier=ReferenceCarrier(baud_rate=32e9, slot_width=50e9))
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


@pytest.mark.parametrize(
    "loss, position, errmsg",
    ((0.5, -2, "Lumped loss positions must be between 0 and the fiber length (80.0 km), boundaries excluded."),
     (0.5, 81, "Lumped loss positions must be between 0 and the fiber length (80.0 km), boundaries excluded.")))
@pytest.mark.usefixtures('set_sim_params')
def test_fiber_lumped_losses(loss, position, errmsg, set_sim_params):
    """Lumped losses length sanity checking."""
    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    fiber_dict = load_json(TEST_DIR / 'data' / 'test_lumped_losses_raman_fiber_config.json')
    fiber_dict['params']['lumped_losses'] = [{'position': position, 'loss': loss}]
    with pytest.raises(NetworkTopologyError) as e:
        Fiber(**fiber_dict)
    assert str(e.value) == errmsg


@pytest.mark.usefixtures('set_sim_params')
def test_fiber_lumped_losses_srs(set_sim_params):
    """Test the accuracy of Fiber with lumped losses propagation."""
    # spectral information generation
    spectral_info_input = create_input_spectral_information(f_min=191.3e12, f_max=196.1e12, roll_off=0.15,
                                                            baud_rate=32e9, power=1e-3, spacing=50e9, tx_osnr=40.0,
                                                            ref_carrier=ReferenceCarrier(baud_rate=32e9, slot_width=50e9))

    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    fiber = Fiber(**load_json(TEST_DIR / 'data' / 'test_lumped_losses_raman_fiber_config.json'))
    raman_fiber = RamanFiber(**load_json(TEST_DIR / 'data' / 'test_lumped_losses_raman_fiber_config.json'))

    # propagation
    # without Raman pumps
    stimulated_raman_scattering = RamanSolver.calculate_stimulated_raman_scattering(
        spectral_info_input, fiber)
    power_profile = stimulated_raman_scattering.power_profile
    expected_power_profile = read_csv(TEST_DIR / 'data' / 'test_lumped_losses_fiber_no_pumps.csv', header=None)
    assert_allclose(power_profile, expected_power_profile, rtol=1e-3)

    # with Raman pumps
    expected_power_profile = read_csv(TEST_DIR / 'data' / 'test_lumped_losses_raman_fiber.csv', header=None)
    stimulated_raman_scattering = RamanSolver.calculate_stimulated_raman_scattering(
        spectral_info_input, raman_fiber)
    power_profile = stimulated_raman_scattering.power_profile
    assert_allclose(power_profile, expected_power_profile, rtol=1e-3)

    # without Stimulated Raman Scattering
    expected_power_profile = read_csv(TEST_DIR / 'data' / 'test_lumped_losses_fiber_no_raman.csv', header=None)
    stimulated_raman_scattering = RamanSolver.calculate_attenuation_profile(spectral_info_input, fiber)
    power_profile = stimulated_raman_scattering.power_profile
    assert_allclose(power_profile, expected_power_profile, rtol=1e-3)
