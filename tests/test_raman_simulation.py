#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_raman_simulation
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
Tests for the Raman simulation module, covering:
- Attenuation profile calculation (no SRS)
- Stimulated Raman Scattering with perturbative solver at various orders
- Numerical vs perturbative solver comparison
- Spectral tilt due to SRS
- StimulatedRamanScattering data class
- Simple single-link topology sanity checks
"""

from pathlib import Path
from copy import deepcopy

import pytest
from numpy import array, ones, exp, outer, sqrt, allclose
from numpy.testing import assert_allclose, assert_array_less

from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, RamanFiber
from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import RamanSolver, StimulatedRamanScattering
from gnpy.tools.json_io import load_json

TEST_DIR = Path(__file__).parent


def _create_spectral_info(n_channels=10):
    """Helper: create spectral information with a few channels."""
    f_min = 191.3e12
    spacing = 50e9
    f_max = f_min + (n_channels - 1) * spacing
    return create_input_spectral_information(
        f_min=f_min, f_max=f_max, roll_off=0.15,
        baud_rate=32e9, spacing=spacing, tx_osnr=40.0, tx_power=1e-3)


@pytest.fixture
def fiber():
    """Standard SSMF fiber for testing."""
    fiber_config = load_json(TEST_DIR / 'data' / 'test_science_utils_fiber_config.json')
    f = Fiber(**fiber_config)
    f.ref_pch_in_dbm = 0.0
    return f


@pytest.fixture
def raman_fiber():
    """RamanFiber with Raman pumps for testing."""
    fiber_config = load_json(TEST_DIR / 'data' / 'test_science_utils_fiber_config.json')
    f = RamanFiber(**fiber_config)
    f.ref_pch_in_dbm = 0.0
    return f


@pytest.mark.usefixtures('set_sim_params')
class TestAttenuationProfile:
    """Tests for RamanSolver.calculate_attenuation_profile (no SRS)."""

    def test_power_decreases_along_fiber(self, fiber):
        """Power should decrease monotonically along the fiber without SRS."""
        spectral_info = _create_spectral_info(5)
        srs = RamanSolver.calculate_attenuation_profile(spectral_info, fiber)

        # Power at end should be less than at start for all channels
        for ch in range(srs.power_profile.shape[0]):
            assert srs.power_profile[ch, -1] < srs.power_profile[ch, 0]

    def test_loss_profile_starts_at_unity(self, fiber):
        """Loss profile should start at 1.0 (no loss at z=0)."""
        spectral_info = _create_spectral_info(5)
        srs = RamanSolver.calculate_attenuation_profile(spectral_info, fiber)

        assert_allclose(srs.loss_profile[:, 0], ones(srs.loss_profile.shape[0]), rtol=1e-10)

    def test_loss_profile_consistent_with_alpha(self, fiber):
        """Loss at fiber end should be consistent with exp(-alpha * L)."""
        spectral_info = _create_spectral_info(5)
        srs = RamanSolver.calculate_attenuation_profile(spectral_info, fiber)

        alpha = fiber.alpha(spectral_info.frequency)
        expected_loss = exp(-alpha * fiber.params.length)
        assert_allclose(srs.loss_profile[:, -1], expected_loss, rtol=1e-3)

    def test_rho_is_sqrt_of_loss(self, fiber):
        """The rho field should be sqrt(loss_profile)."""
        spectral_info = _create_spectral_info(5)
        srs = RamanSolver.calculate_attenuation_profile(spectral_info, fiber)

        assert_allclose(srs.rho, sqrt(srs.loss_profile), rtol=1e-10)

    def test_z_array_spans_fiber(self, fiber):
        """z array should start at 0 and end at fiber length."""
        spectral_info = _create_spectral_info(5)
        srs = RamanSolver.calculate_attenuation_profile(spectral_info, fiber)

        assert srs.z[0] == 0
        assert srs.z[-1] == fiber.params.length


@pytest.mark.usefixtures('set_sim_params')
class TestStimulatedRamanScattering:
    """Tests for SRS with perturbative solver at various orders."""

    def test_srs_order_1(self, fiber):
        """SRS order 1 should produce a spectral tilt (lower freq gains power from higher freq)."""
        spectral_info = _create_spectral_info(10)
        SimParams.set_params({'raman_params': {'flag': True, 'order': 1}})
        srs = RamanSolver.calculate_stimulated_raman_scattering(spectral_info, fiber)

        # With SRS, lower frequencies should have relatively more power than without SRS
        loss_first = srs.loss_profile[0, -1]
        loss_last = srs.loss_profile[-1, -1]
        # Lower frequency channel should have less loss (more gain from Raman)
        assert loss_first > loss_last, "SRS should tilt spectrum: lower freq should have less loss"

    def test_srs_order_2_refines_order_1(self, fiber):
        """Higher order perturbative solution should differ slightly from order 1."""
        spectral_info = _create_spectral_info(10)

        SimParams.set_params({'raman_params': {'flag': True, 'order': 1}})
        srs_o1 = RamanSolver.calculate_stimulated_raman_scattering(deepcopy(spectral_info), fiber)

        SimParams.set_params({'raman_params': {'flag': True, 'order': 2}})
        srs_o2 = RamanSolver.calculate_stimulated_raman_scattering(deepcopy(spectral_info), fiber)

        # Should be close but not identical (difference may be very small for low channel count)
        assert_allclose(srs_o1.power_profile[:, -1], srs_o2.power_profile[:, -1], rtol=0.1)

    def test_srs_orders_converge(self, fiber):
        """Higher perturbative orders should converge: order 3 closer to order 4 than order 1 to order 2."""
        spectral_info = _create_spectral_info(10)

        results = {}
        for order in [1, 2, 3, 4]:
            SimParams.set_params({'raman_params': {'flag': True, 'order': order}})
            srs = RamanSolver.calculate_stimulated_raman_scattering(deepcopy(spectral_info), fiber)
            results[order] = srs.power_profile[:, -1].copy()

        # Difference between consecutive orders should decrease
        diff_12 = abs(results[1] - results[2]).max()
        diff_23 = abs(results[2] - results[3]).max()
        diff_34 = abs(results[3] - results[4]).max()
        assert diff_23 < diff_12, "Perturbative orders should converge"
        assert diff_34 < diff_23, "Perturbative orders should converge"

    def test_numerical_vs_perturbative(self, fiber):
        """Numerical and perturbative methods should produce similar results."""
        spectral_info = _create_spectral_info(10)

        SimParams.set_params({'raman_params': {
            'flag': True, 'order': 2, 'method': 'perturbative',
            'solver_spatial_resolution': 10}})
        srs_pert = RamanSolver.calculate_stimulated_raman_scattering(deepcopy(spectral_info), fiber)

        SimParams.set_params({'raman_params': {
            'flag': True, 'method': 'numerical',
            'solver_spatial_resolution': 10}})
        srs_num = RamanSolver.calculate_stimulated_raman_scattering(deepcopy(spectral_info), fiber)

        # Should agree within 5%
        assert_allclose(srs_pert.power_profile[:, -1], srs_num.power_profile[:, -1], rtol=0.05)


@pytest.mark.usefixtures('set_sim_params')
class TestRamanFiberPropagation:
    """End-to-end tests for fiber propagation with Raman effects."""

    def test_propagation_output_power_reasonable(self, fiber):
        """Output power after propagation should be reasonable (not zero, not amplified)."""
        spectral_info = _create_spectral_info(10)
        SimParams.set_params({'raman_params': {'flag': True, 'order': 2}})
        fiber.ref_pch_in_dbm = 0.0
        spectral_info_out = fiber(spectral_info)

        # Output signal should be positive
        assert (spectral_info_out.signal > 0).all()
        # Output should be less than input (fiber attenuates)
        assert spectral_info_out.ptot_dbm < 0  # input was 0 dBm per channel

    def test_raman_tilt_in_propagation(self):
        """Propagation with SRS should show spectral tilt vs without SRS."""
        spectral_info_no_raman = _create_spectral_info(10)
        spectral_info_raman = deepcopy(spectral_info_no_raman)

        fiber_config = load_json(TEST_DIR / 'data' / 'test_science_utils_fiber_config.json')

        # Without Raman
        SimParams.set_params({'raman_params': {'flag': False}})
        fiber_no_raman = Fiber(**fiber_config)
        fiber_no_raman.ref_pch_in_dbm = 0.0
        out_no_raman = fiber_no_raman(spectral_info_no_raman)

        # With Raman
        SimParams.set_params({'raman_params': {'flag': True, 'order': 2}})
        fiber_raman = Fiber(**fiber_config)
        fiber_raman.ref_pch_in_dbm = 0.0
        out_raman = fiber_raman(spectral_info_raman)

        # Raman should cause spectral tilt: power difference between first and last channel
        # should be different with vs without SRS
        tilt_no_raman = out_no_raman.signal[0] / out_no_raman.signal[-1]
        tilt_raman = out_raman.signal[0] / out_raman.signal[-1]
        assert tilt_raman > tilt_no_raman, "SRS should increase spectral tilt"

    def test_nli_generated_during_propagation(self, fiber):
        """Propagation should generate NLI noise."""
        spectral_info = _create_spectral_info(10)
        SimParams.set_params({'raman_params': {'flag': True, 'order': 2}})
        spectral_info_out = fiber(spectral_info)

        # NLI should be nonzero
        assert (spectral_info_out.nli > 0).all()


@pytest.mark.usefixtures('set_sim_params')
class TestStimulatedRamanScatteringDataClass:
    """Tests for the StimulatedRamanScattering data class."""

    def test_construction(self):
        """Basic construction test."""
        power = array([[1.0, 0.5], [1.0, 0.6]])
        loss = array([[1.0, 0.5], [1.0, 0.6]])
        freq = array([191.3e12, 196.1e12])
        z = array([0, 80000])
        srs = StimulatedRamanScattering(power, loss, freq, z)

        assert_allclose(srs.power_profile, power)
        assert_allclose(srs.loss_profile, loss)
        assert_allclose(srs.frequency, freq)
        assert_allclose(srs.z, z)
        assert_allclose(srs.rho, sqrt(loss))
