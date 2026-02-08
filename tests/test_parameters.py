#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_parameters
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors


"""
Checks that the class SimParams behaves as a mutable Singleton.
"""

import pytest
from pathlib import Path
from numpy.testing import assert_allclose

from gnpy.core.parameters import SimParams, FiberParams
from gnpy.tools.json_io import load_json, Fiber

TEST_DIR = Path(__file__).parent


@pytest.mark.usefixtures('set_sim_params')
def test_sim_parameters():
    sim_params = {'nli_params': {}, 'raman_params': {}}
    SimParams.set_params(sim_params)
    s1 = SimParams()
    assert s1.nli_params.method == 'gn_model_analytic'
    s2 = SimParams()
    assert not s1.raman_params.flag
    sim_params['raman_params']['flag'] = True
    SimParams.set_params(sim_params)
    assert s2.raman_params.flag
    assert s1.raman_params.flag


def test_fiber_parameters():
    fiber_dict_explicit_g0 = load_json(TEST_DIR/'data'/'test_parameters_fiber_config.json')['params']
    fiber_params_explicit_g0 = FiberParams(**fiber_dict_explicit_g0)

    fiber_dict_default_g0 = load_json(TEST_DIR/'data'/'test_science_utils_fiber_config.json')['params']
    fiber_params_default_g0 = FiberParams(**fiber_dict_default_g0)

    fiber_dict_cr = load_json(TEST_DIR/'data'/'test_old_parameters_fiber_config.json')['params']
    fiber_dict_cr.update(Fiber(**fiber_dict_cr).__dict__)
    fiber_params_cr = FiberParams(**fiber_dict_cr)

    norm_gamma_raman_explicit_g0 = fiber_params_explicit_g0.raman_coefficient.normalized_gamma_raman
    norm_gamma_raman_default_g0 = fiber_params_default_g0.raman_coefficient.normalized_gamma_raman

    norm_gamma_raman_cr = fiber_params_cr.raman_coefficient.normalized_gamma_raman

    assert_allclose(norm_gamma_raman_explicit_g0, norm_gamma_raman_default_g0, rtol=1e-10)
    assert_allclose(norm_gamma_raman_explicit_g0, norm_gamma_raman_cr, rtol=1e-10)

    # Change Effective Area
    fiber_dict_default_g0['effective_area'] = 100e-12
    no_ssmf_fiber_params = FiberParams(**fiber_dict_default_g0)

    norm_gamma_raman_default_g0_no_ssmf = no_ssmf_fiber_params.raman_coefficient.normalized_gamma_raman

    assert_allclose(norm_gamma_raman_explicit_g0, norm_gamma_raman_default_g0_no_ssmf, rtol=1e-10)


def test_fiber_total_loss_scalar():
    """Test that total_loss parameter correctly computes loss_coef from total_loss / length."""
    fiber_dict = {
        'length': 80,
        'length_units': 'km',
        'att_in': 0,
        'con_in': 0.5,
        'con_out': 0.5,
        'dispersion': 1.67e-05,
        'effective_area': 8.3e-11,
        'pmd_coef': 1.265e-15,
        'total_loss': 16.0,  # 16 dB total loss for 80 km → 0.2 dB/km
    }
    params = FiberParams(**fiber_dict)
    # total_loss = 16 dB, length = 80 km → loss_coef should be 0.2 dB/km = 0.0002 dB/m
    assert_allclose(params.loss_coef, 0.0002, rtol=1e-10)
    assert_allclose(params.total_loss, 16.0, rtol=1e-10)


def test_fiber_total_loss_per_frequency():
    """Test that frequency-dependent total_loss correctly computes loss_coef per frequency."""
    fiber_dict = {
        'length': 80,
        'length_units': 'km',
        'att_in': 0,
        'con_in': 0.5,
        'con_out': 0.5,
        'dispersion': 1.67e-05,
        'effective_area': 8.3e-11,
        'pmd_coef': 1.265e-15,
        'total_loss': {
            'value': [16.0, 17.6],  # dB at two frequencies
            'frequency': [191.35e12, 196.1e12],
        },
    }
    params = FiberParams(**fiber_dict)
    # 16.0 / 80 = 0.2 dB/km = 0.0002 dB/m
    # 17.6 / 80 = 0.22 dB/km = 0.00022 dB/m
    assert_allclose(params.loss_coef, [0.0002, 0.00022], rtol=1e-10)
    assert_allclose(params.total_loss, [16.0, 17.6], rtol=1e-10)


def test_fiber_total_loss_backward_compat():
    """Ensure loss_coef still works when total_loss is not provided."""
    fiber_dict = {
        'length': 80,
        'length_units': 'km',
        'att_in': 0,
        'con_in': 0.5,
        'con_out': 0.5,
        'dispersion': 1.67e-05,
        'effective_area': 8.3e-11,
        'pmd_coef': 1.265e-15,
        'loss_coef': 0.2,  # dB/km
    }
    params = FiberParams(**fiber_dict)
    assert_allclose(params.loss_coef, 0.0002, rtol=1e-10)
    assert params.total_loss is None


def test_fiber_total_loss_asdict():
    """Test that asdict preserves total_loss when it was the original input."""
    fiber_dict = {
        'length': 80,
        'length_units': 'km',
        'att_in': 0,
        'con_in': 0.5,
        'con_out': 0.5,
        'dispersion': 1.67e-05,
        'effective_area': 8.3e-11,
        'pmd_coef': 1.265e-15,
        'total_loss': 16.0,
    }
    params = FiberParams(**fiber_dict)
    d = params.asdict()
    assert 'total_loss' in d
    assert 'loss_coef' not in d
    assert_allclose(d['total_loss'], 16.0, rtol=1e-10)
