#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    raman_coefficient_explicit_g0 = fiber_params_explicit_g0.raman_coefficient
    raman_coefficient_explicit_g0 =\
        raman_coefficient_explicit_g0.normalized_gamma_raman * fiber_params_explicit_g0._raman_reference_frequency

    raman_coefficient_default_g0 = fiber_params_default_g0.raman_coefficient
    raman_coefficient_default_g0 = \
        raman_coefficient_default_g0.normalized_gamma_raman * fiber_params_default_g0._raman_reference_frequency

    raman_coefficient_cr = fiber_params_cr.raman_coefficient
    raman_coefficient_cr = \
        raman_coefficient_cr.normalized_gamma_raman * fiber_params_cr._raman_reference_frequency

    assert_allclose(raman_coefficient_explicit_g0, raman_coefficient_default_g0, rtol=1e-10)
    assert_allclose(raman_coefficient_explicit_g0, raman_coefficient_cr, rtol=1e-10)
