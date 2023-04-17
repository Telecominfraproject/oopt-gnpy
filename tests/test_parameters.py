#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checks that the class SimParams behaves as a mutable Singleton.
"""

import pytest
from gnpy.core.parameters import SimParams


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
