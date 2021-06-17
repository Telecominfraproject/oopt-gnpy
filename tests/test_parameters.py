#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checks that the class SimParams behaves as a mutable Singleton.
"""

import pytest
from gnpy.core.parameters import SimParams, NLIParams, RamanParams


class MockSimParams(SimParams):
    """Mock simulation parameters for monkey patch"""
    _shared_dict = {'nli_params': NLIParams(), 'raman_params': RamanParams()}


@pytest.fixture
def set_sim_params(monkeypatch):
    monkeypatch.setattr(SimParams, '_shared_dict', MockSimParams._shared_dict)


def test_sim_parameters(set_sim_params):
    sim_params = {'nli_params': {}, 'raman_params': {}}
    MockSimParams.set_params(sim_params)
    s1 = SimParams.get()
    assert s1.nli_params.method == 'gn_model_analytic'
    s2 = SimParams.get()
    assert not s1.raman_params.flag
    sim_params['raman_params']['flag'] = True
    MockSimParams.set_params(sim_params)
    assert s2.raman_params.flag
    assert s1.raman_params.flag
