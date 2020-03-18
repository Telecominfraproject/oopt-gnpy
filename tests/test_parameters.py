#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, pytest
from pathlib import Path

from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import Simulation
from gnpy.core.elements import Fiber

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'

def test_sim_parameters():
    f = open(DATA_DIR / 'test_sim_params.json')
    j = json.load(f)
    sim_params = SimParams(**j)
    Simulation.set_params(sim_params)
    s1 = Simulation.get_simulation()
    assert s1.sim_params.raman_params.flag_raman
    s2 = Simulation.get_simulation()
    assert s2.sim_params.raman_params.flag_raman
    j['raman_parameters']['flag_raman'] = False
    sim_params = SimParams(**j)
    Simulation.set_params(sim_params)
    assert not s2.sim_params.raman_params.flag_raman
    assert not s1.sim_params.raman_params.flag_raman

if __name__ == '__main__':
    from logging import getLogger, basicConfig, INFO
    logger = getLogger(__name__)
    basicConfig(level=INFO)

    test_sim_parameters()

    print('\n')




