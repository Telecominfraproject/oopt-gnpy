#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import Simulation
from gnpy.tools.json_io import load_json

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'


def test_sim_parameters():
    j = load_json(DATA_DIR / 'test_sim_params.json')
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
