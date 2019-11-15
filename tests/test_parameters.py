#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest, json

from gnpy.core.parameters import Parameters, SimParams, FiberParams
from gnpy.core.elements import Fiber

class test_sim_parameters():
    f = open('data/test_sim_params.json')
    j = json.load(f)
    s1 = SimParams(j)
    assert s1.raman_params.flag_raman
    s2 = SimParams(j)
    assert s2.raman_params.flag_raman
    j['raman_parameters']['flag_raman'] = False
    s3 = SimParams(j)
    assert not s2.raman_params.flag_raman
    assert not s1.raman_params.flag_raman
    print(s1.__dict__)

class test_asdict():
    f = open('data/test_network.json')
    j = json.load(f)
    j = [e for e in j['elements'] if e['type'] == 'Fiber'][0]
    params = {'length': 80, 'loss_coef': 0.2, 'length_units': 'km', 'att_in': 0, 'con_in': 0.5, 'con_out': 0.5,
              'type_variety': 'SSMF', 'dispersion': 1.67e-05, 'gamma': 0.00127, 'ref_wavelength': 1550e-9, 'beta3': 0}
    metadata = j.pop('metadata')
    fiber = Fiber(uid='1', metadata=metadata, params=params)
    is_equal = True
    new_params = fiber.params.asdict()
    for key in new_params:
        if is_equal:
            is_equal = new_params[key] == params[key]
        else:
            break
    assert is_equal
    print(fiber.params.length)




