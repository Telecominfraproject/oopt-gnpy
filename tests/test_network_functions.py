# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from pathlib import Path
import pytest
from numpy.testing import assert_allclose

from gnpy.core.exceptions import NetworkTopologyError, ConfigurationError
from gnpy.core.network import span_loss, build_network, select_edfa, get_node_restrictions, \
    estimate_srs_power_deviation, add_missing_elements_in_network, get_next_node
from gnpy.tools.json_io import load_equipment, load_network, network_from_json, load_json
from gnpy.core.utils import lin2db, automatic_nch, merge_amplifier_restrictions
from gnpy.core.elements import Fiber, Edfa, Roadm, Multiband_amplifier
from gnpy.core.parameters import SimParams, EdfaParams, MultiBandParams


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
EQPT_MULTBAND_FILENAME = DATA_DIR / 'eqpt_config_multiband.json'
NETWORK_FILENAME = DATA_DIR / 'bugfixiteratortopo.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": DATA_DIR / "std_medium_gain_advanced_config.json",
                 "Juniper-BoosterHG.json": DATA_DIR / "Juniper-BoosterHG.json"}


@pytest.mark.parametrize("node, attenuation", [
    # first fiber span
    ['fiber1', 10.5],
    ['fiber2', 10.5],
    ['fused1', 10.5],
    # second span
    ['fiber3', 16.0],
    # third span
    ['fiber4', 16.0],
    # direct link between a ROADM and an amplifier
    ['fused5', 0],
    # fourth span
    ['fiber6', 17],
    ['fused7', 17],
    # fifth span
    ['fiber7', 0.2],
    ['fiber8', 12],
    # all other nodes
    ['Site_A', 0],
    ['nodeA', 0],
    ['amp2', 0],
    ['nodeC', 0],
    ['Site_C', 0],
    ['amp3', 0],
    ['amp4', 0],
    ['nodeB', 0],
    ['Site_B', 0],
])
def test_span_loss(node, attenuation):
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = load_network(NETWORK_FILENAME, equipment)
    for x in network.nodes():
        if x.uid == node:
            assert attenuation == span_loss(network, x, equipment)
            return
    assert not f'node "{node}" referenced from test but not found in the topology'  # pragma: no cover


@pytest.mark.parametrize("node", ['fused4'])
def test_span_loss_unconnected(node):
    '''Fused node that has no next and no previous nodes should be detected'''
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = load_network(NETWORK_FILENAME, equipment)
    x = next(x for x in network.nodes() if x.uid == node)
    with pytest.raises(NetworkTopologyError):
        span_loss(network, x, equipment)


@pytest.mark.parametrize('typ, expected_loss',
                         [('Edfa', [11, 11]),
                          ('Fused', [11, 10])])
def test_eol(typ, expected_loss):
    """Check that EOL is added only once on spans. One span can be one fiber or several fused fibers
    EOL is then added on the first fiber only.
    """
    json_data = {
        "elements": [
            {
                "uid": "trx SITE1",
                "type": "Transceiver"
            },
            {
                "uid": "trx SITE2",
                "type": "Transceiver"
            },
            {
                "uid": "roadm SITE1",
                "type": "Roadm"
            },
            {
                "uid": "roadm SITE2",
                "type": "Roadm"
            },
            {
                "uid": "fiber (SITE1 → ILA1)",
                "type": "Fiber",
                "type_variety": "SSMF",
                "params": {
                    "length": 50.0,
                    "loss_coef": 0.2,
                    "length_units": "km"
                }
            },
            {
                "uid": "fiber (ILA1 → SITE2)",
                "type": "Fiber",
                "type_variety": "SSMF",
                "params": {
                    "length": 50.0,
                    "loss_coef": 0.2,
                    "length_units": "km"
                }
            },
            {
                "uid": "east edfa in SITE1 to ILA1",
                "type": "Edfa"
            },
            {
                "uid": "west edfa in SITE2 to ILA1",
                "type": typ
            },
            {
                "uid": "east edfa in ILA1 to SITE2",
                "type": "Edfa"
            }
        ],
        "connections": [
            {
                "from_node": "trx SITE1",
                "to_node": "roadm SITE1"
            },
            {
                "from_node": "roadm SITE1",
                "to_node": "east edfa in SITE1 to ILA1"
            },
            {
                "from_node": "east edfa in SITE1 to ILA1",
                "to_node": "fiber (SITE1 → ILA1)"
            },
            {
                "from_node": "fiber (SITE1 → ILA1)",
                "to_node": "east edfa in ILA1 to SITE2"
            },
            {
                "from_node": "east edfa in ILA1 to SITE2",
                "to_node": "fiber (ILA1 → SITE2)"
            },
            {
                "from_node": "fiber (ILA1 → SITE2)",
                "to_node": "west edfa in SITE2 to ILA1"
            },
            {
                "from_node": "west edfa in SITE2 to ILA1",
                "to_node": "roadm SITE2"
            },
            {
                "from_node": "roadm SITE2",
                "to_node": "trx SITE2"
            }
        ]
    }
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    equipment['Span']['default'].EOL = 1
    network = network_from_json(json_data, equipment)
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))

    build_network(network, equipment, p_db, p_total_db)
    fibers = [f for f in network.nodes() if isinstance(f, Fiber)]
    for i in range(2):
        assert fibers[i].loss == expected_loss[i]


@pytest.mark.parametrize('p_db, power_mode, elem1, elem2, expected_gain, expected_delta_p, expected_voa', [
    (-17, True, 'edfa', 'fiber', 15.0, 15, 15.0),
    (-17, True, 'fiber', 'edfa', 15.0, 5.0, 5.0),
    (-17, False, 'edfa', 'fiber', 0.0, None, 0.0),
    (-17, False, 'fiber', 'edfa', 10.0, None, 0.0),
    (10, True, 'edfa', 'fiber', -9.0, -9.0, 0.0),
    (10, True, 'fiber', 'edfa', 1.0, -9.0, 0.0),
    (10, False, 'edfa', 'fiber', -9.0, None, 0.0),
    (10, False, 'fiber', 'edfa', 1.0, None, 0.0)])
def test_design_non_amplified_link(elem1, elem2, expected_gain, expected_delta_p, expected_voa, power_mode, p_db):
    """Check that the delta_p, gain computed on an amplified link that starts from a transceiver are correct
    """
    json_data = {
        "elements": [
            {
                "uid": "trx SITE1",
                "type": "Transceiver"
            },
            {
                "uid": "trx SITE2",
                "type": "Transceiver"
            },
            {
                "uid": "edfa",
                "type": "Edfa",
                "type_variety": "std_low_gain"
            },
            {
                "uid": "fiber",
                "type": "Fiber",
                "type_variety": "SSMF",
                "params": {
                    "length": 50.0,
                    "loss_coef": 0.2,
                    "length_units": "km"
                }
            }
        ],
        "connections": [
            {
                "from_node": "trx SITE1",
                "to_node": elem1
            },
            {
                "from_node": elem1,
                "to_node": elem2
            },
            {
                "from_node": elem2,
                "to_node": "trx SITE2"
            }
        ]
    }
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    equipment['Span']['default'].power_mode = power_mode
    equipment['SI']['default'].power_dbm = p_db
    equipment['SI']['default'].tx_power_dbm = p_db
    network = network_from_json(json_data, equipment)
    edfa = next(a for a in network.nodes() if a.uid == 'edfa')
    edfa.params.out_voa_auto = True
    p_total_db = p_db + 20.0

    build_network(network, equipment, p_db, p_total_db)
    amps = [a for a in network.nodes() if isinstance(a, Edfa)]
    for amp in amps:
        assert amp.out_voa == expected_voa
        assert amp.delta_p == expected_delta_p
        # max power of std_low_gain is 21 dBm
        assert amp.effective_gain == expected_gain


def network_base(case, site_type, length=50.0, amplifier_type='Multiband_amplifier'):
    base_network = {
        'elements': [
            {
                'uid': 'trx SITE1',
                'type': 'Transceiver'
            },
            {
                'uid': 'trx SITE2',
                'type': 'Transceiver'
            },
            {
                'uid': 'roadm SITE1',
                'type': 'Roadm'
            },
            {
                'uid': 'roadm SITE2',
                'type': 'Roadm'
            },
            {
                'uid': 'fiber (SITE1 → ILA1)',
                'type': 'Fiber',
                'type_variety': 'SSMF',
                'params': {
                    'length': length,
                    'loss_coef': 0.2,
                    'length_units': 'km'
                }
            },
            {
                'uid': 'fiber (ILA1 → ILA2)',
                'type': 'Fiber',
                'type_variety': 'SSMF',
                'params': {
                    'length': 50.0,
                    'loss_coef': 0.2,
                    'length_units': 'km'
                }
            },
            {
                'uid': 'fiber (ILA2 → SITE2)',
                'type': 'Fiber',
                'type_variety': 'SSMF',
                'params': {
                    'length': 50.0,
                    'loss_coef': 0.2,
                    'length_units': 'km'
                }
            },
            {
                'uid': 'east edfa in SITE1 to ILA1',
                'type': amplifier_type
            },
            {
                'uid': 'east edfa or fused in ILA1',
                'type': site_type
            },
            {
                'uid': 'east edfa in ILA2',
                'type': amplifier_type
            }, {
                'uid': 'west edfa in SITE2 to ILA1',
                'type': amplifier_type
            }
        ],
        'connections': [
            {
                'from_node': 'trx SITE1',
                'to_node': 'roadm SITE1'
            },
            {
                'from_node': 'roadm SITE1',
                'to_node': 'east edfa in SITE1 to ILA1'
            },
            {
                'from_node': 'east edfa in SITE1 to ILA1',
                'to_node': 'fiber (SITE1 → ILA1)'
            },
            {
                'from_node': 'fiber (SITE1 → ILA1)',
                'to_node': 'east edfa or fused in ILA1'
            },
            {
                'from_node': 'east edfa or fused in ILA1',
                'to_node': 'fiber (ILA1 → ILA2)'
            },
            {
                'from_node': 'fiber (ILA1 → ILA2)',
                'to_node': 'east edfa in ILA2'
            },
            {
                'from_node': 'east edfa in ILA2',
                'to_node': 'fiber (ILA2 → SITE2)'
            },
            {
                'from_node': 'fiber (ILA2 → SITE2)',
                'to_node': 'west edfa in SITE2 to ILA1'
            },
            {
                'from_node': 'west edfa in SITE2 to ILA1',
                'to_node': 'roadm SITE2'
            },
            {
                'from_node': 'roadm SITE2',
                'to_node': 'trx SITE2'
            }
        ]
    }
    multiband_amps = [e for e in base_network['elements'] if e['type'] == 'Multiband_amplifier']
    edfa2 = next(e for e in base_network['elements'] if e['uid'] == 'east edfa in ILA2')
    roadm1 = next(e for e in base_network['elements'] if e['uid'] == 'roadm SITE1')
    fused = [e for e in base_network['elements'] if e['type'] == 'Fused']
    if case == 'monoband_no_design_band':
        pass
    elif case == 'monoband_roadm':
        roadm1['params'] = {
            'design_bands': [
                {'f_min': 192.3e12, 'f_max': 196.0e12}
            ]
        }
    elif case == 'monoband_per_degree':
        roadm1['params'] = {
            'per_degree_design_bands': {
                'east edfa in SITE1 to ILA1': [
                    {'f_min': 191.5e12, 'f_max': 195.0e12}
                ]
            }
        }
    elif case == 'monoband_design':
        edfa2['type_variety'] = 'std_medium_gain'
    elif case == 'design':
        for elem in multiband_amps:
            elem['type_variety'] = 'std_medium_gain_multiband'
            elem['amplifiers'] = [{
                'type_variety': 'std_medium_gain',
                'operational': {
                    'delta_p': 0,
                    'tilt_target': 0
                }
            }, {
                'type_variety': 'std_medium_gain_L',
                'operational': {
                    'delta_p': -1,
                    'tilt_target': 0
                }
            }]
        for elem in fused:
            elem['params'] = {'loss': 0.0}
    elif case == 'no_design':
        # user must indicate the bands otherwise SI band (single band is assumed) and this is not
        # consistent with multiband amps.
        roadm1['params'] = {
            'per_degree_design_bands': {
                'east edfa in SITE1 to ILA1': [
                    {'f_min': 191.3e12, 'f_max': 196.0e12},
                    {'f_min': 187.0e12, 'f_max': 190.0e12}
                ]
            }
        }
    elif case == 'type_variety':
        # bands are implicit based on amplifiers type_varieties
        for elem in multiband_amps:
            elem['type_variety'] = 'std_medium_gain_multiband'
    return base_network


@pytest.mark.parametrize('case, site_type, amplifier_type, expected_design_bands, expected_per_degree_design_bands', [
    ('monoband_no_design_band', 'Edfa', 'Edfa',
     [{'f_min': 191.3e12, 'f_max': 196.1e12}], [{'f_min': 191.3e12, 'f_max': 196.1e12}]),
    ('monoband_roadm', 'Edfa', 'Edfa',
     [{'f_min': 192.3e12, 'f_max': 196.0e12}], [{'f_min': 192.3e12, 'f_max': 196.0e12}]),
    ('monoband_per_degree', 'Edfa', 'Edfa',
     [{'f_min': 191.3e12, 'f_max': 196.1e12}], [{'f_min': 191.5e12, 'f_max': 195.0e12}]),
    ('monoband_design', 'Edfa', 'Edfa',
     [{'f_min': 191.3e12, 'f_max': 196.1e12}], [{'f_min': 191.3e12, 'f_max': 196.1e12}]),
    ('design', 'Fused', 'Multiband_amplifier',
     [{'f_min': 191.3e12, 'f_max': 196.1e12}],
     [{'f_min': 186.55e12, 'f_max': 190.05e12}, {'f_min': 191.25e12, 'f_max': 196.15e12}]),
    ('no_design', 'Fused', 'Multiband_amplifier',
     [{'f_min': 191.3e12, 'f_max': 196.1e12}],
     [{'f_min': 187.0e12, 'f_max': 190.0e12}, {'f_min': 191.3e12, 'f_max': 196.0e12}])])
def test_design_band(case, site_type, amplifier_type, expected_design_bands, expected_per_degree_design_bands):
    """Check design_band is the one defined:
    - in SI if nothing is defined,
    - in ROADM if no design_band is defined for degree
    - in per_degree
    - if no design is defined,
        - if type variety is defined: use it for determining bands
        - if no type_variety autodesign is as expected, design uses OMS defined set of bands
    EOL is added only once on spans. One span can be one fiber or several fused fibers
    EOL is then added on the first fiber only.
    """
    json_data = network_base(case, site_type, amplifier_type=amplifier_type)
    equipment = load_equipment(EQPT_MULTBAND_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    build_network(network, equipment, p_db, p_total_db)
    roadm1 = next(n for n in network.nodes() if n.uid == 'roadm SITE1')
    assert roadm1.design_bands == expected_design_bands
    assert roadm1.per_degree_design_bands['east edfa in SITE1 to ILA1'] == expected_per_degree_design_bands


@pytest.mark.parametrize('raman_allowed, gain_target, power_target, target_extended_gain, warning, expected_selection', [
    (False, 20, 20, 3, False, ('test_fixed_gain', 0)),
    (False, 20, 25, 3, False, ('test_fixed_gain', -4)),
    (False, 10, 15, 3, False, ('std_low_gain_bis', 0)),
    (False, 5, 15, 3, "is below all available amplifiers min gain", ('std_low_gain_bis', 0)),
    (False, 30, 15, 3, "is beyond all available amplifiers capabilities", ('std_medium_gain', -1)),
])
def test_select_edfa(caplog, raman_allowed, gain_target, power_target, target_extended_gain, warning, expected_selection):
    """
    """
    equipment = load_equipment(EQPT_MULTBAND_FILENAME, EXTRA_CONFIGS)
    edfa_eqpt = {n: a for n, a in equipment['Edfa'].items() if a.type_def != 'multi_band'}
    selection = select_edfa(raman_allowed, gain_target, power_target, edfa_eqpt, "toto", target_extended_gain, verbose=True)
    assert selection == expected_selection
    if warning:
        assert warning in caplog.text


@pytest.mark.parametrize('cls, defaultparams, variety_list, booster_list, band, expected_restrictions', [
    (Edfa, EdfaParams, [], [],
     {'LBAND': {'f_min': 187.0e12, 'f_max': 190.0e12}},
     ['std_medium_gain_L', 'std_low_gain_L_ter', 'std_low_gain_L']),
    (Edfa, EdfaParams, [], [],
     {'CBAND': {'f_min': 191.3e12, 'f_max': 196.0e12}},
     ['CienaDB_medium_gain', 'std_medium_gain', 'std_low_gain', 'std_low_gain_bis', 'test', 'test_fixed_gain']),
    (Edfa, EdfaParams, ['std_medium_gain', 'std_high_gain'], [],
     {'CBAND': {'f_min': 191.3e12, 'f_max': 196.0e12}},
     ['std_medium_gain']),   # name in variety list does not exist in library
    (Edfa, EdfaParams, ['std_medium_gain', 'std_high_gain'], [],
     {'LBAND': {'f_min': 187.0e12, 'f_max': 190.0e12}},
     []),   # restrictions inconsistency with bands
    (Edfa, EdfaParams, ['std_medium_gain', 'std_high_gain'], ['std_booster'],
     {'CBAND': {'f_min': 191.3e12, 'f_max': 196.0e12}},
     ['std_medium_gain']),  # variety list takes precedence over booster constraint
    (Edfa, EdfaParams, [], ['std_booster'],
     {'CBAND': {'f_min': 191.3e12, 'f_max': 196.0e12}},
     ['std_booster']),
    (Multiband_amplifier, MultiBandParams, [], [],
     {'CBAND': {'f_min': 191.3e12, 'f_max': 196.0e12}, 'LBAND': {'f_min': 187.0e12, 'f_max': 190.0e12}},
     ['std_medium_gain_multiband', 'std_low_gain_multiband_bis']),
    (Multiband_amplifier, MultiBandParams, [], ['std_booster_multiband', 'std_booster'],
     {'CBAND': {'f_min': 191.3e12, 'f_max': 196.0e12}, 'LBAND': {'f_min': 187.0e12, 'f_max': 190.0e12}},
     ['std_booster_multiband'])
])
def test_get_node_restrictions(cls, defaultparams, variety_list, booster_list, band, expected_restrictions):
    """Check that all combinations of restrictions are correctly captured
    """
    equipment = load_equipment(EQPT_MULTBAND_FILENAME, EXTRA_CONFIGS)
    edfa_config = {"uid": "Edfa1"}
    if cls == Multiband_amplifier:
        edfa_config['amplifiers'] = {}
    edfa_config['params'] = defaultparams.default_values
    edfa_config['variety_list'] = variety_list
    node = cls(**edfa_config)
    roadm_config = {
        "uid": "roadm Brest_KLA",
        "params": {
            "per_degree_pch_out_db": {},
            "target_pch_out_dbm": -18,
            "add_drop_osnr": 38,
            "pmd": 0,
            "pdl": 0,
            "restrictions": {
                "preamp_variety_list": [],
                "booster_variety_list": booster_list
            },
            "roadm-path-impairments": []
        },
        "metadata": {
            "location": {
                "city": "Brest_KLA",
                "region": "RLD",
                "latitude": 4.0,
                "longitude": 0.0
            }
        }
    }
    prev_node = Roadm(**roadm_config)
    fiber_config = {
        "uid": "fiber (SITE1 → ILA1)",
        "type_variety": "SSMF",
        "params": {
            "length": 100.0,
            "loss_coef": 0.2,
            "length_units": "km"
        }
    }
    extra_params = equipment['Fiber']['SSMF'].__dict__

    fiber_config['params'] = merge_amplifier_restrictions(fiber_config['params'], extra_params)
    next_node = Fiber(**fiber_config)
    restrictions = get_node_restrictions(node, prev_node, next_node, equipment, band)
    assert restrictions == expected_restrictions


@pytest.mark.usefixtures('set_sim_params')
@pytest.mark.parametrize('case, site_type, band, expected_gain, expected_tilt, expected_variety, sim_params', [
    ('design', 'Multiband_amplifier', 'LBAND', 10.0, 0.0, 'std_medium_gain_multiband', False),
    ('no_design', 'Multiband_amplifier', 'LBAND', 10.0, 0.0, 'std_low_gain_multiband_bis', False),
    ('type_variety', 'Multiband_amplifier', 'LBAND', 10.0, 0.0, 'std_medium_gain_multiband', False),
    ('design', 'Multiband_amplifier', 'LBAND', 9.344985, 0.0, 'std_medium_gain_multiband', True),
    ('no_design', 'Multiband_amplifier', 'LBAND', 9.344985, -0.938676, 'std_low_gain_multiband_bis', True),
    ('no_design', 'Multiband_amplifier', 'CBAND', 10.977065, -1.600193, 'std_low_gain_multiband_bis', True),
    ('no_design', 'Fused', 'LBAND', 21.0, 0.0, 'std_medium_gain_multiband', False),
    ('no_design', 'Fused', 'LBAND', 20.344985, -0.819176, 'std_medium_gain_multiband', True),
    ('no_design', 'Fused', 'CBAND', 21.770319, -1.40032, 'std_medium_gain_multiband', True),
    ('design', 'Fused', 'CBAND', 21.21108, 0.0, 'std_medium_gain_multiband', True),
    ('design', 'Multiband_amplifier', 'CBAND', 11.041037, 0.0, 'std_medium_gain_multiband', True)])
def test_multiband(case, site_type, band, expected_gain, expected_tilt, expected_variety, sim_params):
    """Check:
    - if amplifiers are defined in multiband they are used for design,
    - if no design is defined,
        - if type variety is defined: use it for determining bands
        - if no type_variety autodesign is as expected, design uses OMS defined set of bands
    EOL is added only once on spans. One span can be one fiber or several fused fibers
    EOL is then added on the first fiber only.
    """
    json_data = network_base(case, site_type)
    equipment = load_equipment(EQPT_MULTBAND_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))

    if sim_params:
        SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    build_network(network, equipment, p_db, p_total_db)
    amp2 = next(n for n in network.nodes() if n.uid == 'east edfa in ILA2')
    # restore simParams
    save_sim_params = {"raman_params": SimParams._shared_dict['raman_params'].to_json(),
                       "nli_params": SimParams._shared_dict['nli_params'].to_json()}
    SimParams.set_params(save_sim_params)
    print(amp2.to_json)
    assert_allclose(amp2.amplifiers[band].effective_gain, expected_gain, atol=1e-5)
    assert_allclose(amp2.amplifiers[band].tilt_target, expected_tilt, atol=1e-5)
    assert amp2.type_variety == expected_variety


def test_tilt_fused():
    """check that computed tilt is the same for one span 100km as 2 spans 30 +70 km
    """
    design_bands = {'CBAND': {'f_min': 191.3e12, 'f_max': 196.0e12},
                    'LBAND': {'f_min': 187.0e12, 'f_max': 190.0e12}}
    save_sim_params = {"raman_params": SimParams._shared_dict['raman_params'].to_json(),
                       "nli_params": SimParams._shared_dict['nli_params'].to_json()}
    SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    input_powers = {'CBAND': 0.001, 'LBAND': 0.001}
    json_data = network_base("design", "Multiband_amplifier", length=100)
    equipment = load_equipment(EQPT_MULTBAND_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    node = next(n for n in network.nodes() if n.uid == 'fiber (SITE1 → ILA1)')
    tilt_db, tilt_target = estimate_srs_power_deviation(network, node, equipment, design_bands, input_powers)
    json_data = network_base("design", "Fused", length=50)
    equipment = load_equipment(EQPT_MULTBAND_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    node = next(n for n in network.nodes() if n.uid == 'fiber (ILA1 → ILA2)')
    fused_tilt_db, fused_tilt_target = \
        estimate_srs_power_deviation(network, node, equipment, design_bands, input_powers)
    # restore simParams
    SimParams.set_params(save_sim_params)
    for key in tilt_db:
        assert_allclose(tilt_db[key], fused_tilt_db[key], rtol=1e-3)
    for key in tilt_target:
        assert_allclose(tilt_target[key], fused_tilt_target[key], rtol=1e-3)


def network_wo_booster(site_type, bands):
    return {
        'elements': [
            {
                'uid': 'trx SITE1',
                'type': 'Transceiver'
            },
            {
                'uid': 'trx SITE2',
                'type': 'Transceiver'
            },
            {
                'uid': 'roadm SITE1',
                'params': {
                    'design_bands': bands
                },
                'type': 'Roadm'
            },
            {
                'uid': 'roadm SITE2',
                'type': 'Roadm'
            },
            {
                'uid': 'fiber (SITE1 → ILA1)',
                'type': 'Fiber',
                'type_variety': 'SSMF',
                'params': {
                    'length': 50.0,
                    'loss_coef': 0.2,
                    'length_units': 'km'
                }
            },
            {
                'uid': 'fiber (ILA1 → ILA2)',
                'type': 'Fiber',
                'type_variety': 'SSMF',
                'params': {
                    'length': 50.0,
                    'loss_coef': 0.2,
                    'length_units': 'km'
                }
            },
            {
                'uid': 'fiber (ILA2 → SITE2)',
                'type': 'Fiber',
                'type_variety': 'SSMF',
                'params': {
                    'length': 50.0,
                    'loss_coef': 0.2,
                    'length_units': 'km'
                }
            },
            {
                'uid': 'east edfa or fused in ILA1',
                'type': site_type
            }
        ],
        'connections': [
            {
                'from_node': 'trx SITE1',
                'to_node': 'roadm SITE1'
            },
            {
                'from_node': 'roadm SITE1',
                'to_node': 'fiber (SITE1 → ILA1)'
            },
            {
                'from_node': 'fiber (SITE1 → ILA1)',
                'to_node': 'east edfa or fused in ILA1'
            },
            {
                'from_node': 'east edfa or fused in ILA1',
                'to_node': 'fiber (ILA1 → ILA2)'
            },
            {
                'from_node': 'fiber (ILA1 → ILA2)',
                'to_node': 'fiber (ILA2 → SITE2)'
            },
            {
                'from_node': 'fiber (ILA2 → SITE2)',
                'to_node': 'roadm SITE2'
            },
            {
                'from_node': 'roadm SITE2',
                'to_node': 'trx SITE2'
            }
        ]
    }


@pytest.mark.parametrize('site_type, expected_type, bands, expected_bands', [
    ('Multiband_amplifier', Multiband_amplifier,
     [{'f_min': 187.0e12, 'f_max': 190.0e12}, {'f_min': 191.3e12, 'f_max': 196.0e12}],
     [{'f_min': 187.0e12, 'f_max': 190.0e12}, {'f_min': 191.3e12, 'f_max': 196.0e12}]),
    ('Edfa', Edfa,
     [{'f_min': 191.4e12, 'f_max': 196.1e12}],
     [{'f_min': 191.4e12, 'f_max': 196.1e12}]),
    ('Edfa', Edfa,
     [{'f_min': 191.2e12, 'f_max': 196.0e12}],
     []),
    ('Fused', Multiband_amplifier,
     [{'f_min': 187.0e12, 'f_max': 190.0e12}, {'f_min': 191.3e12, 'f_max': 196.0e12}],
     [{'f_min': 187.0e12, 'f_max': 190.0e12}, {'f_min': 191.3e12, 'f_max': 196.0e12}]),
    ('Fused', Edfa,
     [{'f_min': 191.3e12, 'f_max': 196.0e12}],
     [{'f_min': 191.3e12, 'f_max': 196.0e12}])])
def test_insert_amp(site_type, expected_type, bands, expected_bands):
    """Check:
    - if amplifiers are defined in multiband they are used for design,
    - if no design is defined,
        - if type variety is defined: use it for determining bands
        - if no type_variety autodesign is as expected, design uses OMS defined set of bands
    EOL is added only once on spans. One span can be one fiber or several fused fibers
    EOL is then added on the first fiber only.
    """
    json_data = network_wo_booster(site_type, bands)
    equipment = load_equipment(EQPT_MULTBAND_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    p_db = equipment['SI']['default'].power_dbm
    p_total_db = p_db + lin2db(automatic_nch(equipment['SI']['default'].f_min,
                                             equipment['SI']['default'].f_max, equipment['SI']['default'].spacing))
    add_missing_elements_in_network(network, equipment)
    if not expected_bands:
        with pytest.raises(ConfigurationError):
            build_network(network, equipment, p_db, p_total_db)
    else:
        build_network(network, equipment, p_db, p_total_db)
        roadm1 = next(n for n in network.nodes() if n.uid == 'roadm SITE1')
        amp1 = get_next_node(roadm1, network)
        assert isinstance(amp1, expected_type)
        assert roadm1.per_degree_design_bands['Edfa_booster_roadm SITE1_to_fiber (SITE1 → ILA1)'] == expected_bands
