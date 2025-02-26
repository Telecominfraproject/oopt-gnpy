#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_info
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
Checks autodesign functions
"""

from pathlib import Path
import pytest
from numpy.testing import assert_allclose
from numpy import mean

from gnpy.core.exceptions import NetworkTopologyError, ConfigurationError
from gnpy.core.network import span_loss, build_network, select_edfa, get_node_restrictions, \
    estimate_srs_power_deviation, add_missing_elements_in_network, get_next_node
from gnpy.tools.json_io import load_equipment, load_network, network_from_json, load_json
from gnpy.core.utils import watt2dbm, automatic_nch, merge_amplifier_restrictions, dbm2watt
from gnpy.core.info import create_input_spectral_information
from gnpy.core.elements import Fiber, Edfa, Roadm, Multiband_amplifier, Transceiver
from gnpy.core.parameters import SimParams, EdfaParams, MultiBandParams
from gnpy.topology.request import PathRequest


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
EQPT_MULTBAND_FILENAME = DATA_DIR / 'eqpt_config_multiband.json'
NETWORK_FILENAME = DATA_DIR / 'bugfixiteratortopo.json'
EXTRA_CONFIGS = {"std_medium_gain_advanced_config.json": load_json(DATA_DIR / "std_medium_gain_advanced_config.json")}


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


def in_voa_json_data(gain, delta_p, in_voa, out_voa):
    """json_data for test network with in_voa
    """
    return {
        "elements": [{
            "uid": "Tx",
            "type": "Transceiver",
        }, {
            "uid": "Span0",
            "type": "Fiber",
            "type_variety": "SSMF",
            "params": {
                "length": 100,
                "loss_coef": 0.2,
                "length_units": "km"}
        }, {
            "uid": "Edfa1",
            "type": "Edfa",
            "type_variety": "test",
            "operational": {
                "delta_p": delta_p,
                "gain_target": gain,
                "tilt_target": 0,
                "out_voa": out_voa,
                "in_voa": in_voa}
        }, {
            "uid": "Span1",
            "type": "Fiber",
            "type_variety": "SSMF",
            "params": {
                "length": 80,
                "loss_coef": 0.2,
                "length_units": "km"}
        }, {
            "uid": "Edfa2",
            "type": "Edfa",
            "type_variety": "test",
            "operational": {
                "delta_p": -1,
                "gain_target": 17,
                "tilt_target": 0,
                "out_voa": 2,
                "in_voa": 1}
        }, {
            "uid": "Span2",
            "type": "Fiber",
            "type_variety": "SSMF",
            "params": {
                "length": 100,
                "loss_coef": 0.2,
                "length_units": "km"}
        }, {
            "uid": "Rx",
            "type": "Transceiver",
        }],
        "connections": [{
            "from_node": "Tx",
            "to_node": "Span0"
        }, {
            "from_node": "Span0",
            "to_node": "Edfa1"
        }, {
            "from_node": "Edfa1",
            "to_node": "Span1"
        }, {
            "from_node": "Span1",
            "to_node": "Edfa2"
        }, {
            "from_node": "Edfa2",
            "to_node": "Span2"
        }, {
            "from_node": "Span2",
            "to_node": "Rx"
        }]
    }

def pathrequest(pch_dbm: float, p_tot_dbm: float = None, nb_channels: int = None):
    """create ref channel for defined power settings
    """
    params = {
        "power": dbm2watt(pch_dbm),
        "tx_power": dbm2watt(pch_dbm),
        "nb_channel": nb_channels if nb_channels else round(dbm2watt(p_tot_dbm) / dbm2watt(pch_dbm), 0)
    }
    return PathRequest(**params)


@pytest.mark.parametrize('out_voa', [None, 0, 1, 2])
@pytest.mark.parametrize('in_voa', [None, 0, 1, 2])
def test_invoa(in_voa, out_voa):
    """Check that in_voa is correctly loaded and applied"""
    gain = 20
    delta_p = 0
    json_data = in_voa_json_data(gain, delta_p, in_voa, out_voa)
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    # Build the network
    p_db = 0
    build_network(network, equipment, pathrequest(p_db, nb_channels=96))
    [edfa1, edfa2] = [n for n in network.nodes() if isinstance(n, Edfa)]
    [span0, span1, span2] = [n for n in network.nodes() if isinstance(n, Fiber)]
    [tx, rx] = [n for n in network.nodes() if isinstance(n, Transceiver)]

    si = create_input_spectral_information(f_min=191.3e12, f_max=196.05e12,
                                           roll_off=0.15, baud_rate=32e9,
                                           spacing=50e9, tx_osnr=None, tx_power=1e-3)
    si = span0(si)
    assert_allclose(si.pch_dbm, - 20, atol=1e-1)
    si = edfa1(si)
    # in case voa are set to None, build network is supposed to change them to 0
    if in_voa is None:
        in_voa = 0
    if out_voa is None:
        out_voa = 0
    assert edfa1.in_voa == in_voa
    # power_mode is true so gain is computed to obtain delta_p
    # input power in amp after in_voa is -20 - in_voa
    # so in order to get 0dBm out of amp (before out_voa), gain must be delta_p + 20 + in_voa
    assert edfa1.effective_gain == delta_p + 20 + in_voa
    assert_allclose(si.pch_dbm, delta_p - out_voa, atol=1e-1)
    si = span1(si)
    assert_allclose(si.pch_dbm, delta_p - out_voa - 16, atol=1e-1)
    si = edfa2(si)
    assert edfa2.in_voa == 1
    assert edfa2.effective_gain == edfa2.delta_p + 16 + out_voa + edfa2.in_voa
    assert_allclose(si.pch_dbm, edfa2.delta_p - edfa2.out_voa, atol=1e-1)


@pytest.mark.parametrize('out_voa', [0, 1, 2])
@pytest.mark.parametrize('in_voa', [0, 1, 2])
def test_invoa_gainmode(in_voa, out_voa):
    """Check that in_voa is correctly loaded and applied also with power_mode = False"""
    gain = 20
    delta_p = 0
    json_data = in_voa_json_data(gain, delta_p, in_voa, out_voa)
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    equipment['Span']['default'].power_mode = False
    network = network_from_json(json_data, equipment)
    # Build the network
    p_db = 0
    build_network(network, equipment, pathrequest(p_db, nb_channels=96))
    [edfa1, edfa2] = [n for n in network.nodes() if isinstance(n, Edfa)]
    [span0, span1, span2] = [n for n in network.nodes() if isinstance(n, Fiber)]
    [tx, rx] = [n for n in network.nodes() if isinstance(n, Transceiver)]

    si = create_input_spectral_information(f_min=191.3e12, f_max=196.05e12,
                                           roll_off=0.15, baud_rate=32e9,
                                           spacing=50e9, tx_osnr=None, tx_power=1e-3)
    si = span0(si)
    assert_allclose(si.pch_dbm, - 20, atol=1e-1)
    si = edfa1(si)
    # power_mode is false so gain is applied
    assert edfa1.effective_gain == gain
    assert_allclose(si.pch_dbm, -20 - in_voa + gain - out_voa, atol=1e-1)
    si = span1(si)
    assert_allclose(si.pch_dbm, -20 - in_voa + gain - out_voa - 16, atol=1e-1)
    si = edfa2(si)
    assert edfa2.effective_gain == 17
    assert_allclose(si.pch_dbm,
                    -20 - in_voa + gain - out_voa   # first span
                    - 16 - edfa2.in_voa + edfa2.effective_gain - edfa2.out_voa, atol=1e-1)


def test_too_small_gain():
    """Check that padding attenuation integrate in_voa in its computation
    In this scenario target power leads to gain smaller than min gain (15dB in this example)
    Then amplifier adds attenuation at the input to ensure gain is 15 (with nf max = 10) .
    this test ensures that if user already define in_voa, this value is accounted to compute
    attenuation
    """
    gain = 20
    delta_p = -17
    in_voa = 2
    out_voa = 3
    json_data = in_voa_json_data(gain, delta_p, in_voa, out_voa)
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    network = network_from_json(json_data, equipment)
    # Build the network
    p_db = 0
    build_network(network, equipment, pathrequest(p_db, nb_channels=96))
    [edfa1, edfa2] = [n for n in network.nodes() if isinstance(n, Edfa)]
    [span0, span1, span2] = [n for n in network.nodes() if isinstance(n, Fiber)]
    [tx, rx] = [n for n in network.nodes() if isinstance(n, Transceiver)]
    si = create_input_spectral_information(f_min=191.3e12, f_max=196.05e12,
                                           roll_off=0.15, baud_rate=32e9,
                                           spacing=50e9, tx_osnr=None, tx_power=1e-3)
    si = span0(si)
    si = edfa1(si)
    # padd att_in is added to ensure that the amp gain is 15 (min value)
    # min output power is power_in - in_voa + gain min. if min output power is greater than
    # target power delta_p, then padd attenuation is added
    # assert that power_in - in_voa + gain min - delta_p == pad attenuation
    assert 15 - 20 - in_voa - delta_p == edfa1.att_in
    # assert effective gain is amp gain - pad att in + in voa
    assert edfa1.effective_gain == 15 - edfa1.att_in
    assert_allclose(edfa1.nf, 10 + edfa1.att_in, atol=1e-10)


def test_invoa_nf():
    """for a given gain assert that if loss is set before amp (in_voa) generated ASE is increased
    by in_voa value, because power level in amp is reduced by in_voa
    """
    # Build the network with loss set at the output
    json_data = in_voa_json_data(gain=25, delta_p=None, in_voa=0, out_voa=5)
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    equipment['Span']['default'].power_mode = False
    network = network_from_json(json_data, equipment)
    p_db = 0
    build_network(network, equipment, pathrequest(p_db, nb_channels=50))
    [edfa1, edfa2] = [n for n in network.nodes() if isinstance(n, Edfa)]
    [span0, span1, span2] = [n for n in network.nodes() if isinstance(n, Fiber)]
    [tx, rx] = [n for n in network.nodes() if isinstance(n, Transceiver)]
    # with out_voa, with pmax = 21 dBm, can only apply max gain if total power in is below
    # -4 dBm. this is the case with pch = 0.8 mW and 47 channels at fiber input
    si = create_input_spectral_information(f_min=191.3e12, f_max=196.05e12,
                                           roll_off=0.15, baud_rate=32e9,
                                           spacing=100e9, tx_osnr=None, tx_power=0.0008)
    si = span0(si)
    si = edfa1(si)
    nf_after = edfa1.nf
    ase_after = watt2dbm(si.ase)

    # Build the same network with loss set at the input
    json_data = in_voa_json_data(gain=25, delta_p=None, in_voa=5, out_voa=0)
    equipment = load_equipment(EQPT_FILENAME, EXTRA_CONFIGS)
    equipment['Span']['default'].power_mode = False
    network = network_from_json(json_data, equipment)
    # Build the network
    p_db = 0
    build_network(network, equipment, pathrequest(p_db, nb_channels=50))
    [edfa1, edfa2] = [n for n in network.nodes() if isinstance(n, Edfa)]
    [span0, span1, span2] = [n for n in network.nodes() if isinstance(n, Fiber)]
    [tx, rx] = [n for n in network.nodes() if isinstance(n, Transceiver)]

    si = create_input_spectral_information(f_min=191.3e12, f_max=196.05e12,
                                           roll_off=0.15, baud_rate=32e9,
                                           spacing=100e9, tx_osnr=None, tx_power=1e-3)
    si = span0(si)
    si = edfa1(si)
    nf_before = edfa1.nf
    ase_before = watt2dbm(si.ase)

    # check that working point of amp is exactly the same: same gain => same nf
    assert mean(nf_before) == mean(nf_after)
    # check that generated ASE is exactly 5 dB more when loss is set in in_voa
    assert_allclose(ase_before, ase_after + 5, atol=1e-10)


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
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)

    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
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

    build_network(network, equipment, pathrequest(p_db, p_total_db))
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
                {'f_min': 192.3e12, 'f_max': 196.0e12, 'spacing': 50e9}
            ]
        }
    elif case == 'monoband_per_degree':
        roadm1['params'] = {
            'per_degree_design_bands': {
                'east edfa in SITE1 to ILA1': [
                    {'f_min': 191.5e12, 'f_max': 195.0e12, 'spacing': 50e9}
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
     [{'f_min': 191.3e12, 'f_max': 196.1e12, 'spacing': 50e9}], [{'f_min': 191.3e12, 'f_max': 196.1e12, 'spacing': 50e9}]),
    ('monoband_roadm', 'Edfa', 'Edfa',
     [{'f_min': 192.3e12, 'f_max': 196.0e12, 'spacing': 50e9}], [{'f_min': 192.3e12, 'f_max': 196.0e12, 'spacing': 50e9}]),
    ('monoband_per_degree', 'Edfa', 'Edfa',
     [{'f_min': 191.3e12, 'f_max': 196.1e12, 'spacing': 50e9}], [{'f_min': 191.5e12, 'f_max': 195.0e12, 'spacing': 50e9}]),
    ('monoband_design', 'Edfa', 'Edfa',
     [{'f_min': 191.3e12, 'f_max': 196.1e12, 'spacing': 50e9}], [{'f_min': 191.3e12, 'f_max': 196.1e12, 'spacing': 50e9}]),
    ('design', 'Fused', 'Multiband_amplifier',
     [{'f_min': 191.3e12, 'f_max': 196.1e12, 'spacing': 50e9}],
     [{'f_min': 186.55e12, 'f_max': 190.05e12, 'spacing': 50e9}, {'f_min': 191.25e12, 'f_max': 196.15e12, 'spacing': 50e9}]),
    ('no_design', 'Fused', 'Multiband_amplifier',
     [{'f_min': 191.3e12, 'f_max': 196.1e12, 'spacing': 50e9}],
     [{'f_min': 187.0e12, 'f_max': 190.0e12, 'spacing': 50e9}, {'f_min': 191.3e12, 'f_max': 196.0e12, 'spacing': 50e9}])])
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
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
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
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)

    if sim_params:
        SimParams.set_params(load_json(TEST_DIR / 'data' / 'sim_params.json'))
    build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
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
     [{'f_min': 187.0e12, 'f_max': 190.0e12, "spacing": 50e9}, {'f_min': 191.3e12, 'f_max': 196.0e12, "spacing": 50e9}],
     [{'f_min': 187.0e12, 'f_max': 190.0e12, "spacing": 50e9}, {'f_min': 191.3e12, 'f_max': 196.0e12, "spacing": 50e9}]),
    ('Edfa', Edfa,
     [{'f_min': 191.4e12, 'f_max': 196.1e12, "spacing": 50e9}],
     [{'f_min': 191.4e12, 'f_max': 196.1e12, "spacing": 50e9}]),
    ('Edfa', Edfa,
     [{'f_min': 191.2e12, 'f_max': 196.0e12, "spacing": 50e9}],
     []),
    ('Fused', Multiband_amplifier,
     [{'f_min': 187.0e12, 'f_max': 190.0e12, "spacing": 50e9}, {'f_min': 191.3e12, 'f_max': 196.0e12, "spacing": 50e9}],
     [{'f_min': 187.0e12, 'f_max': 190.0e12, "spacing": 50e9}, {'f_min': 191.3e12, 'f_max': 196.0e12, "spacing": 50e9}]),
    ('Fused', Edfa,
     [{'f_min': 191.3e12, 'f_max': 196.0e12, "spacing": 50e9}],
     [{'f_min': 191.3e12, 'f_max': 196.0e12, "spacing": 50e9}])])
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
    nb_channels = automatic_nch(equipment['SI']['default'].f_min,
                                equipment['SI']['default'].f_max, equipment['SI']['default'].spacing)
    add_missing_elements_in_network(network, equipment)
    if not expected_bands:
        with pytest.raises(ConfigurationError):
            build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
    else:
        build_network(network, equipment, pathrequest(p_db, nb_channels=nb_channels))
        roadm1 = next(n for n in network.nodes() if n.uid == 'roadm SITE1')
        amp1 = get_next_node(roadm1, network)
        assert isinstance(amp1, expected_type)
        assert roadm1.per_degree_design_bands['Edfa_booster_roadm SITE1_to_fiber (SITE1 → ILA1)'] == expected_bands
