# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from pathlib import Path
import pytest
from gnpy.core.exceptions import NetworkTopologyError
from gnpy.core.network import span_loss, build_network
from gnpy.tools.json_io import load_equipment, load_network, network_from_json
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.core.elements import Fiber, Edfa


TEST_DIR = Path(__file__).parent
EQPT_FILENAME = TEST_DIR / 'data/eqpt_config.json'
NETWORK_FILENAME = TEST_DIR / 'data/bugfixiteratortopo.json'


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
    equipment = load_equipment(EQPT_FILENAME)
    network = load_network(NETWORK_FILENAME, equipment)
    for x in network.nodes():
        if x.uid == node:
            assert attenuation == span_loss(network, x, equipment)
            return
    assert not f'node "{node}" referenced from test but not found in the topology'  # pragma: no cover


@pytest.mark.parametrize("node", ['fused4'])
def test_span_loss_unconnected(node):
    '''Fused node that has no next and no previous nodes should be detected'''
    equipment = load_equipment(EQPT_FILENAME)
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
    equipment = load_equipment(EQPT_FILENAME)
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
    equipment = load_equipment(EQPT_FILENAME)
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
