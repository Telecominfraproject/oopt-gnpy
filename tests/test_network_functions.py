# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from pathlib import Path
import pytest
from gnpy.core.exceptions import NetworkTopologyError
from gnpy.core.network import span_loss
from gnpy.tools.json_io import load_equipment, load_network


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
            assert attenuation == span_loss(network, x)
            return
    assert not f'node "{node}" referenced from test but not found in the topology'  # pragma: no cover


@pytest.mark.parametrize("node", ['fused4'])
def test_span_loss_unconnected(node):
    '''Fused node that has no next and no previous nodes should be detected'''
    equipment = load_equipment(EQPT_FILENAME)
    network = load_network(NETWORK_FILENAME, equipment)
    x = next(x for x in network.nodes() if x.uid == node)
    with pytest.raises(NetworkTopologyError):
        span_loss(network, x)
