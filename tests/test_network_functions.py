# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from pathlib import Path
import pytest
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
    # not connected anywhere
    ['fused4', 1],
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
