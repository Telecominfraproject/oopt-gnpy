#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: Esther Le Rouzic
# @Date:   2019-05-22
"""
@author: esther.lerouzic
checks behaviour of of span_loss function
"""

from pathlib import Path
import pytest
from gnpy.core.network import span_loss
from gnpy.tools.json_io import load_equipment, load_network


TEST_DIR = Path(__file__).parent
EQPT_FILENAME = TEST_DIR / 'data/eqpt_config.json'
NETWORK_FILENAME = TEST_DIR / 'data/bugfixiteratortopo.json'

def test_iterator():
    """ check that span loss return correct values with various successions of elements
    """
    equipment = load_equipment(EQPT_FILENAME)
    network = load_network(NETWORK_FILENAME, equipment)
    ref = {'fiber1': 10.5, 'fiber2': 10.5, 'fiber3': 16.0, 'fiber4': 16.0, 'fiber6': 17.0, 'fused4': 1, 'fused7': 1}
    for node in network.nodes():
        loss = span_loss(network, node)
        if node.uid in ref:
            assert loss == ref[node.uid]
        else:
            assert loss == 0.0
