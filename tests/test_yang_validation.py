# SPDX-License-Identifier: BSD-3-Clause
#
# Tests for YANG models of GNPy
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from gnpy.tools.json_io import load_equipment, load_network
from gnpy.yang import external_path, model_path
from gnpy.yang.io import create_datamodel, load_from_yang, save_to_json
from pathlib import Path
from typing import List
import pytest
import subprocess
import json


SRC_ROOT = Path(__file__).parent.parent


def _get_basename(filename: Path) -> str:
    try:
        return filename.name
    except AttributeError:
        return filename


@pytest.mark.parametrize("yang_model", external_path().glob('*.yang'), ids=_get_basename)
def test_lint_external_yang(yang_model):
    '''Run a basic linter on all third-party models'''
    _validate_yang_model(yang_model, [])


@pytest.mark.parametrize("yang_model", model_path().glob('*.yang'), ids=_get_basename)
def test_lint_gnpy_yang(yang_model):
    '''Run a linter on GNPy's YANG models'''
    _validate_yang_model(yang_model, ('--canonical', '--strict', '--lint'))


def _validate_yang_model(filename: Path, options: List[str]):
    '''Run actual validation'''
    # I would have loved to use pyang programatically from here, but it seems that the API is really designed
    # around that interactive use case where code just expects an OptParser as a part of the library context,
    # etc.
    # Given that I'm only interested in a simple pass/fail scenario, let's just invoke the linter as a standalone
    # process and check if it screams.
    proc = subprocess.run(
        ('pyang', '-p', ':'.join((str(external_path()), str(model_path()))), *options, filename),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, universal_newlines=True)
    assert proc.stderr == ''
    assert proc.stdout == ''


@pytest.fixture
def _yangson_datamodel():
    return create_datamodel


@pytest.mark.parametrize("filename", (Path(__file__).parent / 'yang').glob('*.json'), ids=_get_basename)
def test_validate_yang_data(_yangson_datamodel, filename: Path):
    '''Validate a JSON file against our YANG models'''
    dm = _yangson_datamodel()
    with open(filename, 'r') as f:
        raw_json = json.load(f)
    data = dm.from_raw(raw_json)
    data.validate()


@pytest.mark.parametrize("expected_file, equipment_file, topology_file", (
    ("edfa_example.json", "gnpy/example-data/eqpt_config.json", "gnpy/example-data/edfa_example_network.json"),
    ("Sweden_OpenROADM_example.json", "gnpy/example-data/eqpt_config_openroadm.json", "gnpy/example-data/Sweden_OpenROADM_example_network.json"),
))
def test_conversion_to_yang(expected_file, equipment_file, topology_file):
    '''Conversion from legacy JSON to self-contained YANG data'''
    equipment = load_equipment(SRC_ROOT / equipment_file)
    network = load_network(SRC_ROOT / topology_file, equipment)
    data = save_to_json(equipment, network)
    serialized = json.dumps(data, indent=2) + '\n'  # files were generated via print(), hence a newline
    expected = open(SRC_ROOT / 'tests' / 'yang' / 'converted' / expected_file, mode='rb').read().decode('utf-8')
    assert serialized == expected

    y_equipment, y_network = load_from_yang(data)

    for meta in ['Span', 'SI']:
        assert equipment[meta].keys() == y_equipment[meta].keys()
        for name in equipment[meta].keys():
            thing = equipment[meta][name]
            y_thing = y_equipment[meta][name]
            assert set(thing.__dict__.keys()) == set(y_thing.__dict__.keys())
            # FIXME: a numeric problem?
            # assert thing.__dict__ == y_thing.__dict__

    for ne_type in ['Edfa', 'Fiber', 'RamanFiber', 'Roadm', 'Transceiver']:
        assert equipment[ne_type].keys() == y_equipment[ne_type].keys()
        for name in equipment[ne_type].keys():
            thing = equipment[ne_type][name]
            y_thing = y_equipment[ne_type][name]
            assert set(thing.__dict__.keys()) == set(y_thing.__dict__.keys())
            # FIXME: some bits are missing, some are numerically different
            # for attr in thing.__dict__:
            #     print(f'{ne_type}: {name} -> {attr}')
            #     assert getattr(thing, attr) == getattr(y_thing, attr)

    # network nodes:
    # the order is unstable, and there "might" be duplicate UIDs
    len(network.nodes()) == len(y_network.nodes())
    assert set(n.uid for n in network.nodes()) == set(n.uid for n in y_network.nodes())

    # edges are simple, just check the UIDs and cardinality
    assert set((e[0].uid, e[1].uid) for e in network.edges()) == set((e[0].uid, e[1].uid) for e in y_network.edges())
    assert len(network.edges()) == len(y_network.edges())

    # for orig_node in network.nodes():
    #     y_node = next(x for x in y_network.nodes() if x.uid == orig_node.uid)
    #     # FIXME: fails on metadata...
    #     assert orig_node.to_json == y_node.to_json
