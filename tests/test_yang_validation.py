# SPDX-License-Identifier: BSD-3-Clause
#
# Tests for YANG models of GNPy
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from gnpy.tools.json_io import load_equipment, load_network
from gnpy.yang import external_path, model_path
from gnpy.yang.io import create_datamodel, save_to_json
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
