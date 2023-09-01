# SPDX-License-Identifier: BSD-3-Clause
#
# Tests for YANG models of GNPy
#
# Copyright (C) 2020-2022 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from gnpy.yang import load_data, Error as YangError
import json
from pathlib import Path
import pytest


def _get_basename(filename: Path) -> str:
    try:
        return filename.name
    except AttributeError:
        return filename


@pytest.mark.parametrize("filename", (Path(__file__).parent / 'yang').glob('*.json'), ids=_get_basename)
def test_validate_yang_data(filename: Path):
    '''Validate a JSON file against our YANG models'''
    data = load_data(filename.read_text())


@pytest.mark.parametrize("data, error_message, where", (
    ({"tip-photonic-equipment:amplifier": [{"type": "fixed", "polynomial-NF": {"x": 666}, "frequency-min": "666.666"}]},
     'Node "x" not found as a child of "polynomial-NF" node.',
     'Data location "/tip-photonic-equipment:amplifier[type=\'fixed\']/polynomial-NF", line number 1.'),
    ({"tip-photonic-equipment:amplifier": [{"frequency-min": "666.666"}]},
     'Unsatisfied range - value "666.666" is out of the allowed range.',
     'Data location "/tip-photonic-equipment:amplifier/frequency-min", line number 1.'),
))
def test_invalid_yang_json(data, error_message, where):
    with pytest.raises(YangError) as excinfo:
        load_data(json.dumps(data))
    exc = excinfo.value
    assert exc.errors != []
    assert exc.errors[0].what.startswith(error_message)
    assert exc.errors[0].where.startswith(where)
