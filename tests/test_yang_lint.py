# SPDX-License-Identifier: BSD-3-Clause
#
# Tests for YANG models of GNPy
#
# Copyright (C) 2020-2022 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

from gnpy.yang import external_path, model_path
import os
from pathlib import Path
from typing import List
import pytest
import oopt_gnpy_libyang as ly


def _get_basename(filename: Path) -> str:
    try:
        return filename.name
    except AttributeError:
        return filename


@pytest.mark.parametrize("yang_model", [x for x in external_path().glob('*.yang')] + [x for x in model_path().glob('*.yang')], ids=_get_basename)
def test_lint_yang(yang_model):
    '''Run a linter on each YANG model'''
    c = ly.Context(str(external_path()) + os.pathsep + str(model_path()),
                   ly.ContextOptions.AllImplemented | ly.ContextOptions.DisableSearchCwd)
    assert c.parse_module(yang_model, ly.SchemaFormat.YANG) is not None
    assert c.errors() == []
