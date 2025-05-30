#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_amplifier
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

import pytest
from gnpy.core.parameters import SimParams, NLIParams, RamanParams


@pytest.fixture
def set_sim_params(monkeypatch):
    monkeypatch.setattr(SimParams, '_shared_dict', {'nli_params': NLIParams(), 'raman_params': RamanParams()})
