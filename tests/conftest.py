# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2020 Telecom Infra Project and GNPy contributors
# see LICENSE.md for a list of contributors
#

import pytest
from gnpy.core.parameters import SimParams, NLIParams, RamanParams


@pytest.fixture
def set_sim_params(monkeypatch):
    monkeypatch.setattr(SimParams, '_shared_dict', {'nli_params': NLIParams(), 'raman_params': RamanParams()})
