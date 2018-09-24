#!/usr/bin/env python

import pytest
from gnpy.core.utils import db2lin

def test_db2lin():
    assert pytest.approx(10.0) == db2lin(10.0)

if __name__ == '__main__':
    from logging import getLogger, basicConfig, INFO
    logger = getLogger(__name__)
    basicConfig(level=INFO)

    logger.info(f'Running {test}')
    test_db2lin()
