# -*- coding: utf-8 -*-

from pathlib import Path
import os
import pytest
import subprocess

SRC_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("output, invocation", (
    ('transmission_main_example',
        ('./examples/transmission_main_example.py',)),
    ('path_requests_run',
        ('./examples/path_requests_run.py',)),
    ('transmission_main_example__raman',
        ('./examples/transmission_main_example.py', 'examples/raman_edfa_example_network.json',
         '--sim', 'examples/sim_params.json', '--show-channels',)),
))
def test_example_invocation(output, invocation):
    '''Make sure that our examples produce useful output'''
    os.chdir(SRC_ROOT)
    expected = open(SRC_ROOT / 'tests' / 'invocation' / output, mode='rb').read()
    proc = subprocess.run(invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert proc.stderr == b''
    assert proc.stdout == expected
