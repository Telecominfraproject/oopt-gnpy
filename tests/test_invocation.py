# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import subprocess

@pytest.mark.parametrize("output, invocation", (
    ('transmission_main_example',
        ('./examples/transmission_main_example.py',)),
    ('path_request_run',
        ('./examples/path_requests_run.py',)),
    ('transmission_main_example__raman',
        ('./examples/transmission_main_example.py', 'examples/raman_edfa_example_network.json',
         '--sim', 'examples/sim_params.json', '--show-channels',)),
))
def test_example_invocation(output, invocation):
    '''Make sure that our examples produce useful output'''
    expected = open(Path(__file__).parent / 'invocation' / output, mode='rb').read()
    proc = subprocess.run(invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    assert proc.stderr == b''
    assert proc.stdout == expected
