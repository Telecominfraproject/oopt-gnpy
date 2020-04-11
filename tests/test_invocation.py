# -*- coding: utf-8 -*-

import pytest
import subprocess

def test_aaa_env():
    print(subprocess.run(['pwd'], check=True).stdout)
    assert(subprocess.run(['ls', '-al', '--recursive'], check=True).stdout == b'')

@pytest.mark.parametrize("invocation", (
    ('./examples/transmission_main_example.py',),
    ('./examples/path_request_run.py',),
    ('./examples/transmission_main_example.py', 'examples/raman_edfa_example_network.json', '--sim', 'examples/sim_params.json', '--show-channels',),
))
def test_example_invocation(invocation):
    '''Make sure that our examples produce useful output'''
    proc = subprocess.run(invocation, check=True)
