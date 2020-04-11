# -*- coding: utf-8 -*-

from pathlib import Path
import os
import pytest
import subprocess

SRC_ROOT = Path(__file__).parent.parent

@pytest.mark.parametrize("invocation", (
    ('./examples/transmission_main_example.py',),
    ('./examples/path_requests_run.py',),
    ('./examples/transmission_main_example.py', 'examples/raman_edfa_example_network.json', '--sim', 'examples/sim_params.json', '--show-channels',),
))
def test_example_invocation(invocation):
    '''Make sure that our examples produce useful output'''
    os.chdir(SRC_ROOT)
    proc = subprocess.run(invocation, check=True)
