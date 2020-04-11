# -*- coding: utf-8 -*-

from pathlib import Path
import pytest
import subprocess

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent

@pytest.mark.parametrize("invocation", (
    ('./examples/transmission_main_example.py',),
    ('./examples/path_request_run.py',),
    ('./examples/transmission_main_example.py', 'examples/raman_edfa_example_network.json', '--sim', 'examples/sim_params.json', '--show-channels',),
))
def test_example_invocation(invocation):
    '''Make sure that our examples produce useful output'''
    proc = subprocess.run(invocation, check=True)
