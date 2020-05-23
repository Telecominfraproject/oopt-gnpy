# -*- coding: utf-8 -*-

from pathlib import Path
import os
import pytest
from gnpy.tools.cli_examples import transmission_main_example, path_requests_run

SRC_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("output, handler, args", (
    ('transmission_main_example', transmission_main_example, []),
    ('path_requests_run', path_requests_run, []),
    ('transmission_main_example__raman', transmission_main_example,
     ['examples/raman_edfa_example_network.json', '--sim', 'examples/sim_params.json', '--show-channels',]),
))
def test_example_invocation(capsys, output, handler, args):
    '''Make sure that our examples produce useful output'''
    os.chdir(SRC_ROOT)
    expected = open(SRC_ROOT / 'tests' / 'invocation' / output, mode='r').read()
    handler(args)
    captured = capsys.readouterr()
    assert captured.out == expected
    assert captured.err == ''
