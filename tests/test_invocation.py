# -*- coding: utf-8 -*-

from pathlib import Path
import os
import pytest
import subprocess
from gnpy.tools.cli_examples import transmission_main_example, path_requests_run

SRC_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("output, handler, args", (
    ('transmission_main_example', transmission_main_example, []),
    ('transmission_saturated', transmission_main_example,
     ['tests/data/testTopology_expected.json', 'lannion', 'lorient', '-e', 'tests/data/eqpt_config.json', '--pow', '3']),
    ('path_requests_run', path_requests_run, []),
    ('transmission_main_example__raman', transmission_main_example,
     ['gnpy/example-data/raman_edfa_example_network.json', '--sim', 'gnpy/example-data/sim_params.json', '--show-channels', ]),
    ('openroadm-v4-Stockholm-Gothenburg', transmission_main_example,
     ['-e', 'gnpy/example-data/eqpt_config_openroadm_ver4.json', 'gnpy/example-data/Sweden_OpenROADMv4_example_network.json', ]),
    ('openroadm-v5-Stockholm-Gothenburg', transmission_main_example,
     ['-e', 'gnpy/example-data/eqpt_config_openroadm_ver5.json', 'gnpy/example-data/Sweden_OpenROADMv5_example_network.json', ]),
    ('transmission_main_example_long', transmission_main_example,
     ['-e', 'tests/data/eqpt_config.json', 'tests/data/test_long_network.json']),
    ('spectrum1_transmission_main_example', transmission_main_example,
     ['--spectrum', 'gnpy/example-data/initial_spectrum1.json', 'gnpy/example-data/meshTopologyExampleV2.xls', ]),
    ('spectrum2_transmission_main_example', transmission_main_example,
     ['--spectrum', 'gnpy/example-data/initial_spectrum2.json', 'gnpy/example-data/meshTopologyExampleV2.xls', '--show-channels', ]),
    ('path_requests_run_CD_PMD_PDL_missing', path_requests_run,
     ['tests/data/CORONET_Global_Topology_expected.json', 'tests/data/CORONET_services.json']),
))
def test_example_invocation(capfd, output, handler, args):
    """Make sure that our examples produce useful output"""
    os.chdir(SRC_ROOT)
    expected = open(SRC_ROOT / 'tests' / 'invocation' / output, mode='r', encoding='utf-8').read()
    handler(args)
    captured = capfd.readouterr()
    assert captured.out == expected
    assert captured.err == ''


@pytest.mark.parametrize('program', ('gnpy-transmission-example', 'gnpy-path-request'))
def test_run_wrapper(program):
    """Ensure that our wrappers really, really work"""
    proc = subprocess.run((program, '--help'), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, universal_newlines=True)
    assert proc.stderr == ''
    assert 'https://github.com/telecominfraproject/oopt-gnpy' in proc.stdout.lower()
    assert 'https://gnpy.readthedocs.io/' in proc.stdout.lower()


def test_conversion_xls():
    proc = subprocess.run(
        ('gnpy-convert-xls', SRC_ROOT / 'tests' / 'data' / 'testTopology.xls', '--output', os.path.devnull),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, universal_newlines=True)
    assert proc.stderr == ''
    assert os.path.devnull in proc.stdout
