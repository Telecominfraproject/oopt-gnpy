# -*- coding: utf-8 -*-

from pathlib import Path
import os
from logging import INFO, Formatter
import pytest
import subprocess
from gnpy.tools.cli_examples import transmission_main_example, path_requests_run

SRC_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("output, log, handler, args", (
    ('transmission_main_example', None, transmission_main_example, []),
    ('transmission_saturated', 'logs_transmission_saturated', transmission_main_example,
     ['tests/data/testTopology_expected.json', 'lannion', 'lorient', '-e', 'tests/data/eqpt_config.json', '--pow', '3']),
    ('path_requests_run', 'logs_path_request', path_requests_run, ['--redesign-per-request', '-v']),
    ('transmission_main_example__raman', None, transmission_main_example,
     ['gnpy/example-data/raman_edfa_example_network.json', '--sim', 'gnpy/example-data/sim_params.json', '--show-channels', ]),
    ('openroadm-v4-Stockholm-Gothenburg', None, transmission_main_example,
     ['gnpy/example-data/Sweden_OpenROADMv4_example_network.json', '-e', 'gnpy/example-data/eqpt_config_openroadm_ver4.json', ]),
    ('openroadm-v5-Stockholm-Gothenburg', None, transmission_main_example,
     ['gnpy/example-data/Sweden_OpenROADMv5_example_network.json', '-e', 'gnpy/example-data/eqpt_config_openroadm_ver5.json', ]),
    ('transmission_main_example_long', None, transmission_main_example,
     ['tests/data/test_long_network.json', '-e', 'tests/data/eqpt_config.json']),
    ('spectrum1_transmission_main_example', None, transmission_main_example,
     ['--spectrum', 'gnpy/example-data/initial_spectrum1.json', 'gnpy/example-data/meshTopologyExampleV2.xls', ]),
    ('spectrum2_transmission_main_example', None, transmission_main_example,
     ['--spectrum', 'gnpy/example-data/initial_spectrum2.json', 'gnpy/example-data/meshTopologyExampleV2.xls', '--show-channels', ]),
    ('path_requests_run_CD_PMD_PDL_missing', 'logs_path_requests_run_CD_PMD_PDL_missing', path_requests_run,
     ['tests/data/CORONET_Global_Topology_expected.json', 'tests/data/CORONET_services.json', '-v']),
    ('power_sweep_example', 'logs_power_sweep_example', transmission_main_example,
     ['tests/data/testTopology_expected.json', 'brest', 'rennes', '-e', 'tests/data/eqpt_config_sweep.json', '--pow', '3']),
    ('transmission_long_pow', None, transmission_main_example,
     ['tests/data/test_long_network.json', '-e', 'tests/data/eqpt_config.json', '--spectrum', 'gnpy/example-data/initial_spectrum2.json']),
    ('transmission_long_psd', None, transmission_main_example,
     ['tests/data/test_long_network.json', '-e', 'tests/data/eqpt_config_psd.json', '--spectrum', 'gnpy/example-data/initial_spectrum2.json', ]),
    ('transmission_long_psw', None, transmission_main_example,
     ['tests/data/test_long_network.json', '-e', 'tests/data/eqpt_config_psw.json', '--spectrum', 'gnpy/example-data/initial_spectrum2.json', ]),
    ('multiband_transmission', None, transmission_main_example,
     ['gnpy/example-data/multiband_example_network.json', 'Site_A', 'Site_D', '-e', 'gnpy/example-data/eqpt_config_multiband.json',
      '--spectrum', 'gnpy/example-data/multiband_spectrum.json', '--show-channels']),
    ('path_requests_run_extra_equipment', 'logs_path_requests_run_extra_equipment', path_requests_run,
     ['gnpy/example-data/meshTopologyExampleV2.xls', 'gnpy/example-data/service_pluggable.json', '--extra-equipment', 'gnpy/example-data/extra_eqpt_config.json', 'tests/data/extra_eqpt_config.json',
      '--extra-config', 'tests/data/user_edfa_config.json'])
))
def test_example_invocation(capfd, caplog, output, log, handler, args):
    """Make sure that our examples produce useful output"""
    os.chdir(SRC_ROOT)
    expected = open(SRC_ROOT / 'tests' / 'invocation' / output, mode='r', encoding='utf-8').read()
    formatter = Formatter('%(levelname)-9s%(name)s:%(filename)s %(message)s')
    caplog.handler.setFormatter(formatter)
    # keep INFO level to at least test those logs once
    caplog.set_level(INFO)
    handler(args)
    captured = capfd.readouterr()
    assert captured.out == expected
    assert captured.err == ''
    if log:
        expected_log = open(SRC_ROOT / 'tests' / 'invocation' / log, mode='r', encoding='utf-8').read()
        assert expected_log == caplog.text


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
    assert proc.stderr == 'missing header delta p\nmissing header delta p\n'
    assert os.path.devnull in proc.stdout
