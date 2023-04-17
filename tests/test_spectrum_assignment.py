#!/usr/bin/env python3
# Module name: test_spectrum_assignment.py
# Version:
# License: BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic

"""

from pathlib import Path
from copy import deepcopy
import json
from math import ceil
import pytest
from gnpy.core.network import build_network
from gnpy.core.utils import lin2db, automatic_nch
from gnpy.core.elements import Roadm, Transceiver
from gnpy.core.exceptions import ServiceError, SpectrumError
from gnpy.topology.request import compute_path_dsjctn, find_reversed_path, deduplicate_disjunctions, PathRequest
from gnpy.topology.spectrum_assignment import (build_oms_list, align_grids, nvalue_to_frequency,
                                           bitmap_sum, Bitmap, spectrum_selection, pth_assign_spectrum)
from gnpy.tools.json_io import (load_equipment, load_network, requests_from_json, disjunctions_from_json,
                                _check_one_request)

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
EQPT_FILENAME = DATA_DIR / 'eqpt_config.json'
NETWORK_FILENAME = DATA_DIR / 'testTopology_auto_design_expected.json'
SERVICE_FILENAME = DATA_DIR / 'testTopology_services_expected.json'

grid = 0.00625e12
slot = 0.0125e12
guardband = 0.15e12
cband_freq_min = 191.3e12
cband_freq_max = 196.1e12


@pytest.fixture()
def equipment():
    equipment = load_equipment(EQPT_FILENAME)
    return equipment


@pytest.fixture()
def setup(equipment):
    """common setup for tests: builds network, equipment and oms only once"""
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    oms_list = build_oms_list(network, equipment)
    return network, oms_list


def test_oms(setup):
    """tests that the OMS is between two ROADMs, that there is no ROADM or transceivers in the OMS
    except end points, checks that the id of OMS is present in the element and that the element
    OMS id is consistant
    """
    network, oms_list = setup
    for oms in oms_list:
        assert isinstance(oms.el_list[0], Roadm) and isinstance(oms.el_list[-1], Roadm)
        for i, elem in enumerate(oms.el_list[1:-1]):
            assert not isinstance(elem, Roadm)
            assert elem in network.nodes()
            assert elem.oms.oms_id == oms.oms_id
            assert elem.uid == oms.el_id_list[i + 1]


@pytest.mark.parametrize('nval', [0, 10, -255])
@pytest.mark.parametrize('mval', [1, 8])
def test_acceptable_values(nval, mval, setup):
    """Reasonable parameters should not throw"""
    network, oms_list = setup
    random_oms = oms_list[5]
    random_oms.assign_spectrum(nval, mval)


@pytest.mark.parametrize('nval,mval', (
    (0, 600),
    (-300, 1),
    (500, 1),
    (0, -2),
    (0, 4.5),
    (0.5, 8),
))
def test_wrong_values(nval, mval, setup):
    """Out of range or invalid values"""
    network, oms_list = setup
    random_oms = oms_list[5]
    with pytest.raises(SpectrumError):
        random_oms.assign_spectrum(nval, mval)


@pytest.mark.parametrize('nmin', [-288, -260, -300])
@pytest.mark.parametrize('nmax', [480, 320, 450])
def test_aligned(nmin, nmax, setup):
    """Checks that the OMS grid is correctly aligned

    Note that bitmap index uses guardband on both ends so that if nmin, nmax = -200, +200,
    min/max navalue in bitmap are -224, +223, which makes 223 -(-224) +1 frequencies.
    """
    network, oms_list = setup
    nguard = guardband / grid
    center = 193.1e12
    freq_min = center + nmin * grid
    freq_max = center + nmax * grid
    random_oms = oms_list[10]

    # We're always starting with full C-band
    assert pytest.approx(nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_min) * 1e-12, abs=1e-12) == 191.3
    assert pytest.approx(nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_max) * 1e-12, abs=1e-12) == 196.1
    ind_max = len(random_oms.spectrum_bitmap.bitmap) - 1

    # "inner" frequencies, without the guard baand
    inner_n_min = random_oms.spectrum_bitmap.getn(0) + nguard
    inner_n_max = random_oms.spectrum_bitmap.getn(ind_max) - nguard + 1
    assert inner_n_min == random_oms.spectrum_bitmap.freq_index_min
    assert inner_n_max == random_oms.spectrum_bitmap.freq_index_max
    assert inner_n_min == -288
    assert inner_n_max == 480

    # checks that changes are applied on bitmap and freq attributes of Bitmap object
    random_oms.update_spectrum(freq_min, freq_max, grid=grid, guardband=guardband)
    assert nmin == random_oms.spectrum_bitmap.freq_index_min
    assert nmax == random_oms.spectrum_bitmap.freq_index_max

    print('Adjusted spectrum: {:f} - {:f}'.format(
        nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_min) * 1e-12,
        nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_max) * 1e-12
    ))

    ind_max = len(random_oms.spectrum_bitmap.bitmap) - 1
    inner_n_min = random_oms.spectrum_bitmap.getn(0) + nguard
    inner_n_max = random_oms.spectrum_bitmap.getn(ind_max) - nguard + 1
    assert inner_n_min == random_oms.spectrum_bitmap.freq_index_min
    assert inner_n_max == random_oms.spectrum_bitmap.freq_index_max

    oms_list = align_grids(oms_list)
    ind_max = len(random_oms.spectrum_bitmap.bitmap) - 1
    nvalmin = random_oms.spectrum_bitmap.getn(0)
    nvalmax = random_oms.spectrum_bitmap.getn(ind_max)
    assert nvalmin <= nmin and nvalmax >= nmax


@pytest.mark.parametrize('nval1', [0, 15, 24])
@pytest.mark.parametrize('nval2', [8, 12])
def test_assign_and_sum(nval1, nval2, setup):
    """checks that bitmap sum gives correct result"""
    network, oms_list = setup
    guardband = grid
    mval = 4  # slot in 12.5GHz
    freq_max = cband_freq_min + 24 * grid
    # arbitrary test on oms #10 and #11
    # first reduce the grid to 24 center frequencies to ease reading when test fails
    oms1 = oms_list[10]
    oms1.update_spectrum(cband_freq_min, freq_max, grid=grid, guardband=guardband)
    oms2 = oms_list[11]
    oms2.update_spectrum(cband_freq_min, freq_max, grid=grid, guardband=guardband)
    print('initial spectrum')
    print(nvalue_to_frequency(oms1.spectrum_bitmap.freq_index_min) * 1e-12,
          nvalue_to_frequency(oms1.spectrum_bitmap.freq_index_max) * 1e-12)
    # checks initial values consistancy
    ind_max = len(oms1.spectrum_bitmap.bitmap) - 1
    print('with guardband', oms1.spectrum_bitmap.getn(0),
          oms1.spectrum_bitmap.getn(ind_max))
    print('without guardband', oms1.spectrum_bitmap.freq_index_min,
          oms1.spectrum_bitmap.freq_index_max)
    # if requested slots exceed grid spectrum should not be assigned and assignment
    # should return False
    if ((nval1 - mval) < oms1.spectrum_bitmap.getn(0) or
            (nval1 + mval - 1) > oms1.spectrum_bitmap.getn(ind_max)):
        with pytest.raises(SpectrumError):
            oms1.assign_spectrum(nval1, mval)
        for elem in oms1.spectrum_bitmap.bitmap:
            assert elem == 1
    else:
        oms2.assign_spectrum(nval2, mval)
        print(oms2.spectrum_bitmap.bitmap)
        test2 = bitmap_sum(oms1.spectrum_bitmap.bitmap, oms2.spectrum_bitmap.bitmap)
        print(test2)
        range1 = range(oms1.spectrum_bitmap.geti(nval1) - mval,
                       oms1.spectrum_bitmap.geti(nval1) + mval - 1)
        range2 = range(oms2.spectrum_bitmap.geti(nval2) - mval,
                       oms2.spectrum_bitmap.geti(nval2) + mval - 1)
        for elem in range1:
            print(f'value should be zero at index {elem}')
            assert test2[elem] == 0

        for elem in range2:
            print(f'value should be zero at index {elem}')
            assert test2[elem] == 0


def test_bitmap_assignment(setup):
    """test that a bitmap can be assigned"""
    network, oms_list = setup
    random_oms = oms_list[2]
    random_oms.assign_spectrum(13, 7)

    btmp = deepcopy(random_oms.spectrum_bitmap.bitmap)
    # try a first assignment that must pass
    spectrum_btmp = Bitmap(cband_freq_min, cband_freq_max, grid=0.00625e12, guardband=0.15e12, bitmap=btmp)

    # try a wrong assignment that should not pass
    btmp = btmp[1:-1]
    with pytest.raises(SpectrumError):
        spectrum_btmp = Bitmap(cband_freq_min, cband_freq_max, grid=0.00625e12, guardband=0.15e12, bitmap=btmp)


@pytest.fixture()
def services(equipment):
    """common setup for service list: builds service only once"""
    with open(SERVICE_FILENAME, encoding='utf-8') as my_f:
        services = json.loads(my_f.read())
    return services


@pytest.fixture()
def requests(equipment, services):
    """common setup for requests, builds requests list only once"""
    requests = requests_from_json(services, equipment)
    return requests


def test_spectrum_assignment_on_path(equipment, setup, requests):
    """test assignment functions on path and network"""
    network, oms_list = setup
    req = [deepcopy(requests[1])]
    paths = compute_path_dsjctn(network, equipment, req, [])

    print(req)
    for nval in range(100):
        req = [deepcopy(requests[1])]
        (center_n, startn, stopn), path_oms = spectrum_selection(paths[0], oms_list, 4)
        pth_assign_spectrum(paths, req, oms_list, [find_reversed_path(paths[0])])
        print(f'testing on following oms {path_oms}')
        # check that only 96 channels are feasible
        if nval >= 96:
            print(center_n, startn, stopn)
            print('only 96 channels of 4 slots pass in this grid')
            assert center_n is None and startn is None and stopn is None
        else:
            print(center_n, startn, stopn)
            print('at least 96 channels of 4 slots should pass in this grid')
            assert center_n is not None and startn is not None and stopn is not None

    req = [requests[2]]
    paths = compute_path_dsjctn(network, equipment, req, [])
    (center_n, startn, stopn), path_oms = spectrum_selection(paths[0], oms_list, 4, 478)
    print(oms_list[0].spectrum_bitmap.freq_index_max)
    print(oms_list[0])
    print(center_n, startn, stopn)
    print('spectrum selection error: should be None')
    assert center_n is None and startn is None and stopn is None
    (center_n, startn, stopn), path_oms = spectrum_selection(paths[0], oms_list, 4, 477)
    print(center_n, startn, stopn)
    print('spectrum selection error should not be None')
    assert center_n is not None and startn is not None and stopn is not None


@pytest.fixture()
def request_set():
    """creates default request dict"""
    return {
        'request_id': '0',
        'source': 'trx a',
        'bidir': False,
        'destination': 'trx g',
        'trx_type': 'Voyager',
        'trx_mode': 'mode 1',
        'format': 'mode1',
        'spacing': 50e9,
        'nodes_list': [],
        'loose_list': [],
        'f_min': 191.1e12,
        'f_max': 196.3e12,
        'baud_rate': 32e9,
        'OSNR': 14,
        'bit_rate': 100e9,
        'cost': 1,
        'roll_off': 0.15,
        'tx_osnr': 38,
        'penalties': {},
        'min_spacing': 37.5e9,
        'nb_channel': None,
        'power': 0,
        'path_bandwidth': 800e9}


def test_freq_slot_exist(setup, equipment, request_set):
    """test that assignment works even if effective_freq_slot is not populated"""
    network, oms_list = setup
    params = request_set
    params['effective_freq_slot'] = None
    rqs = [PathRequest(**params)]
    paths = compute_path_dsjctn(network, equipment, rqs, [])
    pth_assign_spectrum(paths, rqs, oms_list, [find_reversed_path(paths[0])])
    assert rqs[0].N == -256
    assert rqs[0].M == 32


def test_inconsistant_freq_slot(setup, equipment, request_set):
    """test that an inconsistant M correctly raises an error"""
    network, oms_list = setup
    params = request_set
    # minimum required nb of slots is 32 (800Gbit/100Gbit/s channels each occupying 50GHz ie 4 slots)
    params['effective_freq_slot'] = {'N': 0, 'M': 4}
    with pytest.raises(ServiceError):
        _check_one_request(params, 196.05e12)
    params['trx_mode'] = None
    rqs = [PathRequest(**params)]
    paths = compute_path_dsjctn(network, equipment, rqs, [])
    pth_assign_spectrum(paths, rqs, oms_list, [find_reversed_path(paths[0])])
    assert rqs[0].blocking_reason == 'NOT_ENOUGH_RESERVED_SPECTRUM'


@pytest.mark.parametrize('n, m, final_n, final_m, blocking_reason', [
    # regular requests that should be correctly assigned:
    (-100, 32, -100, 32, None),
    (150, 50, 150, 50, None),
    # if n is None, there should be an assignment (enough spectrum cases)
    # and the center frequency should be set on the lower part of the spectrum based on m value if it exists
    # or based on 32
    (None, 32, -256, 32, None),
    (None, 40, -248, 40, None),
    (-100, None, -100, 32, None),
    (None, None, -256, 32, None),
    # -280 and 60 center indexes should result in unfeasible spectrum, either out of band or
    # overlapping with occupied spectrum. The requested spectrum is not available
    (-280, None, None, None, 'NO_SPECTRUM'),
    (-60, 40, None, None, 'NO_SPECTRUM'),
    # 20 is smaller than min 32 required nb of slots so should also be blocked
    (-60, 20, None, None, 'NOT_ENOUGH_RESERVED_SPECTRUM')
    ])
def test_n_m_requests(setup, equipment, n, m, final_n, final_m, blocking_reason, request_set):
    """test that various N and M values for a request end up with the correct path assgnment"""
    network, oms_list = setup
    # add an occupation on one of the span of the expected path OMS list on both directions
    # as defined by its offsets within the OMS list: [17, 20, 13, 22] and reversed path [19, 16, 21, 26]
    expected_path = [17, 20, 13, 22]
    expected_oms = [13, 16, 17, 19, 20, 21, 22, 26]
    some_oms = oms_list[expected_oms[3]]
    some_oms.assign_spectrum(-30, 32)    # means that spectrum is occupied from indexes -62 to 1 on reversed path
    params = request_set
    params['effective_freq_slot'] = {'N': n, 'M': m}
    rqs = [PathRequest(**params)]

    paths = compute_path_dsjctn(network, equipment, rqs, [])
    # check that the computed path is the expected one (independant of blocking issues due to spectrum)
    path_oms = list(set([e.oms_id for e in paths[0] if not isinstance(e, (Transceiver, Roadm))]))
    assert path_oms == expected_path
    # function to be tested:
    pth_assign_spectrum(paths, rqs, oms_list, [find_reversed_path(paths[0])])
    # check that spectrum is correctly assigned
    assert rqs[0].N == final_n
    assert rqs[0].M == final_m
    assert getattr(rqs[0], 'blocking_reason', None) == blocking_reason


def test_reversed_direction(equipment, setup, requests, services):
    """checks that if spectrum is selected on one direction it is also selected on reversed direction"""
    network, oms_list = setup
    dsjn = disjunctions_from_json(services)
    dsjn = deduplicate_disjunctions(dsjn)
    paths = compute_path_dsjctn(network, equipment, requests, dsjn)
    rev_pths = []
    for pth in paths:
        if pth:
            rev_pths.append(find_reversed_path(pth))
        else:
            rev_pths.append([])
    # build the list of spectrum slots that will be used for each request. For this purpose
    # play the selection part of path_assign_spectrum
    spectrum_list = []
    for i, pth in enumerate(paths):
        if pth:
            number_wl = ceil(requests[i].path_bandwidth / requests[i].bit_rate)
            requested_m = ceil(requests[i].spacing / slot) * number_wl
            (center_n, startn, stopn), path_oms = spectrum_selection(pth, oms_list, requested_m,
                                                                     requested_n=None)
            spectrum_list.append([center_n, startn, stopn])
        else:
            spectrum_list.append([])
    pth_assign_spectrum(paths, requests, oms_list, rev_pths)
    # pth-assign concatenates path and reversed path
    for i, pth in enumerate(paths):
        # verifies that each element (not trx and not roadm) in the path has same
        # spectrum occupation
        if pth:
            this_path = [elem for elem in pth if not isinstance(elem, Roadm) and
                         not isinstance(elem, Transceiver)]
            print(f'path {[el.uid for el in this_path]}')
            this_revpath = [elem for elem in rev_pths[i] if not isinstance(elem, Roadm) and
                            not isinstance(elem, Transceiver)]
            print(f'rev_path {[el.uid for el in this_revpath]}')
            print('')
            for j, elem in enumerate(this_revpath):
                imin = elem.oms.spectrum_bitmap.geti(spectrum_list[i][1])
                imax = elem.oms.spectrum_bitmap.geti(spectrum_list[i][2])
                print(f'rev_elem {elem.uid}')
                print(f'    elem {this_path[len(this_path)-j-1].uid}')
                print(f'\trev_spectrum: {elem.oms.spectrum_bitmap.bitmap[imin:imax]}')
                print(f'\t    spectrum: ' +
                      f'{this_path[len(this_path)-j-1].oms.spectrum_bitmap.bitmap[imin:imax]}')
                assert elem.oms.spectrum_bitmap.bitmap[imin:imax] == \
                    this_path[len(this_path) - j - 1].oms.spectrum_bitmap.bitmap[imin:imax]
