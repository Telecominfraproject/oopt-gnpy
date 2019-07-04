#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
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
from gnpy.core.equipment import load_equipment, automatic_nch
from gnpy.core.network import load_network, build_network
from gnpy.core.utils import lin2db
from gnpy.core.elements import Roadm, Transceiver
from gnpy.core.spectrum_assignment import (build_oms_list, align_grids, nvalue_to_frequency,
                                           bitmap_sum, m_to_freq, slots_to_m, frequency_to_n,
                                           Bitmap, spectrum_selection, pth_assign_spectrum)
from gnpy.core.exceptions import SpectrumError
from gnpy.core.request import compute_path_dsjctn, find_reversed_path
from examples.path_requests_run import (requests_from_json, disjunctions_from_json,
                                        correct_disjn, compute_path_with_disjunction)

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
    """ common setup for tests: builds network, equipment and oms only once
    """
    network = load_network(NETWORK_FILENAME, equipment)
    spectrum = equipment['SI']['default']
    p_db = spectrum.power_dbm
    p_total_db = p_db + lin2db(automatic_nch(spectrum.f_min, spectrum.f_max, spectrum.spacing))
    build_network(network, equipment, p_db, p_total_db)
    oms_list = build_oms_list(network, equipment)
    return network, oms_list

def test_oms(setup):
    """ tests that the OMS is between two ROADMs, that there is no ROADM or transceivers in the OMS
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
            assert elem.uid == oms.el_id_list[i+1]

@pytest.mark.parametrize('nmin', [-288, -260, -300])
@pytest.mark.parametrize('nmax', [480, 320, 450])
def test_aligned(nmin, nmax, setup):
    """ checks that the oms grid is correctly aligned. Note that bitmap index uses guardband
        on both ends so that if nmin, nmax = -200, +200, min/max navalue in bitmap are
        -224, +223, which makes 223 -(-224) +1 frequencies
    """
    network, oms_list = setup
    nguard = guardband / grid
    center = 193.1e12
    freq_min = center + nmin * grid
    freq_max = center + nmax * grid
    random_oms = oms_list[10]
    print('initial spectrum')
    assert pytest.approx(nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_min) * 1e-12, abs=1e-12) == 191.3
    assert pytest.approx(nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_max) * 1e-12, abs=1e-12) == 196.1
    # checks initial values consistancy
    ind_max = len(random_oms.spectrum_bitmap.bitmap) - 1
    print('with guardband', random_oms.spectrum_bitmap.getn(0),
          random_oms.spectrum_bitmap.getn(ind_max))
    print('without guardband', random_oms.spectrum_bitmap.freq_index_min,
          random_oms.spectrum_bitmap.freq_index_max)
    nvalmin = random_oms.spectrum_bitmap.getn(0) + nguard
    nvalmax = random_oms.spectrum_bitmap.getn(ind_max) - nguard + 1

    # min index in bitmap must be consistant with min freq attribute
    print(f'test1 expected: {nvalmin}')
    print(f'freq_index_min: {random_oms.spectrum_bitmap.freq_index_min},')
    assert nvalmin == random_oms.spectrum_bitmap.freq_index_min
    print(f'test2 expected: {nvalmax}')
    print(f'freq_index_max: {random_oms.spectrum_bitmap.freq_index_max}')
    assert nvalmax == random_oms.spectrum_bitmap.freq_index_max
    random_oms.update_spectrum(freq_min, freq_max, grid=grid, guardband=guardband)
    # checks that changes are applied on bitmap and freq attributes of Bitmap object
    print(f'expected: {nmin}, {nmax}')
    print(f'test3 obtained: {random_oms.spectrum_bitmap.freq_index_min},' +\
          f' {random_oms.spectrum_bitmap.freq_index_max}')
    assert (nmin == random_oms.spectrum_bitmap.freq_index_min and
            nmax == random_oms.spectrum_bitmap.freq_index_max)

    print('novel spectrum')
    print(nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_min) * 1e-12,
          nvalue_to_frequency(random_oms.spectrum_bitmap.freq_index_max) * 1e-12)
    ind_max = len(random_oms.spectrum_bitmap.bitmap) - 1
    print('with guardband', random_oms.spectrum_bitmap.getn(0),
          random_oms.spectrum_bitmap.getn(ind_max))
    print('without guardband', random_oms.spectrum_bitmap.freq_index_min,
          random_oms.spectrum_bitmap.freq_index_max)
    nvalmin = random_oms.spectrum_bitmap.getn(0) + nguard
    nvalmax = random_oms.spectrum_bitmap.getn(ind_max) - nguard + 1
    # min index in bitmap must be consistant with min freq attribute
    print(f'expected: {nvalmin}')
    print(f'freq_index_min: {random_oms.spectrum_bitmap.freq_index_min},')
    print('inconsistancy in Bitmap object')
    assert nvalmin == random_oms.spectrum_bitmap.freq_index_min
    print(f'expected: {nvalmax}')
    print(f'freq_index_max: {random_oms.spectrum_bitmap.freq_index_max}')
    print('inconsistancy in Bitmap object')
    assert nvalmax == random_oms.spectrum_bitmap.freq_index_max
    oms_list = align_grids(oms_list)
    ind_max = len(random_oms.spectrum_bitmap.bitmap) - 1
    nvalmin = random_oms.spectrum_bitmap.getn(0)
    nvalmax = random_oms.spectrum_bitmap.getn(ind_max)
    print(f'expected: {min(nmin, nvalmin)}, {max(nmax, nvalmax)}')
    print(f'expected: {nmin, nmax}')
    print(f'obtained after alignment: {nvalmin}, {nvalmax}')
    assert nvalmin <= nmin and nvalmax >= nmax

@pytest.mark.parametrize('nval1', [0, 15, 24])
@pytest.mark.parametrize('nval2', [8, 12])
def test_assign_and_sum(nval1, nval2, setup):
    """ checks that bitmap sum gives correct result
    """
    network, oms_list = setup
    guardband = grid
    mval = 4 # slot in 12.5GHz
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
    test1 = oms1.assign_spectrum(nval1, mval)
    print(oms1.spectrum_bitmap.bitmap)
    # if requested slots exceed grid spectrum should not be assigned and assignment
    # should return False
    if ((nval1 - mval) < oms1.spectrum_bitmap.getn(0) or
            (nval1 + mval-1) > oms1.spectrum_bitmap.getn(ind_max)):
        print('assignment on part of bitmap is not allowed')
        assert not test1
        for elem in oms1.spectrum_bitmap.bitmap:
            assert elem == 1
    else:
        oms2.assign_spectrum(nval2, mval)
        print(oms2.spectrum_bitmap.bitmap)
        test2 = bitmap_sum(oms1.spectrum_bitmap.bitmap, oms2.spectrum_bitmap.bitmap)
        print(test2)
        range1 = range(oms1.spectrum_bitmap.geti(nval1) - mval,
                       oms1.spectrum_bitmap.geti(nval1) + mval -1)
        range2 = range(oms2.spectrum_bitmap.geti(nval2) - mval,
                       oms2.spectrum_bitmap.geti(nval2) + mval -1)
        for elem in range1:
            print(f'value should be zero at index {elem}')
            assert test2[elem] == 0

        for elem in range2:
            print(f'value should be zero at index {elem}')
            assert test2[elem] == 0

def test_values(setup):
    """ checks that oms.assign_spectrum(13,7) is (193137500000000.0, 193225000000000.0)
        reference to Recommendation G.694.1 (02/12), Figure I.3
        https://www.itu.int/rec/T-REC-G.694.1-201202-I/en
    """
    network, oms_list = setup
    random_oms = oms_list[17]
    random_oms.assign_spectrum(13, 7)
    fstart, fstop = m_to_freq(13, 7)
    print('expected: 193137500000000.0, 193225000000000.0')
    print(f'obtained: {fstart}, {fstop}')
    assert fstart == 193.1375e12 and fstop == 193.225e12
    nstart = frequency_to_n(fstart)
    nstop = frequency_to_n(fstop)

    nval, mval = slots_to_m(nstart, nstop)
    print('expected n, m: 13, 7')
    print(f'obtained: {nval}, {mval}')
    assert nval == 13 or mval == 7

@pytest.mark.parametrize('nval', [0, 10, -255])
@pytest.mark.parametrize('mval', [1, 8])
def test_acceptable_values(nval, mval, setup):
    """ test n or m not applicable values
    """
    network, oms_list = setup
    random_oms = oms_list[5]
    random_oms.assign_spectrum(nval, mval)
    print(f'n, m values should not raise an error {nval}, {mval}')

@pytest.mark.parametrize('nval', [0, 10, -10, None, 0.5, 9.001])
@pytest.mark.parametrize('mval', [0, 4.5, None, -2])
def test_exception(nval, mval, setup):
    """ test n or m not applicable values
    """
    network, oms_list = setup
    random_oms = oms_list[5]
    with pytest.raises(SpectrumError):
        random_oms.assign_spectrum(nval, mval)
        print(f'n, m values should raise an error {nval}, {mval}')

@pytest.mark.parametrize('nval', [0, -300, 500])
@pytest.mark.parametrize('mval', [1, 600])
def test_wrong_values(nval, mval, setup):
    """ test n or m not applicable values
    """
    network, oms_list = setup
    random_oms = oms_list[8]
    test = random_oms.assign_spectrum(nval, mval)
    if nval != 0 or mval != 1:
        print(f'n, m values should raise an error (exceed bitmap rane, so None assignment) {nval}, {mval}')
        assert not test
    else:
        print(f'Except for the the couple 0, 1')
        assert test

def test_bitmap_assignment(setup):
    """ test that a bitmap can be assigned
    """
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
    print('bitmap direct assignment should create an error if length is not consistant with provided values')

@pytest.fixture()
def services(equipment):
    """ common setup for service list: builds service only once
    """
    with open(SERVICE_FILENAME, encoding='utf-8') as my_f:
        services = json.loads(my_f.read())
    return services

@pytest.fixture()
def requests(equipment, services):
    """ common setup for requests, builds requests list only once
    """
    requests = requests_from_json(services, equipment)
    return requests

def test_spectrum_assignment_on_path(equipment, setup, requests):
    """ test assignment functions on path and network
    """
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

def test_reversed_direction(equipment, setup, requests, services):
    """ checks that if spectrum is selected on one direction it is also selected on reversed
        direction
    """
    network, oms_list = setup
    dsjn = disjunctions_from_json(services)
    dsjn = correct_disjn(dsjn)
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
            this_path = [elem for elem in pth if not isinstance(elem, Roadm) and\
                         not isinstance(elem, Transceiver)]
            print(f'path {[el.uid for el in this_path]}')
            this_revpath = [elem for elem in rev_pths[i] if not isinstance(elem, Roadm) and\
                         not isinstance(elem, Transceiver)]
            print(f'rev_path {[el.uid for el in this_revpath]}')
            print('')
            for j, elem in enumerate(this_revpath):
                imin = elem.oms.spectrum_bitmap.geti(spectrum_list[i][1])
                imax = elem.oms.spectrum_bitmap.geti(spectrum_list[i][2])
                print(f'rev_elem {elem.uid}')
                print(f'    elem {this_path[len(this_path)-j-1].uid}')
                print(f'\trev_spectrum: {elem.oms.spectrum_bitmap.bitmap[imin:imax]}')
                print(f'\t    spectrum: ' +\
                      f'{this_path[len(this_path)-j-1].oms.spectrum_bitmap.bitmap[imin:imax]}')
                assert elem.oms.spectrum_bitmap.bitmap[imin:imax] == \
                   this_path[len(this_path)-j-1].oms.spectrum_bitmap.bitmap[imin:imax]
