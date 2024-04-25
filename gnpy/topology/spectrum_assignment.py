#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.topology.spectrum_assignment: spectrum assignment functionality
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
gnpy.topology.spectrum_assignment
=================================

This module contains the :class:`Oms` and :class:`Bitmap` classes and methods to
select and assign spectrum. The :func:`spectrum_selection` function identifies the free
slots and :func:`select_candidate` selects the candidate spectrum according to
strategy: for example first fit
oms records its elements, and elements are updated with an oms to have
element/oms correspondace
"""

from collections import namedtuple
from logging import getLogger
from typing import List, Union, Optional, Tuple
from enum import Enum
from networkx import DiGraph

from gnpy.core.elements import Roadm, Transceiver, Edfa, Multiband_amplifier, Fiber, RamanFiber, Fused
from gnpy.core.exceptions import ServiceError, SpectrumError
from gnpy.core.utils import order_slots, restore_order
from gnpy.topology.request import compute_spectrum_slot_vs_bandwidth, find_elements_common_range, PathRequest

LOGGER = getLogger(__name__)
DEFAULT_GUARDBAND = 0.025e12
DEFAULT_GRID = 0.00625e12
FIRST_FIT = 'first_fit'
LAST_FIT = 'last_fit'


class BitmapValue(Enum):
    """Bitmap allowed values
    """
    UNUSABLE = 'u'
    OCCUPIED = 0
    FREE = 1

    def __str__(self):
        if self.value == self.OCCUPIED:
            return '0'
        if self.value == self.FREE:
            return '1'
        if self.value == self.UNUSABLE:
            return 'u'
        return ''


class Bitmap:
    """records the spectrum occupation"""

    def __init__(self, f_min, f_max, grid, guardband=DEFAULT_GUARDBAND, bitmap=None):
        # n is the min index including guardband. Guardband is required to be sure
        # that a channel can be assigned  with center frequency fmin (means that its
        # slot occupation goes below freq_index_min
        n_min = frequency_to_n(f_min, grid)
        n_max = frequency_to_n(f_max, grid)
        self.n_min = n_min
        self.n_max = n_max
        self.freq_index_min = frequency_to_n(f_min + guardband)
        self.freq_index_max = frequency_to_n(f_max - guardband)
        self.freq_index = list(range(n_min, n_max + 1))
        self.guardband = guardband
        if bitmap is None:
            self.bitmap = [BitmapValue.FREE] * (n_max - n_min + 1)
        elif len(bitmap) == len(self.freq_index):
            self.bitmap = bitmap
        else:
            raise SpectrumError(f'bitmap is not consistant with f_min{f_min} - n: {n_min} and f_max{f_max} - n:{n_max}')

    def getn(self, i):
        """converts the n (itu grid) into a local index"""
        return self.freq_index[i]

    def geti(self, nvalue):
        """converts the local index into n (itu grid)"""
        return self.freq_index.index(nvalue)

    def insert_left(self, newbitmap):
        """insert bitmap on the left to align oms bitmaps if their start frequencies are different"""
        self.bitmap = newbitmap + self.bitmap
        temp = list(range(self.n_min - len(newbitmap), self.n_min))
        self.freq_index = temp + self.freq_index
        self.n_min = self.freq_index[0]

    def insert_right(self, newbitmap):
        """insert bitmap on the right to align oms bitmaps if their stop frequencies are different"""
        self.bitmap = self.bitmap + newbitmap
        self.freq_index = self.freq_index + list(range(self.n_max, self.n_max + len(newbitmap)))
        self.n_max = self.freq_index[-1]


#    +'grid available_slots f_min f_max services_list')
OMSParams = namedtuple('OMSParams', 'oms_id el_id_list el_list')


class OMS:
    """OMS class is the logical container that represent a link between two adjacent ROADMs and
    records the crossed elements and the occupied spectrum
    """

    def __init__(self, **params):
        params = OMSParams(**params)
        self.oms_id = params.oms_id
        self.el_id_list = params.el_id_list
        self.el_list = params.el_list
        self.spectrum_bitmap = []
        self.nb_channels = 0
        self.service_list = []

    def __str__(self):
        return '\n\t'.join([f'{type(self).__name__} {self.oms_id}',
                            f'{self.el_id_list[0]} - {self.el_id_list[-1]}'])

    def __repr__(self):
        return '\n\t'.join([f'{type(self).__name__} {self.oms_id}',
                            f'{self.el_id_list[0]} - {self.el_id_list[-1]}', '\n'])

    def add_element(self, elem: Union[Roadm, Transceiver, Edfa, Multiband_amplifier,
                                      Fiber, RamanFiber, Fused]):
        """records oms elements"""
        self.el_id_list.append(elem.uid)
        self.el_list.append(elem)

    def update_spectrum(self, f_min: float, f_max: float, guardband=DEFAULT_GUARDBAND,
                        existing_spectrum: Optional[List[Union[int, str]]] = None, grid: float = DEFAULT_GRID):
        """Updates spectrum with an existing bitmap. Frequencies expressed in Hz.

        Use ITU-T G694.1 Flexible DWDM grid definition
        For the flexible DWDM grid, the allowed frequency slots have a nominal central frequency (in THz) defined by:
        193.1 + n × 0.00625 where n is a positive or negative integer including 0
        and 0.00625 is the nominal central frequency granularity in THz
        and a slot width defined by:
        12.5 × m where m is a positive integer and 12.5 is the slot width granularity in GHz.
        Any combination of frequency slots is allowed as long as no two frequency slots overlap.
        If bitmap is not None, then use it: Bitmap checks its consistency with f_min f_max
        else a brand new bitmap is created
        """
        self.spectrum_bitmap = Bitmap(f_min=f_min, f_max=f_max, grid=grid, guardband=guardband,
                                      bitmap=existing_spectrum)

    def assign_spectrum(self, nvalue: int, mvalue: int):
        """change oms spectrum to mark spectrum assigned"""
        if not isinstance(nvalue, int):
            raise SpectrumError(f'N must be a signed integer, got {nvalue}')
        if not isinstance(mvalue, int):
            raise SpectrumError(f'M must be an integer, got {mvalue}')
        if mvalue <= 0:
            raise SpectrumError(f'M must be positive, got {mvalue}')
        if nvalue > self.spectrum_bitmap.freq_index_max:
            raise SpectrumError(f'N {nvalue} over the upper spectrum boundary')
        if nvalue < self.spectrum_bitmap.freq_index_min:
            raise SpectrumError(f'N {nvalue} below the lower spectrum boundary')
        startn, stopn = mvalue_to_slots(nvalue, mvalue)
        if stopn > self.spectrum_bitmap.n_max:
            raise SpectrumError(f'N {nvalue}, M {mvalue} over the N spectrum bitmap bounds')
        if startn <= self.spectrum_bitmap.n_min:
            raise SpectrumError(f'N {nvalue}, M {mvalue} below the N spectrum bitmap bounds')
        self.spectrum_bitmap.bitmap[self.spectrum_bitmap.geti(startn):self.spectrum_bitmap.geti(stopn) + 1] = \
            [BitmapValue.OCCUPIED] * (stopn - startn + 1)

    def add_service(self, service_id: str, nb_wl: int):
        """record service and mark spectrum as occupied"""
        self.service_list.append(service_id)
        self.nb_channels += nb_wl


def frequency_to_n(freq: float, grid: float = DEFAULT_GRID) -> int:
    """converts frequency into the n value (ITU grid)

    reference to Recommendation G.694.1 (02/12), Figure I.3
    https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> frequency_to_n(193.1375e12)
    6
    >>> frequency_to_n(193.225e12)
    20

    """
    return (int)((freq - 193.1e12) / grid)


def nvalue_to_frequency(nvalue: int, grid: float = DEFAULT_GRID) -> float:
    """converts n value into a frequency

    reference to Recommendation G.694.1 (02/12), Table 1
    https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> nvalue_to_frequency(6)
    193137500000000.0
    >>> nvalue_to_frequency(-1, 0.1e12)
    193000000000000.0

    """
    return 193.1e12 + nvalue * grid


def mvalue_to_slots(nvalue: int, mvalue: int) -> Tuple[int, int]:
    """convert center n an m into start and stop n"""
    startn = nvalue - mvalue
    stopn = nvalue + mvalue - 1
    return startn, stopn


def slots_to_m(startn: int, stopn: int) -> Tuple[int, int]:
    """converts the start and stop n values to the center n and m value

    reference to Recommendation G.694.1 (02/12), Figure I.3
    https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> nval, mval = slots_to_m(6, 20)
    >>> nval
    13
    >>> mval
    7

    """
    nvalue = (int)((startn + stopn + 1) / 2)
    mvalue = (int)((stopn - startn + 1) / 2)
    return nvalue, mvalue


def m_to_freq(nvalue: int, mvalue: int, grid: float = DEFAULT_GRID) -> Tuple[float, float]:
    """converts m into frequency range

    spectrum(13,7) is (193137500000000.0, 193225000000000.0)
    reference to Recommendation G.694.1 (02/12), Figure I.3
    https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> fstart, fstop = m_to_freq(13, 7)
    >>> fstart
    193137500000000.0
    >>> fstop
    193225000000000.0

    """
    startn, stopn = mvalue_to_slots(nvalue, mvalue)
    fstart = nvalue_to_frequency(startn, grid)
    fstop = nvalue_to_frequency(stopn + 1, grid)
    return fstart, fstop


def align_grids(oms_list: List[OMS]) -> List[OMS]:
    """Used to apply same grid to all oms : same starting n, stop n and slot size.Out of grid slots are set to 0."""
    n_min = min(o.spectrum_bitmap.n_min for o in oms_list)
    n_max = max(o.spectrum_bitmap.n_max for o in oms_list)
    for this_o in oms_list:
        if (this_o.spectrum_bitmap.n_min - n_min) > 0:
            this_o.spectrum_bitmap.insert_left([BitmapValue.OCCUPIED] * (this_o.spectrum_bitmap.n_min - n_min))
        if (n_max - this_o.spectrum_bitmap.n_max) > 0:
            this_o.spectrum_bitmap.insert_right([BitmapValue.OCCUPIED] * (n_max - this_o.spectrum_bitmap.n_max))
    return oms_list


def find_network_freq_range(network: DiGraph) -> Tuple[float, float]:
    """Find the lowest freq from amps and highest freq among all amps to determine the resulting bitmap
    """
    amp_bands = [band for n in network.nodes() if isinstance(n, (Edfa, Multiband_amplifier)) for band in n.params.bands]
    min_frequencies = [a['f_min'] for a in amp_bands]
    max_frequencies = [a['f_max'] for a in amp_bands]
    return min(min_frequencies), max(max_frequencies)


def create_oms_bitmap(oms: OMS, equipment: dict, f_min: float, f_max: float,
                      grid: float) -> List[BitmapValue]:
    """Create the bitmap window corresponding to the common frequency range on the OMS.
    f_min is the min central frequency but if the channel is 50GHz witdth it occupies 4 slots around this
    central frequency. We took the convention that central frequency is relative index 0
    -4 -3 -2 -1  0  1  2  3
    ___________ ^__________
    so that the channel occupies index(f_min) - 4 to index(f_min) + 3 on the occupation bitmap

    :param oms: oms  object containing the list of elements with their own spectral occupation
    :param equipment: equipment library
    :param f_min, f_max: central frequency range in Hz
    """
    n_min = frequency_to_n(f_min, grid)
    n_max = frequency_to_n(f_max, grid) - 1
    common_range = find_elements_common_range(oms.el_list, equipment)
    band0 = common_range[0]
    band0_n_min = frequency_to_n(band0['f_min'], grid)
    band0_n_max = frequency_to_n(band0['f_max'], grid)
    bitmap = [BitmapValue.UNUSABLE] * (band0_n_min - n_min) + [BitmapValue.FREE] * (band0_n_max - band0_n_min + 1)
    i = 1
    while i < len(common_range):
        band = common_range[i]
        band_n_min = frequency_to_n(band['f_min'], grid)
        band_n_max = frequency_to_n(band['f_max'], grid)
        bitmap = bitmap \
            + [BitmapValue.UNUSABLE] * (band_n_min - band0_n_max - 1) \
            + [BitmapValue.FREE] * (band_n_max - band_n_min + 1)
        band0_n_max = band_n_max
        i += 1
    bitmap = bitmap + [BitmapValue.UNUSABLE] * (n_max - band0_n_max)
    return bitmap


def build_oms_list(network: DiGraph, equipment: dict) -> List[OMS]:
    """initialization of OMS list in the network

    an oms is build reading all intermediate nodes between two adjacent ROADMs
    each element within the list is being added an oms and oms_id to record the
    oms it belongs to.
    the function supports different spectrum width and supposes that the whole network
    works with the min range among OMSs
    """
    oms_id = 0
    oms_list = []
    # identify all vertices of OMS: of course ROADM, but aso links to external chassis transponders
    oms_vertices = [n for n in network.nodes() if isinstance(n, Roadm)] +\
                   [n for n in network.nodes() if isinstance(n, Transceiver)
                    and not isinstance(next(network.successors(n)), Roadm)]
    # determine the size of the bitmap common to all the omses: find min and max frequencies of all amps
    # in the network. These gives the band not the center frequency. Then we use a reference channel
    # slot width (50GHz) to set the f_min, f_max
    f_min, f_max = find_network_freq_range(network)
    for node in oms_vertices:
        for edge in network.edges([node]):
            if not isinstance(edge[1], Transceiver):
                nd_in = edge[0]  # nd_in is a Roadm
                try:
                    nd_in.oms_list.append(oms_id)
                except AttributeError:
                    nd_in.oms_list = []
                    nd_in.oms_list.append(oms_id)
                nd_out = edge[1]

                params = {}
                params['oms_id'] = oms_id
                params['el_id_list'] = []
                params['el_list'] = []
                oms = OMS(**params)
                oms.add_element(nd_in)
                while not isinstance(nd_out, Roadm):
                    oms.add_element(nd_out)
                    # add an oms_id in the element
                    nd_out.oms_id = oms_id
                    nd_out.oms = oms
                    n_temp = nd_out
                    nd_out = next(n[1] for n in network.edges([n_temp]) if n[1].uid != nd_in.uid)
                    nd_in = n_temp

                oms.add_element(nd_out)
                # nd_out is a Roadm
                try:
                    nd_out.oms_list.append(oms_id)
                except AttributeError:
                    nd_out.oms_list = []
                    nd_out.oms_list.append(oms_id)

                bitmap = create_oms_bitmap(oms, equipment, f_min=f_min, f_max=f_max, grid=DEFAULT_GRID)
                oms.update_spectrum(f_min, f_max, guardband=DEFAULT_GUARDBAND, grid=DEFAULT_GRID,
                                    existing_spectrum=bitmap)
                # oms.assign_spectrum(13,7) gives back (193137500000000.0, 193225000000000.0)
                # as in the example in the standard
                # oms.assign_spectrum(13,7)

                oms_list.append(oms)
                oms_id += 1
    oms_list = align_grids(oms_list)
    reversed_oms(oms_list)
    return oms_list


def reversed_oms(oms_list: List[OMS]):
    """identifies reversed OMS

    only applicable for non parallel OMS
    """
    for oms in oms_list:
        has_reversed = False
        for this_o in oms_list:
            if (oms.el_id_list[0] == this_o.el_id_list[-1] and
                    oms.el_id_list[-1] == this_o.el_id_list[0]):
                oms.reversed_oms = this_o
                has_reversed = True
                break
        if not has_reversed:
            oms.reversed_oms = None


def bitmap_sum(band1: List[BitmapValue], band2: List[BitmapValue]) -> List[BitmapValue]:
    """mark occupied bitmap by 0 if the slot is occupied in band1 or in band2"""
    res = []
    for bit1, bit2 in zip(band1, band2):
        if bit1 in [BitmapValue.UNUSABLE, BitmapValue.OCCUPIED] or bit2 in [BitmapValue.UNUSABLE, BitmapValue.OCCUPIED]:
            res.append(BitmapValue.OCCUPIED)
        else:
            res.append(BitmapValue.FREE)
    return res


def build_path_oms_id_list(pth: List[Union[Roadm, Transceiver, Edfa, Multiband_amplifier, Fiber,
                                     RamanFiber, Fused]]) -> List[int]:
    """Returns the list of oms_id composing the path pth
    """
    path_oms = []
    for elem in pth:
        if not isinstance(elem, Roadm) and not isinstance(elem, Transceiver):
            # only edfa, fused and fibers have oms_id attribute
            path_oms.append(elem.oms_id)
    # remove duplicate oms_id, order is not important
    return list(set(path_oms))


def aggregate_oms_bitmap(path_oms: List[int], oms_list: List[OMS]) -> OMS:
    """Returns an OMS class with spectral occupation union of all oms spectral occupation in the path

    this is usefull to find an and to and free set of slots on the path
    """
    spectrum = oms_list[path_oms[0]].spectrum_bitmap
    bitmap = spectrum.bitmap
    # assuming all oms have same freq indices
    for oms in path_oms[1:]:
        bitmap = bitmap_sum(oms_list[oms].spectrum_bitmap.bitmap, bitmap)
    params = {
        'oms_id': 0,
        'el_id_list': 0,
        'el_list': []
    }
    freq_min = nvalue_to_frequency(spectrum.n_min)
    freq_max = nvalue_to_frequency(spectrum.n_max)
    aggregate_oms = OMS(**params)
    aggregate_oms.update_spectrum(freq_min, freq_max, grid=DEFAULT_GRID, guardband=spectrum.guardband,
                                  existing_spectrum=bitmap)
    return aggregate_oms


def spectrum_selection(test_oms: OMS, requested_m: int, requested_n: Optional[int] = None,
                       policy: str = FIRST_FIT) -> Union[Tuple[int, int, int], Tuple[None, None, None]]:
    """Collects spectrum availability and call the select_candidate function"""
    freq_index = test_oms.spectrum_bitmap.freq_index
    freq_index_min = test_oms.spectrum_bitmap.freq_index_min
    freq_index_max = test_oms.spectrum_bitmap.freq_index_max
    freq_availability = test_oms.spectrum_bitmap.bitmap

    if requested_n is None:
        # avoid slots reserved on the edge 0.15e-12 on both sides -> 24
        candidates = [(freq_index[i] + requested_m, freq_index[i], freq_index[i] + 2 * requested_m - 1)
                      for i in range(len(freq_availability))
                      if freq_availability[i:i + 2 * requested_m] == [BitmapValue.FREE] * (2 * requested_m)
                      and freq_index[i] >= freq_index_min
                      and freq_index[i + 2 * requested_m - 1] <= freq_index_max]

        candidate = select_candidate(candidates, policy=policy)
    else:
        i = test_oms.spectrum_bitmap.geti(requested_n)
        if (freq_availability[i - requested_m:i + requested_m] == [BitmapValue.FREE] * (2 * requested_m)
                and freq_index[i - requested_m] >= freq_index_min
                and freq_index[i + requested_m - 1] <= freq_index_max):
            # candidate is the triplet center_n, startn and stopn
            candidate = (requested_n, requested_n - requested_m, requested_n + requested_m - 1)
        else:
            candidate = (None, None, None)
    return candidate


def determine_slot_numbers(test_oms: OMS, requested_n: int, required_m: int, per_channel_m: int) -> int:
    """determines max availability around requested_n. requested_n should not be None"""
    bitmap = test_oms.spectrum_bitmap
    freq_index = bitmap.freq_index
    freq_index_min = bitmap.freq_index_min
    freq_index_max = bitmap.freq_index_max
    freq_availability = bitmap.bitmap
    center_i = bitmap.geti(requested_n)
    i = per_channel_m
    while (freq_availability[center_i - i:center_i + i] == [BitmapValue.FREE] * (2 * i)
           and freq_index[center_i - i] >= freq_index_min
           and freq_index[center_i + i - 1] <= freq_index_max
           and i <= required_m):
        i += per_channel_m
    return i - per_channel_m


def select_candidate(candidates: List[Tuple[int, int, int]], policy: str) -> Tuple[int, int, int]:
    """selects a candidate among all available spectrum"""
    if policy == FIRST_FIT and candidates:
        return candidates[0]
    if policy == LAST_FIT and candidates:
        return candidates[-1]
    if not candidates:
        return (None, None, None)
    raise ServiceError('Only first_fit and last_fit spectrum assignment policy are implemented.')


def compute_n_m(required_m: int, rq: PathRequest, path_oms: List[int], oms_list: List[OMS], per_channel_m: int,
                policy: str = FIRST_FIT) -> Tuple[list, list, int]:
    """ based on requested path_bandwidth fill in M=None values with uint values, using per_channel_m
    and center frequency, with first fit strategy. The function checks the available spectrum but check
    consistencies among M values of the request, but not with other requests.
    For example, if request is for 32 slots corresponding to 8 x 4 slots of 32Gbauds channels,
    the following frequency slots will result in the following assignment

    N = 0, 8,    16, 32           -> 0,   8,   16,   32
    M = 8, None, 8,  None         -> 8,   8,    8,    8

    N = 0,    8,    16, 32        -> 0,   , 16
    M = None, None, 8,  None      -> 24,  , 8
    """
    selected_m = []
    selected_n = []
    remaining_slots_to_serve = required_m
    # order slots for the computation: assign biggest m first
    rq_N, rq_M, order = order_slots([{'N': n, 'M': m} for n, m in zip(rq.N, rq.M)])
    # Create an oms that represents current assignments of all oms listed in path_oms, and test N and M on it.
    # If M is defined, checks that proposed N, M is free
    test_oms = aggregate_oms_bitmap(path_oms, oms_list)
    for n, m in zip(rq_N, rq_M):
        if m is not None and n is not None:
            # check availabilityfor this n, m
            available_slots = determine_slot_numbers(test_oms, n, m, m)
            if available_slots == 0:
                # if n, m are not feasible, break at this point no have non zero remaining_slots_to_serve
                # in order to blocks the request (even is other N,M where feasible)
                break
        elif m is not None and n is None:
            # find a candidate n
            n, _, _ = spectrum_selection(test_oms, m, None, policy=policy)
            if n is None:
                # if no n is feasible for the m, block the request
                break
        elif m is None and n is not None:
            # find a feasible m for this n. If None is found, then block the request
            m = determine_slot_numbers(test_oms, n, remaining_slots_to_serve, per_channel_m)
            if m == 0 or remaining_slots_to_serve == 0:
                break
        else:
            # if n and m are not defined, try to find a single  assignment to fits the remaining slots to serve
            # (first fit strategy)
            n, _, _ = spectrum_selection(test_oms, remaining_slots_to_serve, None, policy=policy)
            if n is None or remaining_slots_to_serve == 0:
                break
            m = remaining_slots_to_serve
        selected_m.append(m)
        selected_n.append(n)
        test_oms.assign_spectrum(n, m)
        remaining_slots_to_serve = remaining_slots_to_serve - m

    # re-order selected_m and selected_n according to initial request N, M order, ignoring None values
    not_selected = [None for i in range(len(rq_N) - len(selected_n))]
    selected_m = restore_order(selected_m + not_selected, order)
    selected_n = restore_order(selected_n + not_selected, order)
    return selected_n, selected_m, remaining_slots_to_serve


def pth_assign_spectrum(pths: List[Union[Roadm, Transceiver, Edfa, Multiband_amplifier, Fiber, RamanFiber, Fused]],
                        rqs: List[PathRequest], oms_list: List[OMS],
                        rpths: List[Union[Roadm, Transceiver, Edfa, Multiband_amplifier, Fiber, RamanFiber, Fused]],
                        policy=FIRST_FIT):
    """basic first fit assignment

    if reversed path are provided, means that occupation is bidir
    """
    for pth, rq, rpth in zip(pths, rqs, rpths):
        if hasattr(rq, 'blocking_reason'):
            rq.N = None
            rq.M = None
        else:
            # computes the number of channels required for path_bandwidth and the min required nb of slots
            # for one channel (corresponds to the spacing)
            nb_wl, required_m = compute_spectrum_slot_vs_bandwidth(rq.path_bandwidth, rq.spacing, rq.bit_rate)
            _, per_channel_m = compute_spectrum_slot_vs_bandwidth(rq.bit_rate,
                                                                  rq.spacing, rq.bit_rate)
            # find oms ids that are concerned both by pth and rpth
            path_oms = build_path_oms_id_list(pth + rpth)
            if getattr(rq, 'M', None) is not None and all(rq.M):
                # if all M are well defined: Consistency check that the requested M are enough to carry the nb_wl:
                # check that the integer number of per_channel_m carried in each M value is enough to carry nb_wl.
                # if not, blocks the demand
                nb_channels_of_request = sum(m // per_channel_m for m in rq.M)
                # TODO: elaborate a more accurate estimate with nb_wl * min_spacing + possibly guardbands in case of
                # superchannel closed packing.
                if nb_wl > nb_channels_of_request:
                    rq.N = None
                    rq.M = None
                    rq.blocking_reason = 'NOT_ENOUGH_RESERVED_SPECTRUM'
                    # need to stop here for this request and not go though spectrum selection process
                    continue
            # Use the req.M even if nb_wl and required_m are smaller.
            # first fit strategy: assign as many lambda as possible in the None remaining N, M values
            selected_n, selected_m, remaining_slots_to_serve = \
                compute_n_m(required_m, rq, path_oms, oms_list, per_channel_m, policy=policy)
            # if there are some remaining_slots_to_serve, this means that provided rq.M and rq.N values
            # were not possible.
            # Then do not go though spectrum assignment process and blocks the demand
            if remaining_slots_to_serve > 0:
                rq.N = None
                rq.M = None
                rq.blocking_reason = 'NO_SPECTRUM'
                continue
            for oms_elem in path_oms:
                for this_n, this_m in zip(selected_n, selected_m):
                    if this_m is not None:
                        oms_list[oms_elem].assign_spectrum(this_n, this_m)
                oms_list[oms_elem].add_service(rq.request_id, nb_wl)
            rq.N = selected_n
            rq.M = selected_m
