#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from math import ceil
from gnpy.core.elements import Roadm, Transceiver
from gnpy.core.exceptions import ServiceError, SpectrumError

LOGGER = getLogger(__name__)


class Bitmap:
    """ records the spectrum occupation
    """

    def __init__(self, f_min, f_max, grid, guardband=0.15e12, bitmap=None):
        # n is the min index including guardband. Guardband is require to be sure
        # that a channel can be assigned  with center frequency fmin (means that its
        # slot occupation goes below freq_index_min
        n_min = frequency_to_n(f_min - guardband, grid)
        n_max = frequency_to_n(f_max + guardband, grid) - 1
        self.n_min = n_min
        self.n_max = n_max
        self.freq_index_min = frequency_to_n(f_min)
        self.freq_index_max = frequency_to_n(f_max)
        self.freq_index = list(range(n_min, n_max + 1))
        if bitmap is None:
            self.bitmap = [1] * (n_max - n_min + 1)
        elif len(bitmap) == len(self.freq_index):
            self.bitmap = bitmap
        else:
            raise SpectrumError(f'bitmap is not consistant with f_min{f_min} - n: {n_min} and f_max{f_max}- n :{n_max}')

    def getn(self, i):
        """ converts the n (itu grid) into a local index
        """
        return self.freq_index[i]

    def geti(self, nvalue):
        """ converts the local index into n (itu grid)
        """
        return self.freq_index.index(nvalue)

    def insert_left(self, newbitmap):
        """ insert bitmap on the left to align oms bitmaps if their start frequencies are different
        """
        self.bitmap = newbitmap + self.bitmap
        temp = list(range(self.n_min - len(newbitmap), self.n_min))
        self.freq_index = temp + self.freq_index
        self.n_min = self.freq_index[0]

    def insert_right(self, newbitmap):
        """ insert bitmap on the right to align oms bitmaps if their stop frequencies are different
        """
        self.bitmap = self.bitmap + newbitmap
        self.freq_index = self.freq_index + list(range(self.n_max, self.n_max + len(newbitmap)))
        self.n_max = self.freq_index[-1]


#    +'grid available_slots f_min f_max services_list')
OMSParams = namedtuple('OMSParams', 'oms_id el_id_list el_list')


class OMS:
    """ OMS class is the logical container that represent a link between two adjacent ROADMs and
        records the crossed elements and the occupied spectrum
    """

    def __init__(self, *args, **params):
        params = OMSParams(**params)
        self.oms_id = params.oms_id
        self.el_id_list = params.el_id_list
        self.el_list = params.el_list
        self.spectrum_bitmap = []
        self.nb_channels = 0
        self.service_list = []
    # TODO

    def __str__(self):
        return '\n\t'.join([f'{type(self).__name__} {self.oms_id}',
                            f'{self.el_id_list[0]} - {self.el_id_list[-1]}'])

    def __repr__(self):
        return '\n\t'.join([f'{type(self).__name__} {self.oms_id}',
                            f'{self.el_id_list[0]} - {self.el_id_list[-1]}', '\n'])

    def add_element(self, elem):
        """ records oms elements
        """
        self.el_id_list.append(elem.uid)
        self.el_list.append(elem)

    def update_spectrum(self, f_min, f_max, guardband=0.15e12, existing_spectrum=None,
                        grid=0.00625e12):
        """ frequencies expressed in Hz
        """
        if existing_spectrum is None:
            # add some 150 GHz margin to enable a center channel on f_min
            # use ITU-T G694.1
            # Flexible DWDM grid definition
            # For the flexible DWDM grid, the allowed frequency slots have a nominal
            # central frequency (in THz) defined by:
            # 193.1 + n × 0.00625 where n is a positive or negative integer including 0
            # and 0.00625 is the nominal central frequency granularity in THz
            # and a slot width defined by:
            # 12.5 × m where m is a positive integer and 12.5 is the slot width granularity in
            # GHz.
            # Any combination of frequency slots is allowed as long as no two frequency
            # slots overlap.

            # TODO : add explaination on that / parametrize ....
            self.spectrum_bitmap = Bitmap(f_min, f_max, grid, guardband)
            # print(len(self.spectrum_bitmap.bitmap))

    def assign_spectrum(self, nvalue, mvalue):
        """ change oms spectrum to mark spectrum assigned
        """
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
        self.spectrum_bitmap.bitmap[self.spectrum_bitmap.geti(startn):self.spectrum_bitmap.geti(stopn) + 1] = [0] * (stopn - startn + 1)

    def add_service(self, service_id, nb_wl):
        """ record service and mark spectrum as occupied
        """
        self.service_list.append(service_id)
        self.nb_channels += nb_wl


def frequency_to_n(freq, grid=0.00625e12):
    """ converts frequency into the n value (ITU grid)
        reference to Recommendation G.694.1 (02/12), Figure I.3
        https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> frequency_to_n(193.1375e12)
    6
    >>> frequency_to_n(193.225e12)
    20

    """
    return (int)((freq - 193.1e12) / grid)


def nvalue_to_frequency(nvalue, grid=0.00625e12):
    """ converts n value into a frequency
        reference to Recommendation G.694.1 (02/12), Table 1
        https://www.itu.int/rec/T-REC-G.694.1-201202-I/en

    >>> nvalue_to_frequency(6)
    193137500000000.0
    >>> nvalue_to_frequency(-1, 0.1e12)
    193000000000000.0

    """
    return 193.1e12 + nvalue * grid


def mvalue_to_slots(nvalue, mvalue):
    """ convert center n an m into start and stop n
    """
    startn = nvalue - mvalue
    stopn = nvalue + mvalue - 1
    return startn, stopn


def slots_to_m(startn, stopn):
    """ converts the start and stop n values to the center n and m value
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


def m_to_freq(nvalue, mvalue, grid=0.00625e12):
    """ converts m into frequency range
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


def align_grids(oms_list):
    """ used to apply same grid to all oms : same starting n, stop n and slot size
        out of grid slots are set to 0
    """
    n_min = min([o.spectrum_bitmap.n_min for o in oms_list])
    n_max = max([o.spectrum_bitmap.n_max for o in oms_list])
    for this_o in oms_list:
        if (this_o.spectrum_bitmap.n_min - n_min) > 0:
            this_o.spectrum_bitmap.insert_left([0] * (this_o.spectrum_bitmap.n_min - n_min))
        if (n_max - this_o.spectrum_bitmap.n_max) > 0:
            this_o.spectrum_bitmap.insert_right([0] * (n_max - this_o.spectrum_bitmap.n_max))
    return oms_list


def build_oms_list(network, equipment):
    """ initialization of OMS list in the network
        an oms is build reading all intermediate nodes between two adjacent ROADMs
        each element within the list is being added an oms and oms_id to record the
        oms it belongs to.
        the function supports different spectrum width and supposes that the whole network
        works with the min range among OMSs
    """
    oms_id = 0
    oms_list = []
    for node in [n for n in network.nodes() if isinstance(n, Roadm)]:
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

                oms.update_spectrum(equipment['SI']['default'].f_min,
                                    equipment['SI']['default'].f_max, grid=0.00625e12)
                # oms.assign_spectrum(13,7) gives back (193137500000000.0, 193225000000000.0)
                # as in the example in the standard
                # oms.assign_spectrum(13,7)

                oms_list.append(oms)
                oms_id += 1
    oms_list = align_grids(oms_list)
    reversed_oms(oms_list)
    return oms_list


def reversed_oms(oms_list):
    """ identifies reversed OMS
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


def bitmap_sum(band1, band2):
    """mark occupied bitmap by 0 if the slot is occupied in band1 or in band2"""
    res = []
    for i, elem in enumerate(band1):
        if band2[i] * elem == 0:
            res.append(0)
        else:
            res.append(1)
    return res


def spectrum_selection(pth, oms_list, requested_m, requested_n=None):
    """Collects spectrum availability and call the select_candidate function"""

    # use indexes instead of ITU-T n values
    path_oms = []
    for elem in pth:
        if not isinstance(elem, Roadm) and not isinstance(elem, Transceiver):
            # only edfa, fused and fibers have oms_id attribute
            path_oms.append(elem.oms_id)
    # remove duplicate oms_id, order is not important
    path_oms = list(set(path_oms))
    # assuming all oms have same freq index
    if not path_oms:
        candidate = (None, None, None)
        return candidate, path_oms
    freq_index = oms_list[path_oms[0]].spectrum_bitmap.freq_index
    freq_index_min = oms_list[path_oms[0]].spectrum_bitmap.freq_index_min
    freq_index_max = oms_list[path_oms[0]].spectrum_bitmap.freq_index_max

    freq_availability = oms_list[path_oms[0]].spectrum_bitmap.bitmap
    for oms in path_oms[1:]:
        freq_availability = bitmap_sum(oms_list[oms].spectrum_bitmap.bitmap, freq_availability)
    if requested_n is None:
        # avoid slots reserved on the edge 0.15e-12 on both sides -> 24
        candidates = [(freq_index[i] + requested_m, freq_index[i], freq_index[i] + 2 * requested_m - 1)
                      for i in range(len(freq_availability))
                      if freq_availability[i:i + 2 * requested_m] == [1] * (2 * requested_m)
                      and freq_index[i] >= freq_index_min
                      and freq_index[i + 2 * requested_m - 1] <= freq_index_max]

        candidate = select_candidate(candidates, policy='first_fit')
    else:
        i = oms_list[path_oms[0]].spectrum_bitmap.geti(requested_n)
        # print(f'N {requested_n} i {i}')
        # print(freq_availability[i-m:i+m] )
        # print(freq_index[i-m:i+m])
        if (freq_availability[i - requested_m:i + requested_m] == [1] * (2 * requested_m) and
                freq_index[i - requested_m] >= freq_index_min
                and freq_index[i + requested_m - 1] <= freq_index_max):
            # candidate is the triplet center_n, startn and stopn
            candidate = (requested_n, requested_n - requested_m, requested_n + requested_m - 1)
        else:
            candidate = (None, None, None)
        # print("coucou11")
        # print(candidate)
    # print(freq_availability[321:321+2*m])
    # a = [i+321 for i in range(2*m)]
    # print(a)
    # print(candidate)
    return candidate, path_oms


def select_candidate(candidates, policy):
    """ selects a candidate among all available spectrum
    """
    if policy == 'first_fit':
        if candidates:
            return candidates[0]
        else:
            return (None, None, None)
    else:
        raise ServiceError('Only first_fit spectrum assignment policy is implemented.')


def pth_assign_spectrum(pths, rqs, oms_list, rpths):
    """ basic first fit assignment
        if reversed path are provided, means that occupation is bidir
    """
    for i, pth in enumerate(pths):
        # computes the number of channels required
        try:
            if rqs[i].blocking_reason:
                rqs[i].blocked = True
                rqs[i].N = 0
                rqs[i].M = 0
        except AttributeError:
            nb_wl = ceil(rqs[i].path_bandwidth / rqs[i].bit_rate)
            # computes the total nb of slots according to requested spacing
            # TODO : express superchannels
            # assumes that all channels must be grouped
            # TODO : enables non contiguous reservation in case of blocking
            requested_m = ceil(rqs[i].spacing / 0.0125e12) * nb_wl
            # concatenate all path and reversed path elements to derive slots availability
            (center_n, startn, stopn), path_oms = spectrum_selection(pth + rpths[i], oms_list, requested_m,
                                                                     requested_n=None)
            # checks that requested_m is fitting startm and stopm
            # if not None, center_n and start, stop frequencies are applicable to all oms of pth
            # checks that spectrum is not None else indicate blocking reason
            if center_n is not None:
                # checks that requested_m is fitting startm and stopm
                if 2 * requested_m > (stopn - startn + 1):
                    msg = f'candidate: {(center_n, startn, stopn)} is not consistant ' +\
                        f'with {requested_m}'
                    LOGGER.critical(msg)
                    raise ValueError(msg)

                for oms_elem in path_oms:
                    oms_list[oms_elem].assign_spectrum(center_n, requested_m)
                    oms_list[oms_elem].add_service(rqs[i].request_id, nb_wl)
                rqs[i].blocked = False
                rqs[i].N = center_n
                rqs[i].M = requested_m
            else:
                rqs[i].blocked = True
                rqs[i].N = 0
                rqs[i].M = 0
                rqs[i].blocking_reason = 'NO_SPECTRUM'
