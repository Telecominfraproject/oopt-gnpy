#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.spectrum_assignment
=============================

This module contains .....

comment TODO
"""

from collections import namedtuple
from gnpy.core.elements import Roadm, Transceiver
from numpy import array
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from math import ceil
from copy import copy

logger = getLogger(__name__)


class Bitmap:
    def __init__(self, f_min,f_max,  grid, guardband = 0.15e12, bitmap=None):
        # n is the min index including guardband. Guardband is require to be sure 
        # that a channel can be assigned  with center frequency fmin (means that its
        # slot occupation goes below freq_index_min)
        n_min = frequency_to_n(f_min-guardband ,grid)  
        n_max = frequency_to_n(f_max+guardband ,grid)-1
        self.n_min = n_min
        self.n_max = n_max
        self.freq_index_min = frequency_to_n(f_min)
        self.freq_index_max = frequency_to_n(f_max) 
        self.freq_index = list(range(n_min,n_max+1))
        if bitmap is None:
            self.bitmap = [1] * (n_max-n_min+1)
        else:
            if len(bitmap) == len(freq_index):
                self.bitmap = bitmap
            else:
                msg = f'bitmap is not consistant with f_min{f_min} - n :{n_min} and f_max{f_max}- n :{n_max}'
                logger.critical(msg)
                exit()               
    def getn(self,i):
        return self.freq_index[i]
    def geti(self,n):
        return self.freq_index.index(n)
    def insert_left(self,newbitmap):
        self.bitmap =  newbitmap + self.bitmap
        temp = list(range(self.n_min-len(newbitmap),self.n_min))
        self.freq_index = temp + self.freq_index
        self.n_min = self.freq_index[0]
    def insert_right(self,newbitmap):
        self.bitmap =  self.bitmap + newbitmap
        self.freq_index = self.freq_index + list(range(self.n_max,self.n_max+len(newbitmap)))
        self.n_max = self.freq_index[-1]

#    +'grid available_slots f_min f_max services_list')
OMSParams = namedtuple('OMSParams','oms_id el_id_list el_list') 

class OMS:
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
        return '\n\t'.join([  f'{type(self).__name__} {self.oms_id}',
        	                f'{self.el_id_list[0]} - {self.el_id_list[-1]}'])
    def __repr__(self):
         return '\n\t'.join([  f'{type(self).__name__} {self.oms_id}',
        	                f'{self.el_id_list[0]} - {self.el_id_list[-1]}',
        	                '\n'])

    def add_element(self,el):
        self.el_id_list.append(el.uid)
        self.el_list.append(el)

    def update_spectrum(self,f_min , f_max, guardband = 0.15e12, existing_spectrum = None, grid = 0.00625e12):
    	# frequencies expressed in Hz
        if existing_spectrum is None:
            # add some 150 GHz margin to eable a center channel on f_min
            # use ITU-T G694.1
            """
            7 Flexible DWDM grid definition
              For the flexible DWDM grid, the allowed frequency slots have a nominal central frequency
              (in THz) defined by:
              193.1 + n × 0.00625 where n is a positive or negative integer including 0
              and 0.00625 is the nominal central frequency granularity in THz
              and a slot width defined by:
              12.5 × m where m is a positive integer and 12.5 is the slot width granularity in GHz.
              Any combination of frequency slots is allowed as long as no two frequency slots overlap.
            """
            # TODO : add explaination on that / parametrize .... 
            self.spectrum_bitmap = Bitmap(f_min,f_max, grid, guardband)
            # print(len(self.spectrum_bitmap.bitmap))

    def assign_spectrum(self,n,m):
        # print("assign_spectrum")
        # print(f'n , m :{n},{m}')
        if n is None or m is None :
            msg = f'could not assign None values'
            logger.critical(msg)
            exit()              
        startn , stopn = m_to_slots(n,m)
        # print(f'startn stop n {startn} , {stopn}')
        # assumes that guardbands are sufficient to ensure that assigning a center channel
        # at fmin or fmax is OK is startn > self.spectrum_bitmap.n_min
        if (n <= self.spectrum_bitmap.freq_index_max and n>= self.spectrum_bitmap.freq_index_min and 
            stopn <= self.spectrum_bitmap.n_max and startn>self.spectrum_bitmap.n_min ):
            # verification that both length are identical
            # print(len(self.spectrum_bitmap.bitmap[self.spectrum_bitmap.geti(startn):self.spectrum_bitmap.geti(stopn)+1]))
            # print(stopn-startn+1)
            self.spectrum_bitmap.bitmap[self.spectrum_bitmap.geti(startn):self.spectrum_bitmap.geti(stopn)+1] = [0] * (stopn-startn+1)
            # print(self.spectrum_bitmap.bitmap)
            # print(m_to_freq(n,m,grid = 0.00625e12))
            return True
        else:
            msg = f'Could not assign n {n}, m {m} values: one or several slots are not available'
            logger.info(msg)
            return False

    def add_service(self,service_id, nb_wl):
        self.service_list.append(service_id)
        self.nb_channels += nb_wl 

def frequency_to_n(freq,grid = 0.00625e12):
    return (int)((freq-193.1e12)/grid)  

def n_to_frequency(n,grid=0.00625e12):
	return 193.1e12 + n*grid

def m_to_slots(n,m):
    startn = n - m
    stopn = n + m -1
    return startn , stopn

def slots_to_m(startn , stopn):
    n = (int)((startn+stopn-1)/2)
    m = (int)((stopn-startn+1)/2)
    return n,m

def m_to_freq(n,m,grid=0.00625):
    startn , stopn = m_to_slots(n,m)
    fstart = n_to_frequency(startn,grid)
    fstop = n_to_frequency(stopn+1,grid)
    return fstart, fstop

def align_grids(oms_list):
    # used to apply same grid to all oms : same starting n, stop n and slot size
    # out of grid slots are set to 0
    n_min = min([o.spectrum_bitmap.n_min for o in oms_list])
    n_max = max([o.spectrum_bitmap.n_max for o in oms_list])
    for o in oms_list:
        if (o.spectrum_bitmap.n_min - n_min) > 0 :
            o.spectrum_bitmap.insert_left([0] * (o.spectrum_bitmap.n_min - n_min))
        if (n_max - o.spectrum_bitmap.n_max) > 0 :
            o.spectrum_bitmap.insert_right( [0] * (n_max - o.spectrum_bitmap.n_max))

    return oms_list

def build_OMS_list(network,equipment):
    oms_id = 0
    OMS_list =[]
    for node in [n for n in network.nodes() if isinstance(n, Roadm)]:
        for edge in network.edges([node]):
            # print(f'coucou  {node.uid} {edge[0].uid}   {edge[1].uid}')
            if not isinstance(edge[1],Transceiver) :
                n_in = edge[0] # n_in is a Roadm
                try: 
                    n_in.oms_list.append(oms_id)
                except AttributeError:
                    n_in.oms_list = []
                    n_in.oms_list.append(oms_id)
                n_out = edge[1]
                el_list = []
                params = {}
                params['oms_id']= oms_id
                params['el_id_list'] = []
                params['el_list']=[]
                oms = OMS(**params)
                oms.add_element(n_in)
                while not isinstance(n_out,Roadm) :
                    oms.add_element(n_out)
                    # add an oms_id in the element
                    n_out.oms_id =  oms_id
                    n_out.oms = oms
                    n_temp = n_out
                    n_out= next(n[1] for n in network.edges([n_temp]) if n[1].uid != n_in.uid)
                    n_in = n_temp 

                oms.add_element(n_out)
                # n_out is a Roadm
                try: 
                    n_out.oms_list.append(oms_id)
                except AttributeError:
                    n_out.oms_list = []
                    n_out.oms_list.append(oms_id)  

                # print(f'coucou2 {oms.oms_id} {oms.el_id_list[0]} {oms.el_id_list[-1]}')
                # for e in oms.el_id_list:
                #     print(f' {e}')

                # TODO do not forget to correct next line !
                # to test different grids
                if oms_id<3:
                    #oms.update_spectrum(equipment['SI']['default'].f_min + 0.5e12,equipment['SI']['default'].f_max, grid = 0.00625e12)
                    oms.update_spectrum(equipment['SI']['default'].f_min,equipment['SI']['default'].f_max, grid = 0.00625e12)
                    # print(len(oms.spectrum_bitmap.bitmap))
                else:
                    oms.update_spectrum(equipment['SI']['default'].f_min,equipment['SI']['default'].f_max, grid = 0.00625e12)
                    # print(len(oms.spectrum_bitmap.bitmap))                    
                # oms.assign_spectrum(13,7) gives back (193137500000000.0, 193225000000000.0) 
                # as in the example in the standard
                # oms.assign_spectrum(13,7)

                OMS_list.append(oms)
                oms_id += 1
                # print('\n')
    OMS_list = align_grids(OMS_list)
    reversed_OMS(OMS_list)
    return OMS_list

def reversed_OMS(OMS_list):
    # only applicable for non parallel OMS
    for oms in OMS_list:
        for o in OMS_list:
            if oms.el_id_list[0] == o.el_id_list[-1] and oms.el_id_list[-1] == o.el_id_list[0] :
                oms.reversed_oms = o 
                break

def spectrum_selection(pth,OMS_list, m, N = None):
    # step 1 collects pth spectrum availability
    # step 2 if n is not None try to assign the spectrum
    #            if the spectrum is not available then sends back an "error"
    #        if n is None selects candidate spectrum
    #            select spectrum that fits the policy ( first fit, random, ABP...)
    # step3 returns the selection

    path_spectrum = []
    # use indexes instead of ITU-T n values
    path_oms = []
    for el in pth:
        # print(el.uid)
        # if not isinstance(el,Roadm) and  not  isinstance(el,Transceiver):
        #     print(el.oms_id)
        if not isinstance(el,Roadm) and not isinstance(el,Transceiver) :
        	# only edfa, fused and fibers have oms_id attribute
            path_oms.append(el.oms_id)
    # print(path_oms)
    # remove duplicate oms_id, order is not important
    path_oms = list(set(path_oms))
    # print(path_oms)
    # print(OMS_list[path_oms[0]].spectrum_bitmap.bitmap)
    freq_availability = 1 - array(OMS_list[path_oms[0]].spectrum_bitmap.bitmap)
    # assuming all oms have same freq index
    freq_index = OMS_list[path_oms[0]].spectrum_bitmap.freq_index
    freq_index_min = OMS_list[path_oms[0]].spectrum_bitmap.freq_index_min
    freq_index_max = OMS_list[path_oms[0]].spectrum_bitmap.freq_index_max

    for oms in path_oms[1:]:
        freq_availability = (freq_availability + 1- array(OMS_list[oms].spectrum_bitmap.bitmap))/2
    freq_availability = freq_availability.astype(int)
    freq_availability = 1- freq_availability.astype(int)
    freq_availability = freq_availability.tolist()
    # print(freq_availability)
    
    if N is None:
        # avoid slots reserved on the edge 0.15e-12 on both sides -> 24 
        candidates =  [(freq_index[i]+m,freq_index[i],freq_index[i]+2*m-1) 
            for i in range(len(freq_availability) )
            if freq_availability[i:i+2*m] == [1] * (2*m) and freq_index[i] >= freq_index_min 
            and freq_index[i] <= freq_index_max]
        candidate = select_candidate(candidates, policy = 'first_fit')
    else:
        i = OMS_list[path_oms[0]].spectrum_bitmap.geti(N)
        # print(f'N {N} i {i}')
        # print(freq_availability[i-m:i+m] )
        # print(freq_index[i-m:i+m])
        if freq_availability[i-m:i+m] == [1] * (2*m):
            candidate = (N , N-m , N+m-1)
        else:
            candidate = (None, None , None)
        # print("coucou11")
        # print(candidate)
    # print(freq_availability[321:321+2*m])
    # a = [i+321 for i in range(2*m)]
    # print(a)
    # print(candidate)
    return candidate, path_oms

def select_candidate(candidates, policy):
    if policy == 'first_fit':
        if candidates:
            return candidates[0]
        else:
            return (None, None , None)

def pth_assign_spectrum(pths, rqs, oms_list):
    
    # baseic first fit assignment
    for i, pth in enumerate(pths) :
        # computes the number of channels required
        try:
            if rqs[i].blocking_reason :
                rqs[i].blocked = True
                rqs[i].N = 0
                rqs[i].M = 0
        except AttributeError :              
            nb_wl = ceil(rqs[i].path_bandwidth / rqs[i].bit_rate) 
            # computes the total nb of slots according to requested spacing
            # todo : express superchannels
            # assumes that all channels must be grouped
            # todo : enables non contiguous reservation in case of blocking
            M = ceil(rqs[i].spacing / 0.0125e12) * nb_wl
            (n,startm,stopm) , path_oms = spectrum_selection(pth,oms_list, M, N = None)
            if n is not None : 
                for o in path_oms:
                    oms_list[o].assign_spectrum(n,M)
                    oms_list[o].add_service(rqs[i].request_id,nb_wl)
                rqs[i].blocked = False
                rqs[i].N = n
                rqs[i].M = M
            else:
                rqs[i].blocked = True
                rqs[i].N = 0
                rqs[i].M = 0
                rqs[i].blocking_reason = 'NO_SPECTRUM'
       

