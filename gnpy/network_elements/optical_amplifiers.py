# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:09:47 2016

@author: briantaylor
"""

import numpy as np
from scipy.constants import h, c
from numpy import array

class Edfa(object):
    class_counter= 0
    def __init__(self, **kwargs):
        '''Reads in configuration data checking for keys.  Sets those attributes
        for each element that exists.
        conventions:
        units are SI except where noted below (meters, seconds, Hz)
        rbw=12.5 GHz today.  TODO add unit checking so inputs can be added in conventional
        nm units.
        nfdB = noise figure in dB units
        psatdB = saturation power in dB units
        gaindB = gain in dB units
        pdgdB = polarization dependent gain in dB
        rippledB = gain ripple in dB
        
        '''
        try:
            for key in ('gaindB', 'nfdB', 'psatdB', 'rbw', 'wavelengths',
                        'pdgdB', 'rippledB', 'id', 'node', 'location'):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                elif 'id' in kwargs is None:
                    setattr(self, 'id', Edfa.class_counter)      
                    Edfa.class_counter += 1
                else:
                    setattr(self, key, None)
                    print('No Value defined for :', key)
            self.pas = [(h*c/ll)*self.rbw*1e9 for ll in self.wavelengths]
    
        except KeyError as e:
            if 'name' in kwargs:
                s = kwargs['name']
        
            print('Missing Edfa Input Key!', 'name:=', s)
            print(e)
            raise
    
    
    
    