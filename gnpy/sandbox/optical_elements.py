# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:09:47 2016

@author: briantaylor
"""

import numpy as np
from gnpy.constants import c, h


class NetworkElement:

    def __init__(self, **kwargs):
        """
        self.direction = [("E", "Z"), ("E", "Z"), ("E", "Z"), ("W", "Z")]
        self.port_mapping = [(1, 5), (2, 5), (3, 5), (4, 5)]
        self.uid = uuid4()
        self.coordinates = (29.9792, 31.1342)
        """
        try:
            for key in ('port_mapping', 'direction', 'coordinates', 'name',
                        'description', 'manufacturer', 'model', 'sn', 'id'):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                else:
                    setattr(self, key, None)
                    # print('No Value defined for :', key)
                    # TODO: add logging functionality
        except KeyError as e:
            if 'name' in kwargs:
                s = kwargs['name']
                print('Missing Required Network Element Key!', 'name:=', s)
#           TODO Put log here instead of print
            print(e)
            raise

    def get_output_ports(self):
        """Translate the port mapping into list of output ports
        """
        return None

    def get_input_ports(self):
        """Translate the port mapping into list of output ports
        """
        return None

    def __repr__(self):
        return self.__class__.__name__


class Edfa(NetworkElement):

    def __init__(self, **kwargs):
        '''Reads in configuration data checking for keys.  Sets those attributes
        for each element that exists.
        conventions:
        units are SI except where noted below (meters, seconds, Hz)
        rbw=12.5 GHz today.
        TODO add unit checking so inputs can be added in conventional
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


class Fiber(NetworkElement):
    class_counter = 0

    def __init__(self, **kwargs):
        """ Reads in configuration data checking for keys.  Sets those
        attributes for each element that exists.
        conventions:
        units are SI (meters, seconds, Hz) except where noted below
        rbw=12.5 GHz today.  TODO add unit checking so inputs can be added
        in conventional nm units.
        nf_db = noise figure in dB units
        psat_db = saturation power in dB units
        gain_db = gain in dB units
        pdg_db = polarization dependent gain in dB
        ripple_db = gain ripple in dB
        """
        try:
            for key in ('fiber_type', 'attenuationdB', 'span_length',
                        'dispersion', 'wavelengths', 'id', 'name', 'location',
                        'direction', 'launch_power', 'rbw'):
                if key in kwargs:
                    setattr(self, key, kwargs[key])
                elif 'id' in kwargs is None:
                    setattr(self, 'id', Span.class_counter)
                    Span.class_counter += 1
                else:
                    setattr(self, key, None)
                    print('No Value defined for :', key)
        except KeyError as e:
            if 'name' in kwargs:
                s = kwargs['name']
                print('Missing Span Input Key!', 'name:=', s)
            print(e)
            raise

    def effective_length(self, loss_coef):
        alpha_dict = self.dbkm_2_lin(loss_coef)
        alpha = alpha_dict['alpha_acoef']
        leff = 1 - np.exp(-2 * alpha * self.span_length)
        return leff

    def asymptotic_length(self, loss_coef):
        alpha_dict = self.dbkm_2_lin(loss_coef)
        alpha = alpha_dict['alpha_acoef']
        aleff = 1/(2 * alpha)
        return aleff



    def beta2(self, dispersion, ref_wavelength=None):
        """ Returns beta2 from dispersion parameter.  Dispersion is entered in
        ps/nm/km.  Disperion can be a numpy array or a single value.  If a
        value ref_wavelength is not entered 1550e-9m will be assumed.
        ref_wavelength can be a numpy array.
        """
        if ref_wavelength is None:
            ref_wavelength = 1550e-9
        wl = ref_wavelength
        D = np.abs(dispersion)
        b2 = (10**21) * (wl**2) * D / (2 * np.pi * c)
#       10^21 scales to ps^2/km
        return b2

#    TODO
    def generic_span(self):
        """ calculates a generic version that shows how all the functions of
        the class are used.  It makes the following assumptions about the span:
        """
        return

    def __repr__(self):
        return f'{self.__class__.__name__}({self.span_length}km)'
