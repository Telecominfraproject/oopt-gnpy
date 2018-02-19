#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network elements class with SpectralInformation propagation using 
__call__ and propagate methods
@author: giladgoldfarb
@author: briantaylor
@author: jeanluc-auge
@acknowledgement : Dave Boertjes
"""
import numpy as np
from scipy.constants import c, h

from gnpy.core.node import Node
from gnpy.core.units import UNITS
from gnpy.core.utils import lin2db, db2lin, itufs


class Transceiver(Node):
    def __init__(self, config):
        super().__init__(config)

    def snr(self, spectral_info):
        osnr_ase = [lin2db(c.power.signal/c.power.ase)
                for c in spectral_info.carriers 
                        if c.power.ase>1e-13]
        ratio_01nm = [lin2db(12.5e9/c.baud_rate) for c in spectral_info.carriers]
        osnr_ase_01nm = [ase - ratio for ase, ratio in zip(osnr_ase, ratio_01nm)]
        osnr_nli = [lin2db(c.power.signal/c.power.nli) for c in spectral_info.carriers]
        snr = [lin2db(c.power.signal/(c.power.nli+c.power.ase)) for c in spectral_info.carriers]
        print('OSNR in signal bandwidth={}dB and in 0.1nm={}dB'.format(osnr_ase[0], osnr_ase_01nm[0]))
        return snr 

    def __call__(self, spectral_info):
        return spectral_info

class Fiber(Node):
    def __init__(self, config):
        super().__init__(config)
        self.length = self.params.length * \
            UNITS[self.params.length_units]    #length in km
        self.loss_coef = self.params.loss_coef #lineic loss dB/km
        self.lin_loss_coef = self.params.loss_coef / 4.3429448190325184
        #TODO discuss factor 2 in the linear lineic attenuation

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid}, length={self.length})'

    def lin_attenuation(self):
        attenuation = self.length * self.loss_coef
        return db2lin(attenuation)

    def effective_length(self, loss_coef):
        alpha_dict = self.dbkm_2_lin(loss_coef)
        alpha = alpha_dict['alpha_acoef']
        leff = 1 - np.exp(-2 * alpha * self.span_length)
        return leff

    def asymptotic_length(self, loss_coef):
        alpha_dict = self.dbkm_2_lin(loss_coef)
        alpha = alpha_dict['alpha_acoef']
        aleff = 1 / (2 * alpha)
        return aleff

    def dbkm_2_lin(self, loss_coef):
        """ calculates the linear loss coefficient
        """
        alpha_pcoef = loss_coef
        alpha_acoef = alpha_pcoef / (2 * 4.3429448190325184)
        s = 'alpha_pcoef is linear loss coefficient in [dB/km^-1] units'
        s = ''.join([s, "alpha_acoef is linear loss field amplitude \
                     coefficient in [km^-1] units"])
        d = {'alpha_pcoef': alpha_pcoef,
             'alpha_acoef': alpha_acoef,
             'description:': s}
        return d

    def beta2(self, dispersion, ref_wavelength=None):
        """ Returns beta2 from dispersion parameter.  Dispersion is entered in
        ps/nm/km.  Disperion can be a numpy array or a single value.  If a
        value ref_wavelength is not entered 1550e-9m will be assumed.
        ref_wavelength can be a numpy array.
        """
        wl = 1550e-9 if ref_wavelength is None else ref_wavelength
        D = np.abs(dispersion)
        b2 = (10**21) * (wl**2) * D / (2 * np.pi * c)  # 10^21 scales [ps^2/km]
        return b2

    def propagate(self, *carriers):
        #TODO integrate and call the gn.ole module to calculate carrier nli noise in fiber
        carrier_nli = db2lin(-28)*1e-3 #temporary Cte nli noise is added for debug
        i=0
        for carrier in carriers:
            pwr = carrier.power
            pwr = pwr._replace(signal=pwr.signal/self.lin_attenuation(),
                               nonlinear_interference=(pwr.nli+carrier_nli)/self.lin_attenuation(),
                               amplified_spontaneous_emission=pwr.ase/self.lin_attenuation())
            i+=1
            yield carrier._replace(power=pwr)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        return spectral_info.update(carriers=carriers)


class Edfa(Node):
    def __init__(self, config):
        super().__init__(config)
        self.interpol_dgt = None #inerpolated dynamic gain tilt: N numpy array
        self.interpol_gain_ripple = None #gain ripple: N numpy array
        self.interpol_nf_ripple = None #nf_ripple: N numpy array
        self.channel_freq = None #SI channel frequencies: N numpy array
        """nf and gprofile attributs are set by interpol_params"""
        self.nf = None #edfa nf @ operational.gain_target: N numpy array 
        self.gprofile = None

    def interpol_params(self, frequencies, pin):
        """interpolate SI channel frequencies with the edfa dgt and gain_ripple frquencies from json
        set the edfa class __init__ None parameters :
                self.channel_freq, self.nf, self.interpol_dgt and self.interpol_gain_ripple
        """
        #TODO read amplifier actual frequencies from additional params in json
        amplifier_freq = itufs(0.05)*1e12 # Hz
        self.channel_freq = frequencies
        self.interpol_dgt = np.interp(self.channel_freq, amplifier_freq, self.params.dgt)
        self.interpol_gain_ripple = np.interp(self.channel_freq, amplifier_freq, self.params.gain_ripple)
        self.interpol_nf_ripple = np.interp(self.channel_freq, amplifier_freq, self.params.nf_ripple)

        """check power saturation and correct target_gain accordingly:"""
        tot_in_power_db = lin2db(np.sum(pin*1e3))
        gain_target = min(self.operational.gain_target, self.params.p_max-tot_in_power_db)
        self.operational.gain_target = gain_target

        self._calc_nf()
        self._gain_profile(pin)

    def _calc_nf(self):
        """nf calculation based on 2 models: self.params.nf_model.enabled from json import:
        True => 2 stages amp modelling based on precalculated nf1, nf2 and delta_p in build_OA_json
        False => polynomial fit based on self.params.nf_fit_coeff"""
        #TODO : tbd alarm rising or input VOA padding in case
        #gain_min > gain_target TBD:
        pad = max(self.params.gain_min - self.operational.gain_target, 0)
        gain_target = self.operational.gain_target + pad

        dg = gain_target - self.params.gain_flatmax # ! <0
        if self.params.nf_model.enabled:
            g1a = gain_target - self.params.nf_model.delta_p + dg
            nf_avg = lin2db(db2lin(self.params.nf_model.nf1) + db2lin(self.params.nf_model.nf2)/db2lin(g1a))
        else:
            nf_avg = np.polyval(self.params.nf_fit_coeff, dg)
        self.nf = self.interpol_nf_ripple + nf_avg + pad #input VOA = 1 for 1 NF degradation

    def noise_profile(self, bw):
        """ noise_profile(bw) computes amplifier ase (W) in signal bw (Hz) 
        noise is calculated at amplifier input

        :bw: signal bandwidth = baud rate in Hz
        :type bw: float

        :return: the asepower in W in the signal bandwidth bw for 96 channels
        :return type: numpy array of float

        ASE POWER USING PER CHANNEL GAIN PROFILE
        INPUTS:
        NF_dB - Noise figure in dB, vector of length number of channels or
                spectral slices
        G_dB  - Actual gain calculated for the EDFA, vector of length number of
                channels or spectral slices
        ffs     - Center frequency grid of the channels or spectral slices in
                THz, vector of length number of channels or spectral slices
        dF    - width of each channel or spectral slice in THz,
                vector of length number of channels or spectral slices
        OUTPUT:
            ase_dBm - ase in dBm per channel or spectral slice
        NOTE: the output is the total ASE in the channel or spectral slice. For
        50GHz channels the ASE BW is effectively 0.4nm. To get to noise power
        in 0.1nm, subtract 6dB.

        ONSR is usually quoted as channel power divided by
        the ASE power in 0.1nm RBW, regardless of the width of the actual
        channel.  This is a historical convention from the days when optical
        signals were much smaller (155Mbps, 2.5Gbps, ... 10Gbps) than the
        resolution of the OSAs that were used to measure spectral power which
        were set to 0.1nm resolution for convenience.  Moving forward into
        flexible grid and high baud rate signals, it may be convenient to begin
        quoting power spectral density in the same BW for both signal and ASE,
        e.g. 12.5GHz."""

        nchan = list(range(len(self.channel_freq)))
        df = np.array([bw]*(nchan[-1] + 1)) #Hz
        ase = h * df * self.channel_freq * db2lin(self.nf)  #W
        return ase #in W, @amplifier input 
        #checked 02/15/2018 @ 02:00pm -45dBm @ nf = 8.8dB in 32GHz

    def _gain_profile(self, pin):
        """
        Pin : input power / channel in W

        :param gain_ripple: design flat gain
        :param dgt: design gain tilt
        :param Pin: total input power in W
        :param gp: Average gain setpoint in dB units
        :param gtp: gain tilt setting
        :type gain_ripple: numpy.ndarray
        :type dgt: numpy.ndarray
        :type Pin: numpy.ndarray
        :type gp: float
        :type gtp: float
        :return: gain profile in dBm
        :rtype: numpy.ndarray

        AMPLIFICATION USING INPUT PROFILE
        INPUTS:
            gain_ripple - vector of length number of channels or spectral slices
            DGT - vector of length number of channels or spectral slices
            Pin - input powers vector of length number of channels or
            spectral slices
            Gp  - provisioned gain length 1
            GTp - provisioned tilt length 1

        OUTPUT:
            amp gain per channel or spectral slice
        NOTE: there is no checking done for violations of the total output
            power capability of the amp.
        EDIT OF PREVIOUS NOTE: power violation now added in interpol_params
            Ported from Matlab version written by David Boerges at Ciena.
        Based on:
            R. di Muro, "The Er3+ fiber gain coefficient derived from a dynamic
            gain
            tilt technique", Journal of Lightwave Technology, Vol. 18, Iss. 3,
            Pp. 343-347, 2000.
        """
        err_tolerance = 1.0e-11
        simple_opt = True

        # TODO check what param should be used (currently length(dgt))
        nchan = np.arange(len(self.interpol_dgt))

        # TODO find a way to use these or lose them.  Primarily we should have
        # a way to determine if exceeding the gain or output power of the amp
        tot_in_power_db = lin2db(np.sum(pin*1e3)) # ! Pin expressed in W

        # Linear fit to get the
        p = np.polyfit(nchan, self.interpol_dgt, 1)
        dgt_slope = p[0]

        # Calculate the target slope-  Currently assumes equal spaced channels
        # TODO make it so that supports arbitrary channel spacing.
        targ_slope = self.operational.tilt_target / (len(nchan) - 1)

        # 1st estimate of DGT scaling
        dgts1 = targ_slope / dgt_slope
        # when simple_opt is true code makes 2 attempts to compute gain and
        # the internal voa value.  This is currently here to provide direct
        # comparison with original Matlab code.  Will be removed.
        # TODO replace with loop

        if simple_opt:

            # 1st estimate of Er gain & voa loss
            g1st = np.array(self.interpol_gain_ripple) + self.params.gain_flatmax + \
                np.array(self.interpol_dgt) * dgts1
            voa = lin2db(np.mean(db2lin(g1st))) - self.operational.gain_target

            # 2nd estimate of Amp ch gain using the channel input profile
            g2nd = g1st - voa
            pout_db = lin2db(np.sum(pin*1e3*db2lin(g2nd)))
            dgts2 = self.operational.gain_target - (pout_db - tot_in_power_db)

            # Center estimate of amp ch gain
            xcent = dgts2
            gcent = g1st - voa + np.array(self.interpol_dgt) * xcent
            pout_db = lin2db(np.sum(pin*1e3*db2lin(gcent)))
            gavg_cent = pout_db - tot_in_power_db

            # Lower estimate of Amp ch gain
            deltax = np.max(g1st) - np.min(g1st)
            xlow = dgts2 - deltax
            glow = g1st - voa + np.array(self.interpol_dgt) * xlow
            pout_db = lin2db(np.sum(pin*1e3*db2lin(glow)))
            gavg_low = pout_db - tot_in_power_db

            # Upper gain estimate
            xhigh = dgts2 + deltax
            ghigh = g1st - voa + np.array(self.interpol_dgt) * xhigh
            pout_db = lin2db(np.sum(pin*1e3*db2lin(ghigh)))        
            gavg_high = pout_db - tot_in_power_db

            # compute slope
            slope1 = (gavg_low - gavg_cent) / (xlow - xcent)
            slope2 = (gavg_cent - gavg_high) / (xcent - xhigh)

            if np.abs(self.operational.gain_target - gavg_cent) <= err_tolerance:
                dgts3 = xcent
            elif self.operational.gain_target < gavg_cent:
                dgts3 = xcent - (gavg_cent - self.operational.gain_target) / slope1
            else:
                dgts3 = xcent + (-gavg_cent + self.operational.gain_target) / slope2

            gprofile = g1st - voa + np.array(self.interpol_dgt) * dgts3
            #print(gprofile[0])
        else:
            gprofile = None

        self.gprofile = gprofile

    def propagate(self, *carriers):
        """add ase noise to the propagating carriers of SpectralInformation"""
        i = 0
        pin = np.array([c.power.signal+c.power.nli+c.power.ase for c in carriers]) #pin in W
        freq = np.array([c.frequency for c in carriers])
        #interpolate the amplifier vectors with the carriers freq, calculate nf & gain profile
        self.interpol_params(freq, pin) 
        gain = db2lin(self.gprofile)

        for carrier in carriers:

            pwr = carrier.power
            bw = carrier.baud_rate            
            carrier_ase = self.noise_profile(bw)[i]

            pwr = pwr._replace(signal=pwr.signal*gain[i],
                               nonlinear_interference=pwr.nli*gain[i],
                               amplified_spontaneous_emission=(pwr.ase+carrier_ase)*gain[i])
            i += 1
            yield carrier._replace(power=pwr)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        return spectral_info.update(carriers=carriers)
