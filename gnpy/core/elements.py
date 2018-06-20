#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
gnpy.core.elements
==================

This module contains standard network elements.

A network element is a Python callable. It takes a .info.SpectralInformation
object and returns a copy with appropriate fields affected. This structure
represents spectral information that is "propogated" by this network element.
Network elements must have only a local "view" of the network and propogate
SpectralInformation using only this information. They should be independent and
self-contained.

Network elements MUST implement two attributes .uid and .name representing a
unique identifier and a printable name.
'''

from numpy import abs, arange, arcsinh, array, exp
from numpy import interp, log10, mean, pi, polyfit, polyval, sum
from scipy.constants import c, h
from collections import namedtuple

from gnpy.core.node import Node
from gnpy.core.units import UNITS
from gnpy.core.utils import lin2db, db2lin, itufs

class Transceiver(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.osnr_ase_01nm = None
        self.osnr_ase = None
        self.osnr_nli = None
        self.snr = None
        self.passive = False

    def _calc_snr(self, spectral_info):
        ase = [c.power.ase for c in spectral_info.carriers]
        nli = [c.power.nli for c in spectral_info.carriers]
        if min(ase) > 1e-20:
            self.osnr_ase = [lin2db(c.power.signal/c.power.ase)
                             for c in spectral_info.carriers]
            ratio_01nm = [lin2db(12.5e9/c.baud_rate)
                          for c in spectral_info.carriers]
            self.osnr_ase_01nm = [ase - ratio for ase, ratio
                                  in zip(self.osnr_ase, ratio_01nm)]
        if min(nli) > 1e-20:
            self.osnr_nli = [lin2db(c.power.signal/c.power.nli)
                             for c in spectral_info.carriers]
            self.snr = [lin2db(c.power.signal/(c.power.nli+c.power.ase))
                        for c in spectral_info.carriers]

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'uid={self.uid!r}, '
                f'osnr_ase_01nm={self.osnr_ase_01nm!r}, '
                f'osnr_ase={self.osnr_ase!r}, '
                f'osnr_nli={self.osnr_nli!r}, '
                f'snr={self.snr!r})')

    def __str__(self):
        if self.snr is None or self.osnr_ase is None:
            return f'{type(self).__name__} {self.uid}'

        snr = round(mean(self.snr),2)
        osnr_ase = round(mean(self.osnr_ase),2)
        osnr_ase_01nm = round(mean(self.osnr_ase_01nm), 2)

        return '\n'.join([f'{type(self).__name__} {self.uid}',

                          f'  OSNR ASE (1nm):        {osnr_ase_01nm:.2f}',
                          f'  OSNR ASE (signal bw):  {osnr_ase:.2f}',
                          f'  SNR total (signal bw): {snr:.2f}'])


    def __call__(self, spectral_info):
        self._calc_snr(spectral_info)
        return spectral_info

RoadmParams = namedtuple('RoadmParams', 'loss')

class Roadm(Node):
    def __init__(self, *args, params=None, **kwargs):
        if params is None:
            # default loss value if not mentioned in loaded network json
            params = {'loss':20}
        super().__init__(*args, params=RoadmParams(**params), **kwargs)
        self.loss = self.params.loss
        self.passive = True

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid!r}, loss={self.loss!r})'

    def __str__(self):
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  loss (dB): {self.loss:.2f}'])

    def propagate(self, *carriers):
        attenuation = db2lin(self.loss)

        for carrier in carriers:
            pwr = carrier.power
            pwr = pwr._replace(signal=pwr.signal/attenuation,
                               nonlinear_interference=pwr.nli/attenuation,
                               amplified_spontaneous_emission=pwr.ase/attenuation)
            yield carrier._replace(power=pwr)

    def update_pref(self, pref):
        return pref._replace(p_span0=pref.p0, p_spani=pref.pi - self.loss)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref)
        print('pi p0',pref.pi, pref.p0)
        return spectral_info.update(carriers=carriers, pref=pref)

FusedParams = namedtuple('FusedParams', 'loss')

class Fused(Node):
    def __init__(self, *args, params=None, **kwargs):
        if params is None:
            # default loss value if not mentioned in loaded network json
            params = {'loss':1}
        super().__init__(*args, params=FusedParams(**params), **kwargs)
        self.loss = self.params.loss
        self.passive = True

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid!r}, loss={self.loss!r})'

    def __str__(self):
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  loss (dB): {self.loss:.2f}'])

    def propagate(self, *carriers):
        attenuation = db2lin(self.loss)

        for carrier in carriers:
            pwr = carrier.power
            pwr = pwr._replace(signal=pwr.signal/attenuation,
                               nonlinear_interference=pwr.nli/attenuation,
                               amplified_spontaneous_emission=pwr.ase/attenuation)
            yield carrier._replace(power=pwr)
    
    def update_pref(self, pref):
        return pref._replace(p_span0=pref.p0, p_spani=pref.pi - self.loss)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref)
        print('pi',pref.pi)
        return spectral_info.update(carriers=carriers, pref=pref)

FiberParams = namedtuple('FiberParams', 'type_variety length loss_coef length_units dispersion gamma')

class Fiber(Node):
    def __init__(self, *args, params=None, **kwargs):
        if params is None:
            params = {}
        super().__init__(*args, params=FiberParams(**params), **kwargs)
        self.type_variety = self.params.type_variety
        self.length = self.params.length * UNITS[self.params.length_units] # in m
        self.loss_coef = self.params.loss_coef * 1e-3 # lineic loss dB/m
        self.lin_loss_coef = self.params.loss_coef / (20 * log10(exp(1)))
        self.dispersion = self.params.dispersion  # s/m/m
        self.gamma = self.params.gamma # 1/W/m       
        # TODO|jla: discuss factor 2 in the linear lineic attenuation


    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid!r}, length={self.length!r}, loss={self.loss!r})'

    def __str__(self):
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  type_variety: {self.type_variety}',
                          f'  length (m):   {self.length:.2f}',
                          f'  loss (dB):    {self.loss:.2f}'])

    @property
    def loss(self):
        # dB loss: useful for polymorphism (roadm, fiber, att)
        return self.loss_coef * self.length

    @property
    def passive(self):
        return True   

    @property
    def lin_attenuation(self):
        return db2lin(self.length * self.loss_coef)

    @property
    def effective_length(self):
        _, alpha = self.dbkm_2_lin()
        leff = (1 - exp(-2 * alpha * self.length)) / (2 * alpha)
        return leff

    @property
    def asymptotic_length(self):
        _, alpha = self.dbkm_2_lin()
        aleff = 1 / (2 * alpha)
        return aleff

    def beta2(self, ref_wavelength=None):
        """ Returns beta2 from dispersion parameter.
        Dispersion is entered in ps/nm/km.
        Disperion can be a numpy array or a single value.  If a
        value ref_wavelength is not entered 1550e-9m will be assumed.
        ref_wavelength can be a numpy array.
        """
        # TODO|jla: discuss beta2 as method or attribute
        wl = 1550e-9 if ref_wavelength is None else ref_wavelength
        D = abs(self.dispersion)
        b2 = (wl ** 2) * D / (2 * pi * c)  # 10^21 scales [ps^2/km]
        return b2 # s/Hz/m

    def dbkm_2_lin(self):
        """ calculates the linear loss coefficient
        """
        # alpha_pcoef is linear loss coefficient in dB/km^-1
        # alpha_acoef is linear loss field amplitude coefficient in m^-1
        alpha_pcoef = self.loss_coef
        alpha_acoef = alpha_pcoef / (2 * 10 * log10(exp(1)))
        return alpha_pcoef, alpha_acoef

    def _psi(self, carrier, interfering_carrier):
        """ Calculates eq. 123 from	arXiv:1209.0394.
        """
        if carrier.num_chan == interfering_carrier.num_chan: # SCI
            psi = arcsinh(0.5 * pi**2 * self.asymptotic_length
                              * abs(self.beta2()) * carrier.baud_rate**2)
        else: # XCI
            delta_f = carrier.freq - interfering_carrier.freq
            psi = arcsinh(pi**2 * self.asymptotic_length * abs(self.beta2())
                                * carrier.baud_rate * (delta_f + 0.5 * interfering_carrier.baud_rate))
            psi -= arcsinh(pi**2 * self.asymptotic_length * abs(self.beta2())
                                 * carrier.baud_rate * (delta_f - 0.5 * interfering_carrier.baud_rate))

        return psi

    def _gn_analytic(self, carrier, *carriers):
        """ Computes the nonlinear interference power on a single carrier.
        The method uses eq. 120 from arXiv:1209.0394.
        :param carrier: the signal under analysis
        :param carriers: the full WDM comb
        :return: carrier_nli: the amount of nonlinear interference in W on the under analysis
        """

        g_nli = 0
        for interfering_carrier in carriers:
            psi = self._psi(carrier, interfering_carrier)
            g_nli += (interfering_carrier.power.signal/interfering_carrier.baud_rate)**2 \
                     * (carrier.power.signal/carrier.baud_rate) * psi

        g_nli *= (16 / 27) * (self.gamma * self.effective_length)**2 \
                 / (2 * pi * abs(self.beta2()) * self.asymptotic_length)

        carrier_nli = carrier.baud_rate * g_nli
        return carrier_nli

    def propagate(self, *carriers):
        for carrier in carriers:
            pwr = carrier.power
            carrier_nli = self._gn_analytic(carrier, *carriers)
            pwr = pwr._replace(signal=pwr.signal/self.lin_attenuation,
                               nonlinear_interference=(pwr.nli+carrier_nli)/self.lin_attenuation,
                               amplified_spontaneous_emission=pwr.ase/self.lin_attenuation)
            yield carrier._replace(power=pwr)

    def update_pref(self, pref):
        return pref._replace(p_span0=pref.p0, p_spani=pref.pi - self.loss)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref)
        print('after fiber pi',pref.pi)
        return spectral_info.update(carriers=carriers, pref=pref)

# TODO|dutc: eliminate duplication with .equipment.EdfaBase
EdfaParams = namedtuple('EdfaParams',
    'type_variety, gain_flatmax gain_min p_max nf_min nf_max'
    ' nf_model nf_fit_coeff nf_ripple dgt gain_ripple')
class EdfaOperational:
    def __init__(self, gain_target, tilt_target):
        self.gain_target = gain_target
        self.tilt_target = tilt_target
    def __repr__(self):
        return (f'{type(self).__name__}('
                f'gain_target={self.gain_target!r}, '
                f'tilt_target={self.tilt_target!r})')

class Edfa(Node):
    def __init__(self, *args, params=None, operational=None, **kwargs):
        if params is None:
            params = {}
        if operational is None:
            operational = {}
        super().__init__(
            *args,
            params=EdfaParams(**params),
            operational=EdfaOperational(**operational),
            **kwargs
        )
        self.interpol_dgt = None # interpolated dynamic gain tilt
        self.interpol_gain_ripple = None # gain ripple
        self.interpol_nf_ripple = None # nf_ripple
        self.channel_freq = None # SI channel frequencies
        # nf, gprofile, pin and pout attributes are set by interpol_params
        self.nf = None # dB edfa nf at operational.gain_target
        self.gprofile = None
        self.pin_db = None
        self.pout_db = None
        self.dp_db = None #delta P with Pref (power swwep) in power mode
        self.pref_db = None
        self.passive = False
        self.effective_gain = self.operational.gain_target

    def __repr__(self):
        return (f'{type(self).__name__}(uid={self.uid!r}, '
                f'type_variety={self.params.type_variety!r}'
                f'interpol_dgt={self.interpol_dgt!r}, '
                f'interpol_gain_ripple={self.interpol_gain_ripple!r}, '
                f'interpol_nf_ripple={self.interpol_nf_ripple!r}, '
                f'channel_freq={self.channel_freq!r}, '
                f'nf={self.nf!r}, '
                f'gprofile={self.gprofile!r}, '
                f'pin_db={self.pin_db!r}, '
                f'pout_db={self.pout_db!r})')

    def __str__(self):
        if self.pin_db is None or self.pout_db is None:
            return f'{type(self).__name__} {self.uid}'
        nf = mean(self.nf)
        return '\n'.join([f'{type(self).__name__} {self.uid}', 
                          f'  type_variety:           {self.params.type_variety}',
                          f'  operational gain (dB):  {self.operational.gain_target:.2f}',
                          f'  effective gain(dB):     {self.effective_gain}',
                          f'  noise figure (dB):      {nf:.2f}',
                          f'  Power In (dBm):         {self.pin_db:.2f}',
                          f'  Power Out (dBm):        {self.pout_db:.2f}',
                          f'  Delta_P (dB):           {self.dp_db!r}',
                          f'  Pref (dBm):             {self.pref_db!r}'])

    def interpol_params(self, frequencies, pin, baud_rates, pref):
        """interpolate SI channel frequencies with the edfa dgt and gain_ripple frquencies from json
        set the edfa class __init__ None parameters :
                self.channel_freq, self.nf, self.interpol_dgt and self.interpol_gain_ripple
        """
        # TODO|jla: read amplifier actual frequencies from additional params in json
        amplifier_freq = itufs(0.05) * 1e12 # Hz
        self.channel_freq = frequencies
        self.interpol_dgt = interp(self.channel_freq, amplifier_freq, self.params.dgt)
        self.interpol_gain_ripple = interp(self.channel_freq, amplifier_freq, self.params.gain_ripple)
        self.interpol_nf_ripple =interp(self.channel_freq, amplifier_freq, self.params.nf_ripple)

        self.pin_db = lin2db(sum(pin*1e3))
        """check power saturation and correct target_gain accordingly:"""

        if self.dp_db is not None:
            print('interpolate pi', pref.pi)
            print('interpolate dp', self.dp_db)
            self.effective_gain = self.dp_db + pref.p0 - pref.pi
            self.pref_db = self.dp_db + pref.p0
        else:
            self.effective_gain = self.operational.gain_target
        self.effective_gain = min(self.effective_gain, self.params.p_max - self.pin_db)

        self.nf = self._calc_nf()
        self.gprofile = self._gain_profile(pin)

        pout = (pin + self.noise_profile(baud_rates))*db2lin(self.gprofile)
        self.pout_db = lin2db(sum(pout*1e3))
        # ase & nli are only calculated in signal bandwidth
        #    pout_db is not the absolute full output power (negligible if sufficient channels)

    def _calc_nf(self):
        """nf calculation based on 2 models: self.params.nf_model.enabled from json import:
        True => 2 stages amp modelling based on precalculated nf1, nf2 and delta_p in build_OA_json
        False => polynomial fit based on self.params.nf_fit_coeff"""
        # TODO|jla: TBD alarm rising or input VOA padding in case
        # gain_min > gain_target TBD:
        pad = max(self.params.gain_min - self.effective_gain, 0)
        gain_target = self.effective_gain + pad
        dg = gain_target - self.params.gain_flatmax
        if self.params.nf_model:
            g1a = gain_target - self.params.nf_model.delta_p + dg
            nf_avg = lin2db(db2lin(self.params.nf_model.nf1) + db2lin(self.params.nf_model.nf2)/db2lin(g1a))
        else:
            nf_avg = polyval(self.params.nf_fit_coeff, dg)
        return self.interpol_nf_ripple + nf_avg + pad # input VOA = 1 for 1 NF degradation

    def noise_profile(self, df):
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

        ase = h * df * self.channel_freq * db2lin(self.nf) # W
        return ase # in W at amplifier input

    def _gain_profile(self, pin, err_tolerance=1.0e-11, simple_opt=True):
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

        # TODO|jla: check what param should be used (currently length(dgt))
        nchan = arange(len(self.interpol_dgt))

        # TODO|jla: find a way to use these or lose them. Primarily we should have
        # a way to determine if exceeding the gain or output power of the amp
        tot_in_power_db = lin2db(sum(pin*1e3)) # Pin in W

        # linear fit to get the
        p = polyfit(nchan, self.interpol_dgt, 1)
        dgt_slope = p[0]

        # Calculate the target slope - currently assumes equal spaced channels
        # TODO|jla: support arbitrary channel spacing
        targ_slope = self.operational.tilt_target / (len(nchan) - 1)

        # first estimate of DGT scaling
        if abs(dgt_slope) > 0.001: # check for zero value due to flat dgt
            dgts1 = targ_slope / dgt_slope
        else:
            dgts1 = 0

        # when simple_opt is true, make 2 attempts to compute gain and
        # the internal voa value. This is currently here to provide direct
        # comparison with original Matlab code. Will be removed.
        # TODO|jla: replace with loop

        if not simple_opt:
            return

        # first estimate of Er gain & VOA loss
        g1st = array(self.interpol_gain_ripple) + self.params.gain_flatmax \
               + array(self.interpol_dgt) * dgts1
        voa = lin2db(mean(db2lin(g1st))) - self.effective_gain

        # second estimate of amp ch gain using the channel input profile
        g2nd = g1st - voa

        pout_db = lin2db(sum(pin*1e3*db2lin(g2nd)))
        dgts2 = self.effective_gain - (pout_db - tot_in_power_db)

        # center estimate of amp ch gain
        xcent = dgts2
        gcent = g1st - voa + array(self.interpol_dgt) * xcent
        pout_db = lin2db(sum(pin*1e3*db2lin(gcent)))
        gavg_cent = pout_db - tot_in_power_db

        # Lower estimate of amp ch gain
        deltax = max(g1st) - min(g1st)
        # if no ripple deltax = 0 and xlow = xcent: div 0
        # TODO|jla: add check for flat gain response
        if abs(deltax) <= 0.05: # not enough ripple to consider calculation
            return g1st - voa

        xlow = dgts2 - deltax
        glow = g1st - voa + array(self.interpol_dgt) * xlow
        pout_db = lin2db(sum(pin * 1e3 * db2lin(glow)))
        gavg_low = pout_db - tot_in_power_db

        # upper gain estimate
        xhigh = dgts2 + deltax
        ghigh = g1st - voa + array(self.interpol_dgt) * xhigh
        pout_db = lin2db(sum(pin * 1e3 * db2lin(ghigh)))
        gavg_high = pout_db - tot_in_power_db

        # compute slope
        slope1 = (gavg_low - gavg_cent) / (xlow - xcent)
        slope2 = (gavg_cent - gavg_high) / (xcent - xhigh)

        if abs(self.effective_gain - gavg_cent) <= err_tolerance:
            dgts3 = xcent
        elif self.effective_gain < gavg_cent:
            dgts3 = xcent - (gavg_cent - self.effective_gain) / slope1
        else:
            dgts3 = xcent + (-gavg_cent + self.effective_gain) / slope2

        return g1st - voa + array(self.interpol_dgt) * dgts3

    def propagate(self, pref, *carriers):
        """add ase noise to the propagating carriers of SpectralInformation"""
        i = 0
        pin = array([c.power.signal+c.power.nli+c.power.ase for c in carriers]) # pin in W
        freq = array([c.frequency for c in carriers])
        brate = array([c.baud_rate for c in carriers])
        # interpolate the amplifier vectors with the carriers freq, calculate nf & gain profile
        self.interpol_params(freq, pin, brate, pref)

        gains = db2lin(self.gprofile)
        carrier_ases = self.noise_profile(brate)

        for gain, carrier_ase, carrier in zip(gains, carrier_ases, carriers):
            pwr = carrier.power
            bw = carrier.baud_rate
            pwr = pwr._replace(signal=pwr.signal*gain,
                               nonlinear_interference=pwr.nli*gain,
                               amplified_spontaneous_emission=(pwr.ase+carrier_ase)*gain)
            yield carrier._replace(power=pwr)

    def update_pref(self, pref):
        return pref._replace(p_span0=pref.p0, p_spani=pref.pi + self.effective_gain)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(spectral_info.pref, *spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref)
        print('after edfa pi',pref.pi)
        return spectral_info.update(carriers=carriers, pref=pref)
