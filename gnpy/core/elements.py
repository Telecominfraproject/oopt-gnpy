#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.constants import c, h

from gnpy.core.node import Node
from gnpy.core.units import UNITS
from gnpy.core.utils import lin2db, db2lin


class Transceiver(Node):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, spectral_info):
        return spectral_info


class Fiber(Node):
    def __init__(self, config):
        super().__init__(config)
        self.length = self.params.length * \
            UNITS[self.params.length_units]

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid}, length={self.length})'

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
        for carrier in carriers:
            pwr = carrier.power
            pwr = pwr._replace(signal=0.5 * pwr.signal * .5,
                               nonlinear_interference=2 * pwr.nli,
                               amplified_spontaneous_emission=2 * pwr.ase)
            yield carrier._replace(power=pwr)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        return spectral_info.update(carriers=carriers)


class Edfa(Node):
    def __init__(self, config):
        super().__init__(config)
        self.gain_target = None
        self.tilt_target = None
        self.nf = None

    def noise_profile(self, gain, ffs, df):
        """ noise_profile(nf, gain, ffs, df) computes amplifier ase

        :param nf: Noise figure in dB
        :param gain: Actual gain calculated for the EDFA in dB units
        :param ffs: A numpy array of frequencies
        :param df: the reference bw in THz
        :type nf: numpy.ndarray
        :type gain: numpy.ndarray
        :type ffs: numpy.ndarray
        :type df: float
        :return: the asepower in dBm
        :rtype: numpy.ndarray

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

        h_mWThz = 1e-3 * h * (1e14)**2
        nf_lin = db2lin(self.nf)
        g_lin = db2lin(gain)
        ase = h_mWThz * df * ffs * (nf_lin * g_lin - 1)
        asedb = lin2db(ase)

        return asedb

    def gain_profile(self, Pin):
        """
        :param dfg: design flat gain
        :param dgt: design gain tilt
        :param Pin: channing input power profile
        :param gp: Average gain setpoint in dB units
        :param gtp: gain tilt setting
        :type dfg: numpy.ndarray
        :type dgt: numpy.ndarray
        :type Pin: numpy.ndarray
        :type gp: float
        :type gtp: float
        :return: gain profile in dBm
        :rtype: numpy.ndarray

        AMPLIFICATION USING INPUT PROFILE
        INPUTS:
            DFG - vector of length number of channels or spectral slices
            DGT - vector of length number of channels or spectral slices
            Pin - input powers vector of length number of channels or
            spectral slices
            Gp  - provisioned gain length 1
            GTp - provisioned tilt length 1

        OUTPUT:
            amp gain per channel or spectral slice
        NOTE: there is no checking done for violations of the total output
            power capability of the amp.
            Ported from Matlab version written by David Boerges at Ciena.
        Based on:
            R. di Muro, "The Er3+ fiber gain coefficient derived from a dynamic
            gain
            tilt technique", Journal of Lightwave Technology, Vol. 18, Iss. 3,
            Pp. 343-347, 2000.
        """
        err_tolerance = 1.0e-11
        simple_opt = True

        # TODO make all values linear unit and convert to dB units as needed
        # within this function.
        nchan = np.arange(len(Pin))

        # TODO find a way to use these or lose them.  Primarily we should have
        # a way to determine if exceeding the gain or output power of the amp
        tot_in_power_db = lin2db(np.sum(db2lin(Pin)))

        # Linear fit to get the
        p = np.polyfit(nchan, self.params.dgt, 1)
        dgt_slope = p[0]

        # Calculate the target slope-  Currently assumes equal spaced channels
        # TODO make it so that supports arbitrary channel spacing.
        targ_slope = self.tilt_target / (len(nchan) - 1)

        # 1st estimate of DGT scaling
        dgts1 = targ_slope / dgt_slope

        # when simple_opt is true code makes 2 attempts to compute gain and
        # the internal voa value.  This is currently here to provide direct
        # comparison with original Matlab code.  Will be removed.
        # TODO replace with loop

        if simple_opt:

            # 1st estimate of Er gain & voa loss
            g1st = np.array(self.params.dfg) + \
                np.array(self.params.dgt) * dgts1
            voa = lin2db(np.mean(db2lin(g1st))) - self.gain_target

            # 2nd estimate of Amp ch gain using the channel input profile
            g2nd = g1st - voa
            pout_db = lin2db(np.sum(db2lin(Pin + g2nd)))
            dgts2 = self.gain_target - (pout_db - tot_in_power_db)

            # Center estimate of amp ch gain
            xcent = dgts2
            gcent = g1st - voa + np.array(self.params.dgt) * xcent
            pout_db = lin2db(np.sum(db2lin(Pin + gcent)))
            gavg_cent = pout_db - tot_in_power_db

            # Lower estimate of Amp ch gain
            deltax = np.max(g1st) - np.min(g1st)
            xlow = dgts2 - deltax
            glow = g1st - voa + np.array(self.params.dgt) * xlow
            pout_db = lin2db(np.sum(db2lin(Pin + glow)))
            gavg_low = pout_db - tot_in_power_db

            # Upper gain estimate
            xhigh = dgts2 + deltax
            ghigh = g1st - voa + np.array(self.params.dgt) * xhigh
            pout_db = lin2db(np.sum(db2lin(Pin + ghigh)))
            gavg_high = pout_db - tot_in_power_db

            # compute slope
            slope1 = (gavg_low - gavg_cent) / (xlow - xcent)
            slope2 = (gavg_cent - gavg_high) / (xcent - xhigh)

            if np.abs(self.gain_target - gavg_cent) <= err_tolerance:
                dgts3 = xcent
            elif self.gain_target < gavg_cent:
                dgts3 = xcent - (gavg_cent - self.gain_target) / slope1
            else:
                dgts3 = xcent + (-gavg_cent + self.gain_target) / slope2

            gprofile = g1st - voa + np.array(self.params.dgt) * dgts3
        else:
            gprofile = None

        return gprofile

    def calc_nf(self):
        dg = self.gain_target - np.mean(self.params.dfg)
        nf_avg = np.polyval(self.params.nf_fit_coeff, dg)
        self.nf = self.params.nf_ripple + nf_avg
