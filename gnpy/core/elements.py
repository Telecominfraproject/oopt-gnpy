#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.elements
==================

Standard network elements which propagate optical spectrum

A network element is a Python callable. It takes a :class:`.info.SpectralInformation`
object and returns a copy with appropriate fields affected. This structure
represents spectral information that is "propogated" by this network element.
Network elements must have only a local "view" of the network and propogate
:class:`.info.SpectralInformation` using only this information. They should be independent and
self-contained.

Network elements MUST implement two attributes :py:attr:`uid` and :py:attr:`name` representing a
unique identifier and a printable name, and provide the :py:meth:`__call__` method taking a
:class:`SpectralInformation` as an input and returning another :class:`SpectralInformation`
instance as a result.
"""

from numpy import abs, array, errstate, ones, interp, mean, pi, polyfit, polyval, sum, sqrt, log10, exp, asarray, full,\
    squeeze, zeros, append, flip, outer, minimum
from scipy.constants import h, c
from scipy.interpolate import interp1d
from collections import namedtuple
from copy import deepcopy

from gnpy.core.utils import lin2db, db2lin, arrange_frequencies, snr_sum, per_label_summary, pretty_summary_print,\
    watt2dbm, psd2powerdbm, psdmwperghz, powerdbm2psdmwperghz
from gnpy.core.parameters import RoadmParams, FusedParams, FiberParams, PumpParams, EdfaParams, EdfaOperational
from gnpy.core.science_utils import NliSolver, RamanSolver
from gnpy.core.info import SpectralInformation
from gnpy.core import ansi_escapes
from gnpy.core.exceptions import NetworkTopologyError


class Location(namedtuple('Location', 'latitude longitude city region')):
    def __new__(cls, latitude=0, longitude=0, city=None, region=None):
        return super().__new__(cls, latitude, longitude, city, region)


class _Node:
    '''Convenience class for providing common functionality of all network elements

    This class is just an internal implementation detail; do **not** assume that all network elements
    inherit from :class:`_Node`.
    '''
    def __init__(self, uid, name=None, params=None, metadata=None, operational=None, type_variety=None):
        if name is None:
            name = uid
        self.uid, self.name = uid, name
        if metadata is None:
            metadata = {'location': {}}
        if metadata and not isinstance(metadata.get('location'), Location):
            metadata['location'] = Location(**metadata.pop('location', {}))
        self.params, self.metadata, self.operational = params, metadata, operational
        if type_variety:
            self.type_variety = type_variety

    @property
    def location(self):
        return self.metadata['location']
    loc = location

    @property
    def longitude(self):
        return self.location.longitude
    lng = longitude

    @property
    def latitude(self):
        return self.location.latitude
    lat = latitude


class Transceiver(_Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.osnr_ase_01nm = None
        self.osnr_ase = None
        self.osnr_nli = None
        self.snr = None
        self.passive = False
        self.baud_rate = None
        self.chromatic_dispersion = None
        self.pmd = None

    def _calc_cd(self, spectral_info):
        """ Updates the Transceiver property with the CD of the received channels. CD in ps/nm.
        """
        self.chromatic_dispersion = spectral_info.chromatic_dispersion * 1e3

    def _calc_pmd(self, spectral_info):
        """Updates the Transceiver property with the PMD of the received channels. PMD in ps.
        """
        self.pmd = spectral_info.pmd * 1e12

    def _calc_snr(self, spectral_info):
        with errstate(divide='ignore'):
            self.label = spectral_info.label
            self.baud_rate = spectral_info.baud_rate
            ratio_01nm = lin2db(12.5e9 / self.baud_rate)
            # set raw values to record original calculation, before update_snr()
            self.raw_osnr_ase = lin2db(spectral_info.signal / spectral_info.ase)
            self.raw_osnr_ase_01nm = self.raw_osnr_ase - ratio_01nm
            self.raw_osnr_nli = lin2db(spectral_info.signal / spectral_info.nli)
            self.raw_snr = lin2db(spectral_info.signal / (spectral_info.ase + spectral_info.nli))
            self.raw_snr_01nm = self.raw_snr - ratio_01nm

            self.osnr_ase = self.raw_osnr_ase
            self.osnr_ase_01nm = self.raw_osnr_ase_01nm
            self.osnr_nli = self.raw_osnr_nli
            self.snr = self.raw_snr
            self.snr_01nm = self.raw_snr_01nm

    def update_snr(self, *args):
        """
        snr_added in 0.1nm
        compute SNR penalties such as transponder Tx_osnr or Roadm add_drop_osnr
        only applied in request.py / propagate on the last Trasceiver node of the path
        all penalties are added in a single call because to avoid uncontrolled cumul
        """
        # use raw_values so that the added SNR penalties are not cumulated
        snr_added = 0
        for s in args:
            snr_added += db2lin(-s)
        snr_added = -lin2db(snr_added)
        self.osnr_ase = snr_sum(self.raw_osnr_ase, self.baud_rate, snr_added)
        self.snr = snr_sum(self.raw_snr, self.baud_rate, snr_added)
        self.osnr_ase_01nm = snr_sum(self.raw_osnr_ase_01nm, 12.5e9, snr_added)
        self.snr_01nm = snr_sum(self.raw_snr_01nm, 12.5e9, snr_added)

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'metadata': {
                    'location': self.metadata['location']._asdict()
                }
                }

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'uid={self.uid!r}, '
                f'osnr_ase_01nm={self.osnr_ase_01nm!r}, '
                f'osnr_ase={self.osnr_ase!r}, '
                f'osnr_nli={self.osnr_nli!r}, '
                f'snr={self.snr!r}, '
                f'chromatic_dispersion={self.chromatic_dispersion!r}, '
                f'pmd={self.pmd!r})')

    def __str__(self):
        if self.snr is None or self.osnr_ase is None:
            return f'{type(self).__name__} {self.uid}'

        snr = per_label_summary(self.snr, self.label)
        osnr_ase = per_label_summary(self.osnr_ase, self.label)
        osnr_ase_01nm = per_label_summary(self.osnr_ase_01nm, self.label)
        snr_01nm = per_label_summary(self.snr_01nm, self.label)
        cd = mean(self.chromatic_dispersion)
        pmd = mean(self.pmd)

        return '\n'.join([f'{type(self).__name__} {self.uid}',

                          f'  GSNR (0.1nm, dB):          {pretty_summary_print(snr_01nm)}',
                          f'  GSNR (signal bw, dB):      {pretty_summary_print(snr)}',
                          f'  OSNR ASE (0.1nm, dB):      {pretty_summary_print(osnr_ase_01nm)}',
                          f'  OSNR ASE (signal bw, dB):  {pretty_summary_print(osnr_ase)}',
                          f'  CD (ps/nm):                {cd:.2f}',
                          f'  PMD (ps):                  {pmd:.2f}'])

    def __call__(self, spectral_info):
        self._calc_snr(spectral_info)
        self._calc_cd(spectral_info)
        self._calc_pmd(spectral_info)
        return spectral_info


class Roadm(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if not params:
            params = {}
        super().__init__(*args, params=RoadmParams(**params), **kwargs)
        # target power as defined by user
        self.target_pch_out_dbm = self.params.target_pch_out_db
        # reference power is target power by default. depending on propagation this may change (due to equalization)
        self.ref_pch_out_dbm = self.params.target_pch_out_db
        self.loss = 0  # auto-design interest
        self.effective_loss = None
        self.passive = True
        self.restrictions = self.params.restrictions
        self.per_degree_pch_out_db = self.params.per_degree_pch_out_db
        # element contains the two types of equalisation parameters, but only one is not None or empty
        self.target_psd_out_mWperGHz = self.params.target_psd_out_mWperGHz
        self.per_degree_pch_psd = self.params.per_degree_pch_psd

    @property
    def to_json(self):
        if self.ref_pch_out_dbm:
            equalisation, value = 'target_pch_out_db', self.ref_pch_out_dbm
            perdegree, perdegreevalue = 'per_degree_pch_out_db', self.per_degree_pch_out_db
        if self.target_psd_out_mWperGHz:
            equalisation, value = 'target_psd_out_mWperGHz', self.target_psd_out_mWperGHz
            perdegree, perdegreevalue = 'per_degree_psd_out_mWperGHz', self.per_degree_pch_psd
        return {'uid': self.uid,
                'type': type(self).__name__,
                'params': {
                    equalisation: value,
                    'restrictions': self.restrictions,
                    perdegree: perdegreevalue
                    },
                'metadata': {
                    'location': self.metadata['location']._asdict()
                }
                }

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid!r}, loss={self.loss!r})'

    def __str__(self):
        if self.effective_loss is None:
            return f'{type(self).__name__} {self.uid}'

        total_pch = pretty_summary_print(per_label_summary(self.pch_out_dbm, self.label))
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  effective loss (dB):  {self.effective_loss:.2f}',
                          f'  pch out (dBm):        {self.ref_pch_out_dbm:.2f}',
                          f'  total pch (dBm):      {total_pch}'])

    def propagate(self, spectral_info, degree):
        # pin_target and loss are read from eqpt_config.json['Roadm']
        # all ingress channels in xpress are set to this power level
        # but add channels are not, so we define an effective loss
        # in the case of add channels
        # find the target power on this degree:
        # if a target power has been defined for this degree use it else use the global one.
        # if the input power is lower than the target one, use the input power instead because
        # a ROADM doesn't amplify, it can only attenuate

        # TODO maybe add a minimum loss for the ROADM
        # check equalization: if ref_pch_out_dbm is defined then use it
        # change per_degree_pch from scalar to an array / add a ref_power, ref_baudrate ...
        ref_baud_rate = spectral_info.pref.ref_carrier['baud_rate']
        if self.target_pch_out_dbm:
            per_degree_pch = self.per_degree_pch_out_db[degree] \
                if degree in self.per_degree_pch_out_db else self.target_pch_out_dbm
            ref_per_degree_pch = per_degree_pch
            per_degree_pch = per_degree_pch * ones(len(spectral_info.channel_number))
        elif self.target_psd_out_mWperGHz:
            per_degree_pch = psd2powerdbm(self.per_degree_pch_psd[degree], spectral_info.baud_rate) \
                if degree in self.per_degree_pch_psd \
                else psd2powerdbm(self.target_psd_out_mWperGHz, spectral_info.baud_rate)
            ref_per_degree_pch = psd2powerdbm(self.per_degree_pch_psd[degree], ref_baud_rate) \
                if degree in self.per_degree_pch_psd \
                else psd2powerdbm(self.target_psd_out_mWperGHz, ref_baud_rate)
        # definition of ref_pch_out_db for the reference channel: depending on propagation input power (p_spani) might
        # be smaller than the target power out of the roadm on this degree then use the min value between both
        self.ref_pch_out_dbm = min(spectral_info.pref.p_spani, ref_per_degree_pch)
        # definition of effective_loss: value for the reference channel
        self.effective_loss = spectral_info.pref.p_spani - self.ref_pch_out_dbm
        input_power = spectral_info.signal + spectral_info.nli + spectral_info.ase
        # computation of the per channel target power according to equalization policy
        if self.target_pch_out_dbm:
            min_power = watt2dbm(min(input_power))
            per_degree_pch = minimum(per_degree_pch, min_power)
        elif self.target_psd_out_mWperGHz:
            # Assume that channels with same baudrate are equalized to the same power.
            # Applies the same min strategy if the input power is lower than the target one:
            # use the min psd of all carriers.
            # If all carriers have identical baud_rate, the target power is the min power
            # of the carrier compared to per_degree_pch (same as power equalization)
            min_psd = min(psdmwperghz(input_power, spectral_info.baud_rate))
            temp = powerdbm2psdmwperghz(per_degree_pch, spectral_info.baud_rate)
            per_degree_pch = psd2powerdbm(minimum(temp, min_psd), spectral_info.baud_rate)
        # target power should follow same delta power as in p_span0_per_channel
        # if no specific delta, then apply equalization (later on)
        pref = spectral_info.pref
        delta_channel_power = pref.p_span0_per_channel - pref.p_span0
        delta_power = watt2dbm(input_power) - (per_degree_pch + delta_channel_power)
        spectral_info.apply_attenuation_db(delta_power)
        spectral_info.pmd = sqrt(spectral_info.pmd ** 2 + self.params.pmd ** 2)
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        self.label = spectral_info.label

    def update_pref(self, spectral_info):
        """ updates the value in Pref in spectral_info. p_span0 and p_span0_per_channel are unchanged, only p_spani
        which contains the power for the reference channel after propagation in the ROADM.
        p_span0_per_channel corresponds exactly to the current mix of channels {freq: pow},
        it serves as reference to compute delta_power wrt ref channel.
        So, we implement two main equalisation: all channels with same power or same psd plus
        a user defined delta power based on p_span0_per_channel vector
        """
        spectral_info.pref = spectral_info.pref._replace(p_spani=self.ref_pch_out_dbm)

    def __call__(self, spectral_info, degree):
        self.propagate(spectral_info, degree=degree)
        self.update_pref(spectral_info)
        return spectral_info


class Fused(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if not params:
            params = {}
        super().__init__(*args, params=FusedParams(**params), **kwargs)
        self.loss = self.params.loss
        self.passive = True

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'params': {
                    'loss': self.loss
                },
                'metadata': {
                    'location': self.metadata['location']._asdict()
                }
                }

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid!r}, loss={self.loss!r})'

    def __str__(self):
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  loss (dB): {self.loss:.2f}'])

    def propagate(self, spectral_info):
        spectral_info.apply_attenuation_db(self.loss)

    def update_pref(self, spectral_info):
        self.pch_out_db = round(watt2dbm(mean(spectral_info.signal)), 2)
        spectral_info.pref = spectral_info.pref._replace(p_span0=spectral_info.pref.p_span0,
                                                         p_spani=spectral_info.pref.p_spani - self.loss)

    def __call__(self, spectral_info):
        self.propagate(spectral_info)
        self.update_pref(spectral_info)
        return spectral_info


class Fiber(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if not params:
            params = {}
        super().__init__(*args, params=FiberParams(**params), **kwargs)
        self.pch_out_db = None
        self.passive = True

        # Raman efficiency matrix function of the delta frequency
        if self.params.raman_efficiency:
            frequency_offset = self.params.raman_efficiency['frequency_offset']
            frequency_offset = append(-flip(frequency_offset[1:]), frequency_offset)
            cr = self.params.raman_efficiency['cr']
            cr = append(- flip(cr[1:]), cr)
            self._cr_function = lambda frequency: interp(frequency, frequency_offset, cr)
        else:
            self._cr_function = lambda frequency: zeros(squeeze(frequency).shape)

        # Lumped losses
        if self.params.lumped_losses:
            z_lumped_losses = array([lumped['position'] for lumped in self.params.lumped_losses])  # km
            lumped_losses_power = array([lumped['loss'] for lumped in self.params.lumped_losses])  # dB
            if not ((z_lumped_losses > 0) * (z_lumped_losses < 1e-3 * self.params.length)).all():
                raise NetworkTopologyError(
                    f"Lumped loss positions must be between 0 and the fiber length ({1e-3 * self.params.length} km), " +
                    f"boundaries excluded.")
            self.lumped_losses = db2lin(- lumped_losses_power)  # [linear units]
            self.z_lumped_losses = array(z_lumped_losses) * 1e3  # [m]
        else:
            self.lumped_losses = None
            self.z_lumped_losses = None

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'type_variety': self.type_variety,
                'params': {
                    # have to specify each because namedtupple cannot be updated :(
                    'length': round(self.params.length * 1e-3, 6),
                    'loss_coef': self.params.loss_coef * 1e3,
                    'length_units': 'km',
                    'att_in': self.params.att_in,
                    'con_in': self.params.con_in,
                    'con_out': self.params.con_out
                },
                'metadata': {
                    'location': self.metadata['location']._asdict()
                }
                }

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid!r}, ' \
            f'length={round(self.params.length * 1e-3,1)!r}km, ' \
            f'loss={round(self.loss,1)!r}dB)'

    def __str__(self):
        if self.pch_out_db is None:
            return f'{type(self).__name__} {self.uid}'

        total_pch = pretty_summary_print(per_label_summary(self.pch_out_dbm, self.label))
        return '\n'.join([f'{type(self).__name__}          {self.uid}',
                          f'  type_variety:                {self.type_variety}',
                          f'  length (km):                 {self.params.length * 1e-3:.2f}',
                          f'  pad att_in (dB):             {self.params.att_in:.2f}',
                          f'  total loss (dB):             {self.loss:.2f}',
                          f'  (includes conn loss (dB) in: {self.params.con_in:.2f} out: {self.params.con_out:.2f})',
                          f'  (conn loss out includes EOL margin defined in eqpt_config.json)',
                          f'  pch out (dBm):               {self.pch_out_db:.2f}',
                          f'  total pch (dBm):             {total_pch}'])

    def loss_coef_func(self, frequency):
        frequency = asarray(frequency)
        if self.params.loss_coef.size > 1:
            try:
                loss_coef = interp1d(self.params.f_loss_ref, self.params.loss_coef)(frequency)
            except ValueError as e:
                print(ansi_escapes.red +
                      f'\n WARNING {e} ' + 'Value extrapolated.' + ansi_escapes.reset)
                loss_coef = interp1d(self.params.f_loss_ref, self.params.loss_coef, bounds_error=False,
                                     fill_value='extrapolate')(frequency)
        else:
            loss_coef = full(frequency.size, self.params.loss_coef)
        return squeeze(loss_coef)


    @property
    def loss(self):
        """total loss including padding att_in: useful for polymorphism with roadm loss"""
        return self.loss_coef_func(self.params.ref_frequency) * self.params.length + \
            self.params.con_in + self.params.con_out + self.params.att_in

    def lin_attenuation(self, frequency):
        return 1 / db2lin(self.params.length * self.loss_coef_func(frequency))

    def alpha(self, frequency):
        """Returns the linear exponent attenuation coefficient such that 
        :math: `lin_attenuation = e^{- alpha length}` 

        :param frequency: the frequency at which alpha is computed [Hz]
        :return: alpha: power attenuation coefficient for f in frequency [Neper/m]
        """
        return self.loss_coef_func(frequency) / (10 * log10(exp(1)))

    def cr(self, frequency):
        """Returns the raman efficiency matrix including the vibrational loss

        :param frequency: the frequency at which cr is computed [Hz]
        :return: cr: raman efficiency matrix [1 / (W m)]
        """
        df = outer(ones(frequency.shape), frequency) - outer(frequency, ones(frequency.shape))
        cr = self._cr_function(df)
        vibrational_loss = outer(frequency, ones(frequency.shape)) / outer(ones(frequency.shape), frequency)
        return cr * (cr >= 0) + cr * (cr < 0) * vibrational_loss  # Raman efficiency [1/(W m)]

    def chromatic_dispersion(self, freq=193.5e12):
        """Returns accumulated chromatic dispersion (CD).

        :param freq: the frequency at which the chromatic dispersion is computed
        :return: chromatic dispersion: the accumulated dispersion [s/m]
        """
        beta2 = self.params.beta2
        beta3 = self.params.beta3
        ref_f = self.params.ref_frequency
        length = self.params.length
        beta = beta2 + 2 * pi * beta3 * (freq - ref_f)
        dispersion = -beta * 2 * pi * ref_f**2 / c
        return dispersion * length

    @property
    def pmd(self):
        """differential group delay (PMD) [s]"""
        return self.params.pmd_coef * sqrt(self.params.length)

    def propagate(self, spectral_info: SpectralInformation):
        """Modifies the spectral information computing the attenuation, the non-linear interference generation,
        the CD and PMD accumulation.
        """
        # apply the attenuation due to the input connector loss
        attenuation_in_db = self.params.con_in + self.params.att_in
        spectral_info.apply_attenuation_db(attenuation_in_db)

        # inter channels Raman effect
        stimulated_raman_scattering = RamanSolver.calculate_stimulated_raman_scattering(spectral_info, self)

        # NLI noise evaluated at the fiber input
        spectral_info.nli += NliSolver.compute_nli(spectral_info, stimulated_raman_scattering, self)

        # chromatic dispersion and pmd variations
        spectral_info.chromatic_dispersion += self.chromatic_dispersion(spectral_info.frequency)
        spectral_info.pmd = sqrt(spectral_info.pmd ** 2 + self.pmd ** 2)

        # apply the attenuation due to the fiber losses
        attenuation_fiber = stimulated_raman_scattering.loss_profile[:, -1]
        spectral_info.apply_attenuation_lin(attenuation_fiber)

        # apply the attenuation due to the output connector loss
        attenuation_out_db = self.params.con_out
        spectral_info.apply_attenuation_db(attenuation_out_db)

    def update_pref(self, spectral_info):
        # in case of Raman, the resulting loss of the fiber is not equivalent to self.loss
        # because of Raman gain. In order to correctly update pref, we need the resulting loss:
        # power_out - power_in. We use the total signal power (sum on all channels) to compute
        # this loss, because pref is a noiseless reference.
        loss = round(lin2db(sum(self._psig_in) / sum(spectral_info.signal)), 2)
        self.pch_out_db = spectral_info.pref.p_spani - loss
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        self.label = spectral_info.label
        spectral_info.pref = spectral_info.pref._replace(p_span0=spectral_info.pref.p_span0,
                                                         p_spani=spectral_info.pref.p_spani - loss)

    def __call__(self, spectral_info):
        # _psig_in records the spectral information before propagartion. deepcopy is needed
        # otherwise the object gets chaged with propagation. 
        self._psig_in = deepcopy(spectral_info.signal)
        self.propagate(spectral_info)
        self.update_pref(spectral_info)
        return spectral_info


class RamanFiber(Fiber):
    def __init__(self, *args, params=None, **kwargs):
        super().__init__(*args, params=params, **kwargs)
        if self.operational and 'raman_pumps' in self.operational:
            self.raman_pumps = tuple(PumpParams(p['power'], p['frequency'], p['propagation_direction'])
                                     for p in self.operational['raman_pumps'])
        else:
            raise NetworkTopologyError(f'Fiber element uid:{self.uid} '
                                       f'defined as RamanFiber without raman pumps description')
        self.temperature = self.operational['temperature'] if 'temperature' in self.operational else None

    def propagate(self, spectral_info: SpectralInformation):
        """Modifies the spectral information computing the attenuation, the non-linear interference generation,
        the CD and PMD accumulation.
        """
        # apply the attenuation due to the input connector loss
        attenuation_in_db = self.params.con_in + self.params.att_in
        spectral_info.apply_attenuation_db(attenuation_in_db)

        # Raman pumps and inter channel Raman effect
        stimulated_raman_scattering = RamanSolver.calculate_stimulated_raman_scattering(spectral_info, self)
        spontaneous_raman_scattering = \
            RamanSolver.calculate_spontaneous_raman_scattering(spectral_info, stimulated_raman_scattering, self)

        # nli and ase noise evaluated at the fiber input
        spectral_info.nli += NliSolver.compute_nli(spectral_info, stimulated_raman_scattering, self)
        spectral_info.ase += spontaneous_raman_scattering

        # chromatic dispersion and pmd variations
        spectral_info.chromatic_dispersion += self.chromatic_dispersion(spectral_info.frequency)
        spectral_info.pmd = sqrt(spectral_info.pmd ** 2 + self.pmd ** 2)

        # apply the attenuation due to the fiber losses
        attenuation_fiber = stimulated_raman_scattering.loss_profile[:spectral_info.number_of_channels, -1]

        spectral_info.apply_attenuation_lin(attenuation_fiber)

        # apply the attenuation due to the output connector loss
        attenuation_out_db = self.params.con_out
        spectral_info.apply_attenuation_db(attenuation_out_db)


class Edfa(_Node):
    def __init__(self, *args, params=None, operational=None, **kwargs):
        if params is None:
            params = {}
        if operational is None:
            operational = {}
        self.variety_list = kwargs.pop('variety_list', None)
        super().__init__(*args, params=EdfaParams(**params), operational=EdfaOperational(**operational), **kwargs)
        self.interpol_dgt = None  # interpolated dynamic gain tilt
        self.interpol_gain_ripple = None  # gain ripple
        self.interpol_nf_ripple = None  # nf_ripple
        self.channel_freq = None  # SI channel frequencies
        # nf, gprofile, pin and pout attributes are set by interpol_params
        self.nf = None  # dB edfa nf at operational.gain_target
        self.gprofile = None
        self.pin_db = None
        self.nch = None
        self.pout_db = None
        self.target_pch_out_db = None
        self.effective_pch_out_db = None
        self.passive = False
        self.att_in = None
        self.effective_gain = self.operational.gain_target
        self.delta_p = self.operational.delta_p  # delta P with Pref (power swwep) in power mode
        self.tilt_target = self.operational.tilt_target
        self.out_voa = self.operational.out_voa

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'type_variety': self.params.type_variety,
                'operational': {
                    'gain_target': self.effective_gain,
                    'delta_p': self.delta_p,
                    'tilt_target': self.tilt_target,
                    'out_voa': self.out_voa
                },
                'metadata': {
                    'location': self.metadata['location']._asdict()
                }
                }

    def __repr__(self):
        return (f'{type(self).__name__}(uid={self.uid!r}, '
                f'type_variety={self.params.type_variety!r}, '
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
        total_pch = pretty_summary_print(per_label_summary(self.pch_out_dbm, self.label))
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  type_variety:           {self.params.type_variety}',
                          f'  effective gain(dB):     {self.effective_gain:.2f}',
                          f'  (before att_in and before output VOA)',
                          f'  noise figure (dB):      {nf:.2f}',
                          f'  (including att_in)',
                          f'  pad att_in (dB):        {self.att_in:.2f}',
                          f'  Power In (dBm):         {self.pin_db:.2f}',
                          f'  Power Out (dBm):        {self.pout_db:.2f}',
                          f'  Delta_P (dB):           ' + (f'{self.delta_p:.2f}' if self.delta_p is not None else 'None'),
                          f'  target pch (dBm):       ' + (f'{self.target_pch_out_db:.2f}' if self.target_pch_out_db is not None else 'None'),
                          f'  effective pch (dBm):    {self.effective_pch_out_db:.2f}',
                          f'  total pch (dBm):        {total_pch}',
                          f'  output VOA (dB):        {self.out_voa:.2f}'])

    def interpol_params(self, spectral_info):
        """interpolate SI channel frequencies with the edfa dgt and gain_ripple frquencies from JSON
        :param spectral_info: instance of gnpy.core.info.SpectralInformation
        :return: None
        """
        # TODO|jla: read amplifier actual frequencies from additional params in json

        self.channel_freq = spectral_info.frequency
        amplifier_freq = arrange_frequencies(len(self.params.dgt), self.params.f_min, self.params.f_max)  # Hz
        self.interpol_dgt = interp(spectral_info.frequency, amplifier_freq, self.params.dgt)

        amplifier_freq = arrange_frequencies(len(self.params.gain_ripple), self.params.f_min, self.params.f_max)  # Hz
        self.interpol_gain_ripple = interp(spectral_info.frequency, amplifier_freq, self.params.gain_ripple)

        amplifier_freq = arrange_frequencies(len(self.params.nf_ripple), self.params.f_min, self.params.f_max)  # Hz
        self.interpol_nf_ripple = interp(spectral_info.frequency, amplifier_freq, self.params.nf_ripple)

        self.nch = spectral_info.number_of_channels
        pin  =  spectral_info.signal + spectral_info.ase + spectral_info.nli
        self.pin_db = lin2db(sum(pin * 1e3))

        """in power mode: delta_p is defined and can be used to calculate the power target
        This power target is used calculate the amplifier gain"""
        pref = spectral_info.pref
        if self.delta_p is not None:
            self.target_pch_out_db = round(self.delta_p + pref.p_span0, 2)
            self.effective_gain = self.target_pch_out_db - pref.p_spani

        """check power saturation and correct effective gain & power accordingly:"""
        # compute the sum of powers of carriers at the input of the amplifier accounting for the expected power mixt
        delta_channel_power = pref.p_span0_per_channel - pref.p_span0
        input_total_power = lin2db(sum([db2lin(pref.p_spani + d) for d in delta_channel_power]))
        self.effective_gain = min(
            self.effective_gain,
            # self.params.p_max - (pref.p_spani + pref.neq_ch)
            self.params.p_max - input_total_power
        )
        #print(self.uid, self.effective_gain, self.operational.gain_target)
        self.effective_pch_out_db = round(pref.p_spani + self.effective_gain, 2)

        """check power saturation and correct target_gain accordingly:"""
        #print(self.uid, self.effective_gain, self.pin_db, pref.p_spani)
        self.nf = self._calc_nf()
        self.gprofile = self._gain_profile(pin)

        pout = (pin + self.noise_profile(spectral_info)) * db2lin(self.gprofile)
        self.pout_db = lin2db(sum(pout * 1e3))
        # ase & nli are only calculated in signal bandwidth
        #    pout_db is not the absolute full output power (negligible if sufficient channels)

    def _nf(self, type_def, nf_model, nf_fit_coeff, gain_min, gain_flatmax, gain_target):
        # if hybrid raman, use edfa_gain_flatmax attribute, else use gain_flatmax
        #gain_flatmax = getattr(params, 'edfa_gain_flatmax', params.gain_flatmax)
        pad = max(gain_min - gain_target, 0)
        gain_target += pad
        dg = max(gain_flatmax - gain_target, 0)
        if type_def == 'variable_gain':
            g1a = gain_target - nf_model.delta_p - dg
            nf_avg = lin2db(db2lin(nf_model.nf1) + db2lin(nf_model.nf2) / db2lin(g1a))
        elif type_def == 'fixed_gain':
            nf_avg = nf_model.nf0
        elif type_def == 'openroadm':
            pin_ch = self.pin_db - lin2db(self.nch)
            # model OSNR = f(Pin)
            nf_avg = pin_ch - polyval(nf_model.nf_coef, pin_ch) + 58
        elif type_def == 'openroadm_preamp':
            pin_ch = self.pin_db - lin2db(self.nch)
            # model OSNR = f(Pin)
            nf_avg = pin_ch - min((4 * pin_ch + 275) / 7, 33) + 58
        elif type_def == 'openroadm_booster':
            # model a zero-noise amp with "infinitely negative" (in dB) NF
            nf_avg = float('-inf')
        elif type_def == 'advanced_model':
            nf_avg = polyval(nf_fit_coeff, -dg)
        return nf_avg + pad, pad

    def _calc_nf(self, avg=False):
        """nf calculation based on 2 models: self.params.nf_model.enabled from json import:
        True => 2 stages amp modelling based on precalculated nf1, nf2 and delta_p in build_OA_json
        False => polynomial fit based on self.params.nf_fit_coeff"""
        # gain_min > gain_target TBD:
        if self.params.type_def == 'dual_stage':
            g1 = self.params.preamp_gain_flatmax
            g2 = self.effective_gain - g1
            nf1_avg, pad = self._nf(self.params.preamp_type_def,
                                    self.params.preamp_nf_model,
                                    self.params.preamp_nf_fit_coeff,
                                    self.params.preamp_gain_min,
                                    self.params.preamp_gain_flatmax,
                                    g1)
            # no padding expected for the 1stage because g1 = gain_max
            nf2_avg, pad = self._nf(self.params.booster_type_def,
                                    self.params.booster_nf_model,
                                    self.params.booster_nf_fit_coeff,
                                    self.params.booster_gain_min,
                                    self.params.booster_gain_flatmax,
                                    g2)
            nf_avg = lin2db(db2lin(nf1_avg) + db2lin(nf2_avg - g1))
            # no padding expected for the 1stage because g1 = gain_max
            pad = 0
        else:
            nf_avg, pad = self._nf(self.params.type_def,
                                   self.params.nf_model,
                                   self.params.nf_fit_coeff,
                                   self.params.gain_min,
                                   self.params.gain_flatmax,
                                   self.effective_gain)

        self.att_in = pad  # not used to attenuate carriers, only used in _repr_ and _str_
        if avg:
            return nf_avg
        else:
            return self.interpol_nf_ripple + nf_avg  # input VOA = 1 for 1 NF degradation

    def noise_profile(self, spectral_info):
        """Computes amplifier ASE noise integrated over the signal bandwidth. This is calculated at amplifier input.

        :param spectral_info: instance of gnpy.core.info.SpectralInformation
        :return: the asepower in W in the signal bandwidth bw for 96 channels
        :return type: numpy array of float

        ASE power using per channel gain profile inputs:

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

        NOTE:

            The output is the total ASE in the channel or spectral slice. For
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

        ase = h * spectral_info.baud_rate * spectral_info.frequency * db2lin(self.nf)  # W
        return ase  # in W at amplifier input

    def _gain_profile(self, pin, err_tolerance=1.0e-11, simple_opt=True):
        """
        Pin : input power / channel in W

        :param gain_ripple: design flat gain
        :param dgt: design gain tilt
        :param Pin: total input power in W
        :param gp: Average gain setpoint in dB units (provisioned gain)
        :param gtp: gain tilt setting (provisioned tilt)
        :type gain_ripple: numpy.ndarray
        :type dgt: numpy.ndarray
        :type Pin: numpy.ndarray
        :type gp: float
        :type gtp: float
        :return: gain profile in dBm, per channel or spectral slice
        :rtype: numpy.ndarray

        Checking of output power clamping is implemented in interpol_params().


        Based on:

            R. di Muro, "The Er3+ fiber gain coefficient derived from a dynamic
            gain tilt technique", Journal of Lightwave Technology, Vol. 18,
            Iss. 3, Pp. 343-347, 2000.

            Ported from Matlab version written by David Boerges at Ciena.
        """

        # TODO|jla: check what param should be used (currently length(dgt))
        if len(self.interpol_dgt) == 1:
            return array([self.effective_gain])

        # TODO|jla: find a way to use these or lose them. Primarily we should have
        # a way to determine if exceeding the gain or output power of the amp
        tot_in_power_db = self.pin_db  # Pin in W

        # linear fit to get the
        p = polyfit(self.channel_freq, self.interpol_dgt, 1)
        dgt_slope = p[0]

        # Calculate the target slope
        targ_slope = -self.tilt_target / (self.params.f_max - self.params.f_min)

        # first estimate of DGT scaling
        dgts1 = targ_slope / dgt_slope if dgt_slope != 0. else 0.

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

        pout_db = lin2db(sum(pin * 1e3 * db2lin(g2nd)))
        dgts2 = self.effective_gain - (pout_db - tot_in_power_db)

        # center estimate of amp ch gain
        xcent = dgts2
        gcent = g1st - voa + array(self.interpol_dgt) * xcent
        pout_db = lin2db(sum(pin * 1e3 * db2lin(gcent)))
        gavg_cent = pout_db - tot_in_power_db

        # Lower estimate of amp ch gain
        deltax = max(g1st) - min(g1st)
        # if no ripple deltax = 0 and xlow = xcent: div 0
        # TODO|jla: add check for flat gain response
        if abs(deltax) <= 0.05:  # not enough ripple to consider calculation
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

    def propagate(self, spectral_info):
        """add ASE noise to the propagating carriers of :class:`.info.SpectralInformation`"""
        # interpolate the amplifier vectors with the carriers freq, calculate nf & gain profile
        self.interpol_params(spectral_info)

        ase = self.noise_profile(spectral_info)
        spectral_info.ase += ase

        spectral_info.apply_gain_db(self.gprofile - self.out_voa)
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        self.label = spectral_info.label

    def update_pref(self, spectral_info):
        self.pch_out_db = watt2dbm(spectral_info.signal)
        spectral_info.pref = spectral_info.pref._replace(p_span0=spectral_info.pref.p_span0,
                                                         p_spani=spectral_info.pref.p_spani
                                                         + self.effective_gain - self.out_voa)

    def __call__(self, spectral_info):
        self.propagate(spectral_info)
        self.update_pref(spectral_info)
        return spectral_info
