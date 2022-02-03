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
    squeeze, zeros, append, flip, outer
from scipy.constants import h, c
from scipy.interpolate import interp1d
from collections import namedtuple

from gnpy.core.utils import lin2db, db2lin, arrange_frequencies, snr_sum
from gnpy.core.parameters import RoadmParams, FusedParams, FiberParams, PumpParams, EdfaParams, EdfaOperational
from gnpy.core.science_utils import NliSolver, RamanSolver
from gnpy.core.info import SpectralInformation
from gnpy.core.exceptions import NetworkTopologyError, SpectrumError


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
        self.pdl = None
        self.penalties = {}
        self.total_penalty = 0

    def _calc_cd(self, spectral_info):
        """ Updates the Transceiver property with the CD of the received channels. CD in ps/nm.
        """
        self.chromatic_dispersion = spectral_info.chromatic_dispersion * 1e3

    def _calc_pmd(self, spectral_info):
        """Updates the Transceiver property with the PMD of the received channels. PMD in ps.
        """
        self.pmd = spectral_info.pmd * 1e12

    def _calc_pdl(self, spectral_info):
        """Updates the Transceiver property with the PDL of the received channels. PDL in dB.
        """
        self.pdl = spectral_info.pdl

    def _calc_penalty(self, impairment_value, boundary_list):
        return interp(impairment_value, boundary_list['up_to_boundary'], boundary_list['penalty_value'],
                      left=float('inf'), right=float('inf'))

    def calc_penalties(self, penalties):
        """Updates the Transceiver property with penalties (CD, PMD, etc.) of the received channels in dB.
           Penalties are linearly interpolated between given points and set to 'inf' outside interval.
        """
        self.penalties = {impairment: self._calc_penalty(getattr(self, impairment), boundary_list)
                          for impairment, boundary_list in penalties.items()}
        self.total_penalty = sum(list(self.penalties.values()), axis=0)

    def _calc_snr(self, spectral_info):
        with errstate(divide='ignore'):
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
                f'pmd={self.pmd!r}, '
                f'pdl={self.pdl!r}, '
                f'penalties={self.penalties!r})')

    def __str__(self):
        if self.snr is None or self.osnr_ase is None:
            return f'{type(self).__name__} {self.uid}'

        snr = round(mean(self.snr), 2)
        osnr_ase = round(mean(self.osnr_ase), 2)
        osnr_ase_01nm = round(mean(self.osnr_ase_01nm), 2)
        snr_01nm = round(mean(self.snr_01nm), 2)
        cd = mean(self.chromatic_dispersion)
        pmd = mean(self.pmd)
        pdl = mean(self.pdl)

        result = '\n'.join([f'{type(self).__name__} {self.uid}',

                          f'  GSNR (0.1nm, dB):          {snr_01nm:.2f}',
                          f'  GSNR (signal bw, dB):      {snr:.2f}',
                          f'  OSNR ASE (0.1nm, dB):      {osnr_ase_01nm:.2f}',
                          f'  OSNR ASE (signal bw, dB):  {osnr_ase:.2f}',
                          f'  CD (ps/nm):                {cd:.2f}',
                          f'  PMD (ps):                  {pmd:.2f}',
                          f'  PDL (dB):                  {pdl:.2f}'])

        cd_penalty = self.penalties.get('chromatic_dispersion')
        if cd_penalty is not None:
            result += f'\n  CD penalty (dB):           {mean(cd_penalty):.2f}'
        pmd_penalty = self.penalties.get('pmd')
        if pmd_penalty is not None:
            result += f'\n  PMD penalty (dB):          {mean(pmd_penalty):.2f}'
        pdl_penalty = self.penalties.get('pdl')
        if pdl_penalty is not None:
            result += f'\n  PDL penalty (dB):          {mean(pdl_penalty):.2f}'

        return result

    def __call__(self, spectral_info):
        self._calc_snr(spectral_info)
        self._calc_cd(spectral_info)
        self._calc_pmd(spectral_info)
        self._calc_pdl(spectral_info)
        return spectral_info


class Roadm(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if not params:
            params = {}
        super().__init__(*args, params=RoadmParams(**params), **kwargs)
        self.pch_out_db = self.params.target_pch_out_db
        self.loss = 0  # auto-design interest
        self.effective_loss = None
        self.passive = True
        self.restrictions = self.params.restrictions
        self.per_degree_pch_out_db = self.params.per_degree_pch_out_db

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'params': {
                    'target_pch_out_db': self.pch_out_db,
                    'restrictions': self.restrictions,
                    'per_degree_pch_out_db': self.per_degree_pch_out_db
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

        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  effective loss (dB):  {self.effective_loss:.2f}',
                          f'  pch out (dBm):        {self.pch_out_db:.2f}'])

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
        per_degree_pch = self.per_degree_pch_out_db[degree] \
            if degree in self.per_degree_pch_out_db else self.pch_out_db
        self.pch_out_db = min(spectral_info.pref.p_spani, per_degree_pch)
        self.effective_loss = spectral_info.pref.p_spani - self.pch_out_db
        input_power = spectral_info.signal + spectral_info.nli + spectral_info.ase
        min_power = min(lin2db(input_power * 1e3))
        per_degree_pch = per_degree_pch if per_degree_pch < min_power else min_power
        delta_power = lin2db(input_power * 1e3) - per_degree_pch
        spectral_info.apply_attenuation_db(delta_power)
        spectral_info.pmd = sqrt(spectral_info.pmd ** 2 + self.params.pmd ** 2)
        spectral_info.pdl = sqrt(spectral_info.pdl ** 2 + self.params.pdl ** 2)

    def update_pref(self, spectral_info):
        spectral_info.pref = spectral_info.pref._replace(p_span0=spectral_info.pref.p_span0, p_spani=self.pch_out_db)

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

        # Raman efficiency matrix function of the delta frequency constructed such that each row is related to a
        # fixed frequency: positive elements represent a gain (from higher frequency) and negative elements represent
        # a loss (to lower frequency)
        if self.params.raman_efficiency:
            frequency_offset = self.params.raman_efficiency['frequency_offset']
            frequency_offset = append(-flip(frequency_offset[1:]), frequency_offset)
            cr = self.params.raman_efficiency['cr']
            cr = append(- flip(cr[1:]), cr)
            self._cr_function = lambda frequency: interp(frequency, frequency_offset, cr)
        else:
            self._cr_function = lambda frequency: zeros(squeeze(frequency).shape)

        # Lumped losses
        z_lumped_losses = array([lumped['position'] for lumped in self.params.lumped_losses])  # km
        lumped_losses_power = array([lumped['loss'] for lumped in self.params.lumped_losses])  # dB
        if not ((z_lumped_losses > 0) * (z_lumped_losses < 1e-3 * self.params.length)).all():
            raise NetworkTopologyError("Lumped loss positions must be between 0 and the fiber length "
                                       f"({1e-3 * self.params.length} km), boundaries excluded.")
        self.lumped_losses = db2lin(- lumped_losses_power)  # [linear units]
        self.z_lumped_losses = array(z_lumped_losses) * 1e3  # [m]

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'type_variety': self.type_variety,
                'params': {
                    # have to specify each because namedtupple cannot be updated :(
                    'length': round(self.params.length * 1e-3, 6),
                    'loss_coef': round(self.params.loss_coef * 1e3, 6),
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

        return '\n'.join([f'{type(self).__name__}          {self.uid}',
                          f'  type_variety:                {self.type_variety}',
                          f'  length (km):                 {self.params.length * 1e-3:.2f}',
                          f'  pad att_in (dB):             {self.params.att_in:.2f}',
                          f'  total loss (dB):             {self.loss:.2f}',
                          f'  (includes conn loss (dB) in: {self.params.con_in:.2f} out: {self.params.con_out:.2f})',
                          f'  (conn loss out includes EOL margin defined in eqpt_config.json)',
                          f'  pch out (dBm): {self.pch_out_db:.2f}'])

    def loss_coef_func(self, frequency):
        frequency = asarray(frequency)
        if self.params.loss_coef.size > 1:
            try:
                loss_coef = interp1d(self.params.f_loss_ref, self.params.loss_coef)(frequency)
            except ValueError:
                raise SpectrumError('The spectrum bandwidth exceeds the frequency interval used to define the fiber '
                                    f'loss coefficient in "{type(self).__name__} {self.uid}".'
                                    f'\nSpectrum f_min-f_max: {round(frequency[0]*1e-12,2)}-'
                                    f'{round(frequency[-1]*1e-12,2)}'
                                    f'\nLoss coefficient f_min-f_max: {round(self.params.f_loss_ref[0]*1e-12,2)}-'
                                    f'{round(self.params.f_loss_ref[-1]*1e-12,2)}')
        else:
            loss_coef = full(frequency.size, self.params.loss_coef)
        return squeeze(loss_coef)


    @property
    def loss(self):
        """total loss including padding att_in: useful for polymorphism with roadm loss"""
        return self.loss_coef_func(self.params.ref_frequency) * self.params.length + \
            self.params.con_in + self.params.con_out + self.params.att_in

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

    def chromatic_dispersion(self, freq=None):
        """Returns accumulated chromatic dispersion (CD).

        :param freq: the frequency at which the chromatic dispersion is computed
        :return: chromatic dispersion: the accumulated dispersion [s/m]
        """
        freq = self.params.ref_frequency if freq is None else freq
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
        loss = round(lin2db(self._psig_in / sum(spectral_info.signal)), 2)
        self.pch_out_db = spectral_info.pref.p_spani - loss
        spectral_info.pref = spectral_info.pref._replace(p_span0=spectral_info.pref.p_span0,
                                                         p_spani=self.pch_out_db)

    def __call__(self, spectral_info):
        # _psig_in records the total signal power of the spectral information before propagation.
        self._psig_in = sum(spectral_info.signal)
        self.propagate(spectral_info)
        self.update_pref(spectral_info)
        return spectral_info


class RamanFiber(Fiber):
    def __init__(self, *args, params=None, **kwargs):
        super().__init__(*args, params=params, **kwargs)
        if not self.operational:
            raise NetworkTopologyError(f'Fiber element uid:{self.uid} '
                                       'defined as RamanFiber without operational parameters')

        if 'raman_pumps' not in self.operational:
            raise NetworkTopologyError(f'Fiber element uid:{self.uid} '
                                       'defined as RamanFiber without raman pumps description in operational')

        if 'temperature' not in self.operational:
            raise NetworkTopologyError(f'Fiber element uid:{self.uid} '
                                       'defined as RamanFiber without temperature in operational')

        pump_loss = db2lin(self.params.con_out)
        self.raman_pumps = tuple(PumpParams(p['power'] / pump_loss, p['frequency'], p['propagation_direction'])
                                 for p in self.operational['raman_pumps'])
        self.temperature = self.operational['temperature']

    @property
    def to_json(self):
        return dict(super().to_json, operational=self.operational)

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
                    'gain_target': round(self.effective_gain, 6),
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
        pin = spectral_info.signal + spectral_info.ase + spectral_info.nli
        self.pin_db = lin2db(sum(pin * 1e3))
        # The following should be changed when we have the new spectral information including slot widths.
        # For now, with homogeneous spectrum, we can calculate it as the difference between neighbouring channels.
        self.slot_width = self.channel_freq[1] - self.channel_freq[0]

        """in power mode: delta_p is defined and can be used to calculate the power target
        This power target is used calculate the amplifier gain"""
        pref = spectral_info.pref
        if self.delta_p is not None:
            self.target_pch_out_db = round(self.delta_p + pref.p_span0, 2)
            self.effective_gain = self.target_pch_out_db - pref.p_spani

        """check power saturation and correct effective gain & power accordingly:"""
        self.effective_gain = min(
            self.effective_gain,
            self.params.p_max - (pref.p_spani + pref.neq_ch)
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
            # OpenROADM specifies OSNR vs. input power per channel for 50 GHz slot width so we
            # scale it to 50 GHz based on actual slot width.
            pin_ch_50GHz = self.pin_db - lin2db(self.nch) + lin2db(50e9 / self.slot_width)
            # model OSNR = f(Pin per 50 GHz channel)
            nf_avg = pin_ch_50GHz - polyval(nf_model.nf_coef, pin_ch_50GHz) + 58
        elif type_def == 'openroadm_preamp':
            # OpenROADM specifies OSNR vs. input power per channel for 50 GHz slot width so we
            # scale it to 50 GHz based on actual slot width.
            pin_ch_50GHz = self.pin_db - lin2db(self.nch) + lin2db(50e9 / self.slot_width)
            # model OSNR = f(Pin per 50 GHz channel)
            nf_avg = pin_ch_50GHz - min((4 * pin_ch_50GHz + 275) / 7, 33) + 58
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

    def noise_profile(self, spectral_info: SpectralInformation):
        """Computes amplifier ASE noise integrated over the signal bandwidth. This is calculated at amplifier input.

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
        spectral_info.pmd = sqrt(spectral_info.pmd ** 2 + self.params.pmd ** 2)
        spectral_info.pdl = sqrt(spectral_info.pdl ** 2 + self.params.pdl ** 2)

    def update_pref(self, spectral_info):
        spectral_info.pref = \
            spectral_info.pref._replace(p_span0=spectral_info.pref.p_span0,
                                        p_spani=spectral_info.pref.p_spani + self.effective_gain - self.out_voa)

    def __call__(self, spectral_info):
        self.propagate(spectral_info)
        self.update_pref(spectral_info)
        return spectral_info
