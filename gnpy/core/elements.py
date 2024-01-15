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
    squeeze, zeros, append, flip, outer, ndarray
from scipy.constants import h, c
from scipy.interpolate import interp1d
from collections import namedtuple
from typing import Union
from logging import getLogger

from gnpy.core.utils import lin2db, db2lin, arrange_frequencies, snr_sum, per_label_average, pretty_summary_print, \
    watt2dbm, psd2powerdbm
from gnpy.core.parameters import RoadmParams, FusedParams, FiberParams, PumpParams, EdfaParams, EdfaOperational
from gnpy.core.science_utils import NliSolver, RamanSolver
from gnpy.core.info import SpectralInformation, ReferenceCarrier
from gnpy.core.exceptions import NetworkTopologyError, SpectrumError, ParametersError


_logger = getLogger(__name__)


class Location(namedtuple('Location', 'latitude longitude city region')):
    def __new__(cls, latitude=0, longitude=0, city=None, region=None):
        return super().__new__(cls, latitude, longitude, city, region)


class _Node:
    """Convenience class for providing common functionality of all network elements

    This class is just an internal implementation detail; do **not** assume that all network elements
    inherit from :class:`_Node`.
    """
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
        self.latency = None
        self.penalties = {}
        self.total_penalty = 0
        self.propagated_labels = [""]

    def _calc_cd(self, spectral_info):
        """Updates the Transceiver property with the CD of the received channels. CD in ps/nm.
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

    def _calc_latency(self, spectral_info):
        """Updates the Transceiver property with the latency of the received channels. Latency in ms.
        """
        self.latency = spectral_info.latency * 1e3

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
            self.propagated_labels = spectral_info.label
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
                f'latency={self.latency!r}, '
                f'penalties={self.penalties!r})')

    def __str__(self):
        if self.snr is None or self.osnr_ase is None:
            return f'{type(self).__name__} {self.uid}'

        snr = per_label_average(self.snr, self.propagated_labels)
        osnr_ase = per_label_average(self.osnr_ase, self.propagated_labels)
        osnr_ase_01nm = per_label_average(self.osnr_ase_01nm, self.propagated_labels)
        snr_01nm = per_label_average(self.snr_01nm, self.propagated_labels)
        cd = mean(self.chromatic_dispersion)
        pmd = mean(self.pmd)
        pdl = mean(self.pdl)
        latency = mean(self.latency)

        result = '\n'.join([f'{type(self).__name__} {self.uid}',
                            f'  GSNR (0.1nm, dB):          {pretty_summary_print(snr_01nm)}',
                            f'  GSNR (signal bw, dB):      {pretty_summary_print(snr)}',
                            f'  OSNR ASE (0.1nm, dB):      {pretty_summary_print(osnr_ase_01nm)}',
                            f'  OSNR ASE (signal bw, dB):  {pretty_summary_print(osnr_ase)}',
                            f'  CD (ps/nm):                {cd:.2f}',
                            f'  PMD (ps):                  {pmd:.2f}',
                            f'  PDL (dB):                  {pdl:.2f}',
                            f'  Latency (ms):              {latency:.2f}'])

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
        self._calc_latency(spectral_info)
        return spectral_info


class Roadm(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if not params:
            params = {}
        try:
            super().__init__(*args, params=RoadmParams(**params), **kwargs)
        except ParametersError as e:
            msg = f'Config error in {kwargs["uid"]}: {e}'
            raise ParametersError(msg) from e

        # Target output power for the reference carrier, can only be computed on the fly, because it depends
        # on the path, since it depends on the equalization definition on the degree.
        self.ref_pch_out_dbm = None
        self.loss = 0  # auto-design interest

        # Optical power of carriers are equalized by the ROADM, so that the experienced loss is not the same for
        # different carriers. The ref_effective_loss records the loss for a reference carrier.
        self.ref_effective_loss = None

        self.passive = True
        self.restrictions = self.params.restrictions
        self.propagated_labels = [""]
        # element contains the two types of equalisation parameters, but only one is not None or empty
        # target for equalization for the ROADM only one must be not None
        self.target_pch_out_dbm = self.params.target_pch_out_db
        self.target_psd_out_mWperGHz = self.params.target_psd_out_mWperGHz
        self.target_out_mWperSlotWidth = self.params.target_out_mWperSlotWidth
        self.per_degree_pch_out_dbm = self.params.per_degree_pch_out_db
        self.per_degree_pch_psd = self.params.per_degree_pch_psd
        self.per_degree_pch_psw = self.params.per_degree_pch_psw
        self.ref_pch_in_dbm = {}
        self.ref_carrier = None

    @property
    def to_json(self):
        if self.target_pch_out_dbm is not None:
            equalisation, value = 'target_pch_out_db', self.target_pch_out_dbm
        elif self.target_psd_out_mWperGHz is not None:
            equalisation, value = 'target_psd_out_mWperGHz', self.target_psd_out_mWperGHz
        elif self.target_out_mWperSlotWidth is not None:
            equalisation, value = 'target_out_mWperSlotWidth', self.target_out_mWperSlotWidth
        else:
            assert False, 'There must be one default equalization defined in ROADM'
        to_json = {
            'uid': self.uid,
            'type': type(self).__name__,
            'params': {
                equalisation: value,
                'restrictions': self.restrictions,
            },
            'metadata': {
                'location': self.metadata['location']._asdict()
            }
        }
        # several per_degree equalization may coexist on different degrees
        if self.per_degree_pch_out_dbm:
            to_json['params']['per_degree_pch_out_db'] = self.per_degree_pch_out_dbm
        if self.per_degree_pch_psd:
            to_json['params']['per_degree_psd_out_mWperGHz'] = self.per_degree_pch_psd
        if self.per_degree_pch_psw:
            to_json['params']['per_degree_psd_out_mWperSlotWidth'] = self.per_degree_pch_psw
        return to_json

    def __repr__(self):
        return f'{type(self).__name__}(uid={self.uid!r}, loss={self.loss!r})'

    def __str__(self):
        if self.ref_effective_loss is None:
            return f'{type(self).__name__} {self.uid}'

        total_pch = pretty_summary_print(per_label_average(self.pch_out_dbm, self.propagated_labels))
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  effective loss (dB):     {self.ref_effective_loss:.2f}',
                          f'  reference pch out (dBm): {self.ref_pch_out_dbm:.2f}',
                          f'  actual pch out (dBm):    {total_pch}'])

    def get_roadm_target_power(self, spectral_info: SpectralInformation = None) -> Union[float, ndarray]:
        """Computes the power in dBm for a reference carrier or for a spectral information.
        power is computed based on equalization target.
        if spectral_info baud_rate is baud_rate = [32e9, 42e9, 64e9, 42e9, 32e9], and
        target_pch_out_dbm is defined to -20 dbm, then the function returns an array of powers
        [-20, -20, -20, -20, -20]
        if target_psd_out_mWperGHz is defined instead with 3.125e-4mW/GHz then it returns
        [-20, -18.819, -16.9897, -18.819, -20]
        if instead a reference_baud_rate is defined, the functions computes the result for a
        single reference carrier whose baud_rate is reference_baudrate
        """
        if spectral_info:
            if self.target_pch_out_dbm is not None:
                return full(len(spectral_info.channel_number), self.target_pch_out_dbm)
            if self.target_psd_out_mWperGHz is not None:
                return psd2powerdbm(self.target_psd_out_mWperGHz, spectral_info.baud_rate)
            if self.target_out_mWperSlotWidth is not None:
                return psd2powerdbm(self.target_out_mWperSlotWidth, spectral_info.slot_width)
        else:
            if self.target_pch_out_dbm is not None:
                return self.target_pch_out_dbm
            if self.target_psd_out_mWperGHz is not None:
                return psd2powerdbm(self.target_psd_out_mWperGHz, self.ref_carrier.baud_rate)
            if self.target_out_mWperSlotWidth is not None:
                return psd2powerdbm(self.target_out_mWperSlotWidth, self.ref_carrier.slot_width)
        return None

    def get_per_degree_ref_power(self, degree):
        """Get the target power in dBm out of ROADM degree for the reference bandwidth
        If no equalization is defined on this degree use the ROADM level one.
        """
        if degree in self.per_degree_pch_out_dbm:
            return self.per_degree_pch_out_dbm[degree]
        elif degree in self.per_degree_pch_psd:
            return psd2powerdbm(self.per_degree_pch_psd[degree], self.ref_carrier.baud_rate)
        elif degree in self.per_degree_pch_psw:
            return psd2powerdbm(self.per_degree_pch_psw[degree], self.ref_carrier.slot_width)
        return self.get_roadm_target_power()

    def get_per_degree_power(self, degree, spectral_info):
        """Get the target power in dBm out of ROADM degree for the spectral information
        If no equalization is defined on this degree use the ROADM level one.
        """
        if degree in self.per_degree_pch_out_dbm:
            return self.per_degree_pch_out_dbm[degree]
        elif degree in self.per_degree_pch_psd:
            return psd2powerdbm(self.per_degree_pch_psd[degree], spectral_info.baud_rate)
        elif degree in self.per_degree_pch_psw:
            return psd2powerdbm(self.per_degree_pch_psw[degree], spectral_info.slot_width)
        return self.get_roadm_target_power(spectral_info=spectral_info)

    def propagate(self, spectral_info, degree, from_degree):
        """Equalization targets are read from topology file if defined and completed with default
        definition of the library.
        If the input power is lower than the target one, use the input power instead because
        a ROADM doesn't amplify, it can only attenuate.
        There is no difference for add or express : the same target is applied. For the moment
        propagates operates with spectral info carriers all having the same source or destination.
        """
        # TODO maybe add a minimum loss for the ROADM

        # find the target power for the reference carrier
        ref_per_degree_pch = self.get_per_degree_ref_power(degree)
        # find the target powers for each signal carrier
        per_degree_pch = self.get_per_degree_power(degree, spectral_info=spectral_info)

        # Definition of ref_pch_out_dbm for the reference channel:
        # Depending on propagation upstream from this ROADM, the input power might be smaller than
        # the target power out configured for this ROADM degree's egress. Since ROADM does not amplify,
        # the power out of the ROADM for the ref channel is the min value between target power and input power.
        # (TODO add a minimum loss for the ROADM crossing)
        self.ref_pch_out_dbm = min(self.ref_pch_in_dbm[from_degree], ref_per_degree_pch)
        # Definition of effective_loss:
        # Optical power of carriers are equalized by the ROADM, so that the experienced loss is not the same for
        # different carriers. effective_loss records the loss for the reference carrier.
        self.ref_effective_loss = self.ref_pch_in_dbm[from_degree] - self.ref_pch_out_dbm
        input_power = spectral_info.signal + spectral_info.nli + spectral_info.ase
        target_power_per_channel = per_degree_pch + spectral_info.delta_pdb_per_channel
        # Computation of the per channel target power according to equalization policy
        # If target_power_per_channel has some channels power above input power, then the whole target is reduced.
        # For example, if user specifies delta_pdb_per_channel:
        # freq1: 1dB, freq2: 3dB, freq3: -3dB, and target is -20dBm out of the ROADM,
        # then the target power for each channel uses the specified delta_pdb_per_channel.
        # target_power_per_channel[f1, f2, f3] = -19, -17, -23
        # However if input_signal = -23, -16, -26, then the target can not be applied, because
        # -23 < -19dBm and -26 < -23dBm. Then the target is only applied to signals whose power is above the
        # threshold. others are left unchanged and unequalized.
        # the new target is [-23, -17, -26]
        # and the attenuation to apply is [-23, -16, -26] - [-23, -17, -26] = [0, 1, 0]
        # note that this changes the previous behaviour that equalized all identical channels based on the one
        # that had the min power.
        # This change corresponds to a discussion held during coders call. Please look at this document for
        # a reference: https://telecominfraproject.atlassian.net/wiki/spaces/OOPT/pages/669679645/PSE+Meeting+Minutes
        correction = (abs(watt2dbm(input_power) - target_power_per_channel)
                      - (watt2dbm(input_power) - target_power_per_channel)) / 2
        new_target = target_power_per_channel - correction
        delta_power = watt2dbm(input_power) - new_target

        spectral_info.apply_attenuation_db(delta_power)
        spectral_info.pmd = sqrt(spectral_info.pmd ** 2 + self.params.pmd ** 2)
        spectral_info.pdl = sqrt(spectral_info.pdl ** 2 + self.params.pdl ** 2)
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        self.propagated_labels = spectral_info.label

    def __call__(self, spectral_info, degree, from_degree):
        self.propagate(spectral_info, degree=degree, from_degree=from_degree)
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

    def __call__(self, spectral_info):
        self.propagate(spectral_info)
        return spectral_info


class Fiber(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if not params:
            params = {}
        try:
            super().__init__(*args, params=FiberParams(**params), **kwargs)
        except ParametersError as e:
            msg = f'Config error in {kwargs["uid"]}: {e}'
            raise ParametersError(msg) from e
        self.pch_out_db = None
        self.passive = True
        self.propagated_labels = [""]

        # Lumped losses
        z_lumped_losses = array([lumped['position'] for lumped in self.params.lumped_losses])  # km
        lumped_losses_power = array([lumped['loss'] for lumped in self.params.lumped_losses])  # dB
        if not ((z_lumped_losses > 0) * (z_lumped_losses < 1e-3 * self.params.length)).all():
            raise NetworkTopologyError("Lumped loss positions must be between 0 and the fiber length "
                                       f"({1e-3 * self.params.length} km), boundaries excluded.")
        self.lumped_losses = db2lin(- lumped_losses_power)  # [linear units]
        self.z_lumped_losses = array(z_lumped_losses) * 1e3  # [m]
        self.ref_pch_in_dbm = None

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

        total_pch = pretty_summary_print(per_label_average(self.pch_out_dbm, self.propagated_labels))
        return '\n'.join([f'{type(self).__name__}          {self.uid}',
                          f'  type_variety:                {self.type_variety}',
                          f'  length (km):                 {self.params.length * 1e-3:.2f}',
                          f'  pad att_in (dB):             {self.params.att_in:.2f}',
                          f'  total loss (dB):             {self.loss:.2f}',
                          f'  (includes conn loss (dB) in: {self.params.con_in:.2f} out: {self.params.con_out:.2f})',
                          f'  (conn loss out includes EOL margin defined in eqpt_config.json)',
                          f'  reference pch out (dBm):     {self.pch_out_db:.2f}',
                          f'  actual pch out (dBm):        {total_pch}'])

    def interpolate_parameter_over_spectrum(self, parameter, ref_frequency, spectrum_frequency, name):
        try:
            interpolation = interp1d(ref_frequency, parameter)(spectrum_frequency)
            return interpolation
        except ValueError:
            try:
                start = spectrum_frequency[0]
                stop = spectrum_frequency[-1]
            except IndexError:
                # when frequency is a 0-dimensionnal array
                start = spectrum_frequency
                stop = spectrum_frequency
            raise SpectrumError('The spectrum bandwidth exceeds the frequency interval used to define the fiber '
                                f'{name} in "{type(self).__name__} {self.uid}".'
                                f'\nSpectrum f_min-f_max: {round(start * 1e-12, 2)}-'
                                f'{round(stop * 1e-12, 2)}'
                                f'\n{name} f_min-f_max: {round(ref_frequency[0] * 1e-12, 2)}-'
                                f'{round(ref_frequency[-1] * 1e-12, 2)}')

    def loss_coef_func(self, frequency):
        frequency = asarray(frequency)
        if self.params.loss_coef.size > 1:
            loss_coef = self.interpolate_parameter_over_spectrum(self.params.loss_coef, self.params.f_loss_ref,
                                                                 frequency, 'Loss Coefficient')
        else:
            loss_coef = full(frequency.size, self.params.loss_coef)
        return squeeze(loss_coef)

    @property
    def loss(self):
        """total loss including padding att_in: useful for polymorphism with roadm loss"""
        return self.loss_coef_func(self.params.ref_frequency) * self.params.length + \
            self.params.con_in + self.params.con_out + self.params.att_in + sum(lin2db(1 / self.lumped_losses))

    def alpha(self, frequency):
        """Returns the linear exponent attenuation coefficient such that
        :math: `lin_attenuation = e^{- alpha length}`

        :param frequency: the frequency at which alpha is computed [Hz]
        :return: alpha: power attenuation coefficient for f in frequency [Neper/m]
        """
        return self.loss_coef_func(frequency) / (10 * log10(exp(1)))

    def beta2(self, frequency=None):
        """Returns the beta2 chromatic dispersion coefficient as the second order term of the beta function
        expanded as a Taylor series evaluated at the given frequency

        :param frequency: the frequency at which alpha is computed [Hz]
        :return: beta2: beta2 chromatic dispersion coefficient for f in frequency # 1/(m * Hz^2)
        """
        frequency = asarray(self.params.ref_frequency if frequency is None else frequency)
        if self.params.dispersion.size > 1:
            dispersion = self.interpolate_parameter_over_spectrum(self.params.dispersion, self.params.f_dispersion_ref,
                                                                  frequency, 'Chromatic Dispersion')
        else:
            if self.params.dispersion_slope is None:
                dispersion = (frequency / self.params.f_dispersion_ref) ** 2 * self.params.dispersion
            else:
                wavelength = c / frequency
                dispersion = self.params.dispersion + self.params.dispersion_slope * \
                             (wavelength - c / self.params.f_dispersion_ref)
        beta2 = -((c / frequency) ** 2 * dispersion) / (2 * pi * c)
        return beta2

    def beta3(self, frequency=None):
        """Returns the beta3 chromatic dispersion coefficient as the third order term of the beta function
        expanded as a Taylor series evaluated at the given frequency

        :param frequency: the frequency at which alpha is computed [Hz]
        :return: beta3: beta3 chromatic dispersion coefficient for f in frequency # 1/(m * Hz^3)
        """
        frequency = asarray(self.params.ref_frequency if frequency is None else frequency)
        if self.params.dispersion.size > 1:
            beta3 = polyfit(self.params.f_dispersion_ref - self.params.ref_frequency,
                            self.beta2(self.params.f_dispersion_ref), 2)[1] / (2*pi)
            beta3 = full(frequency.size, beta3)
        else:
            if self.params.dispersion_slope is None:
                beta3 = zeros(frequency.size)
            else:
                dispersion_slope = self.params.dispersion_slope
                beta2 = self.beta2(frequency)
                beta3 = (dispersion_slope - (4 * pi * frequency ** 3 / c ** 2) * beta2) / (
                            2 * pi * frequency ** 2 / c) ** 2
        return beta3

    def gamma(self, frequency=None):
        """Returns the nonlinear interference coefficient such that
        :math: `gamma(f) = 2 pi f n_2 c^{-1} A_{eff}^{-1}`

        :param frequency: the frequency at which gamma is computed [Hz]
        :return: gamma: nonlinear interference coefficient for f in frequency [1/(W m)]
        """
        frequency = self.params.ref_frequency if frequency is None else frequency
        return self.params.gamma_scaling(frequency)

    def cr(self, frequency):
        """Returns the raman gain coefficient matrix including the vibrational loss

        :param frequency: the frequency at which cr is computed [Hz]
        :return: cr: raman gain coefficient matrix [1 / (W m)]
        """
        df = outer(ones(frequency.shape), frequency) - outer(frequency, ones(frequency.shape))
        effective_area_overlap = self.params.effective_area_overlap(frequency, frequency)
        cr = interp(df, self.params.raman_coefficient.frequency_offset,
                    self.params.raman_coefficient.normalized_gamma_raman) * frequency / effective_area_overlap
        vibrational_loss = outer(frequency, ones(frequency.shape)) / outer(ones(frequency.shape), frequency)
        return cr * (cr >= 0) + cr * (cr < 0) * vibrational_loss  # [1/(W m)]

    def chromatic_dispersion(self, freq=None):
        """Returns accumulated chromatic dispersion (CD).

        :param freq: the frequency at which the chromatic dispersion is computed
        :return: chromatic dispersion: the accumulated dispersion [s/m]
        """
        freq = self.params.ref_frequency if freq is None else freq
        beta2 = self.beta2(freq)
        beta3 = self.beta3(freq)
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

        # latency
        spectral_info.latency += self.params.latency

        # apply the attenuation due to the fiber losses
        attenuation_fiber = stimulated_raman_scattering.loss_profile[:, -1]
        spectral_info.apply_attenuation_lin(attenuation_fiber)

        # apply the attenuation due to the output connector loss
        attenuation_out_db = self.params.con_out
        spectral_info.apply_attenuation_db(attenuation_out_db)
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        self.propagated_labels = spectral_info.label

    def __call__(self, spectral_info):
        # _psig_in records the total signal power of the spectral information before propagation.
        self._psig_in = sum(spectral_info.signal)
        self.propagate(spectral_info)
        # In case of Raman, the resulting loss of the fiber is not equivalent to self.loss
        # because of Raman gain. The resulting loss is:
        # power_out - power_in. We use the total signal power (sum on all channels) to compute
        # this loss.
        loss = round(lin2db(self._psig_in / sum(spectral_info.signal)), 2)
        self.pch_out_db = self.ref_pch_in_dbm - loss
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

        # latency
        spectral_info.latency += self.params.latency

        # apply the attenuation due to the fiber losses
        attenuation_fiber = stimulated_raman_scattering.loss_profile[:spectral_info.number_of_channels, -1]

        spectral_info.apply_attenuation_lin(attenuation_fiber)

        # apply the attenuation due to the output connector loss
        attenuation_out_db = self.params.con_out
        spectral_info.apply_attenuation_db(attenuation_out_db)
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        self.propagated_labels = spectral_info.label


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
        self.target_pch_out_dbm = None
        self.effective_pch_out_db = None
        self.passive = False
        self.att_in = None
        self.effective_gain = self.operational.gain_target
        # self.operational.delta_p is defined by user for reference channel
        # self.delta_p is set with self.operational.delta_p, but it may be changed during design:
        # - if operational.delta_p is None, self.delta_p is computed at design phase
        # - if operational.delta_p can not be applied because of saturation, self.delta_p is recomputed
        # - if power_mode is False, then it is set to None
        self.delta_p = self.operational.delta_p
        # self._delta_p contains computed delta_p during design even if power_mode is False
        self._delta_p = None
        self.tilt_target = self.operational.tilt_target
        self.out_voa = self.operational.out_voa
        self.propagated_labels = [""]

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'type_variety': self.params.type_variety,
                'operational': {
                    'gain_target': round(self.effective_gain, 6) if self.effective_gain else None,
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
        total_pch = pretty_summary_print(per_label_average(self.pch_out_dbm, self.propagated_labels))
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  type_variety:           {self.params.type_variety}',
                          f'  effective gain(dB):     {self.effective_gain:.2f}',
                          f'  (before att_in and before output VOA)',
                          f'  noise figure (dB):      {nf:.2f}',
                          f'  (including att_in)',
                          f'  pad att_in (dB):        {self.att_in:.2f}',
                          f'  Power In (dBm):         {self.pin_db:.2f}',
                          f'  Power Out (dBm):        {self.pout_db:.2f}',
                          '  Delta_P (dB):           ' + (f'{self.delta_p:.2f}'
                                                          if self.delta_p is not None else 'None'),
                          '  target pch (dBm):       ' + (f'{self.target_pch_out_dbm:.2f}'
                                                          if self.target_pch_out_dbm is not None else 'None'),
                          f'  actual pch out (dBm):   {total_pch}',
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
        self.pin_db = watt2dbm(sum(pin))
        # The following should be changed when we have the new spectral information including slot widths.
        # For now, with homogeneous spectrum, we can calculate it as the difference between neighbouring channels.
        self.slot_width = self.channel_freq[1] - self.channel_freq[0]

        """check power saturation and correct effective gain & power accordingly:"""
        # Compute the saturation accounting for actual power at the input of the amp
        self.effective_gain = min(
            self.effective_gain,
            self.params.p_max - self.pin_db
        )

        """check power saturation and correct target_gain accordingly:"""
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
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        self.propagated_labels = spectral_info.label

    def __call__(self, spectral_info):
        self.propagate(spectral_info)
        return spectral_info
