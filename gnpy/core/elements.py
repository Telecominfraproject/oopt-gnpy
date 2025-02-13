#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.elements
==================

Standard network elements which propagate optical spectrum.

A network element is a Python callable. It takes a :class:`.info.SpectralInformation`
object and returns a copy with appropriate fields affected. This structure
represents spectral information that is "propagated" by this network element.
Network elements must have only a local "view" of the network and propagate
:class:`.info.SpectralInformation` using only this information. They should be independent and
self-contained.

Network elements MUST implement two attributes :py:attr:`uid` and :py:attr:`name` representing a
unique identifier and a printable name, and provide the :py:meth:`__call__` method taking a
:class:`SpectralInformation` as an input and returning another :class:`SpectralInformation`
instance as a result.
"""

from copy import deepcopy
from collections import namedtuple
from typing import Union, List
from logging import getLogger
import warnings
from numpy import abs, array, errstate, ones, interp, mean, pi, polyfit, polyval, sum, sqrt, log10, exp, asarray, \
    full, squeeze, zeros, outer, ndarray
from scipy.constants import h, c
from scipy.interpolate import interp1d

from gnpy.core.utils import lin2db, db2lin, arrange_frequencies, snr_sum, per_label_average, pretty_summary_print, \
    watt2dbm, psd2powerdbm, calculate_absolute_min_or_zero, nice_column_str
from gnpy.core.parameters import RoadmParams, FusedParams, FiberParams, PumpParams, EdfaParams, EdfaOperational, \
    MultiBandParams, RoadmPath, RoadmImpairment, TransceiverParams, find_band_name, FrequencyBand
from gnpy.core.science_utils import NliSolver, RamanSolver
from gnpy.core.info import SpectralInformation, muxed_spectral_information, demuxed_spectral_information
from gnpy.core.exceptions import NetworkTopologyError, SpectrumError, ParametersError


_logger = getLogger(__name__)


class Location(namedtuple('Location', 'latitude longitude city region')):
    """Represents a geographical location with latitude, longitude, city, and region."""
    def __new__(cls, latitude: float = 0, longitude: float = 0, city: str = None, region: str = None):
        return super().__new__(cls, latitude, longitude, city, region)


class _Node:
    """Convenience class for providing common functionality of all network elements

    This class is just an internal implementation detail; do **not** assume that all network elements
    inherit from :class:`_Node`.

    :ivar uid: Unique identifier for the node.
    :vartype uid: str
    :ivar name: Printable name of the node.
    :vartype name: str
    :ivar params: Parameters associated with the node.
    :vartype params: Any
    :ivar metadata: Metadata including location.
    :vartype metadata: Dict[str, Any]
    :ivar operational: Operational parameters.
    :vartype operational: Any
    :ivar type_variety: Type variety of the node.
    :vartype type_variety: str
    """
    def __init__(self, uid, name=None, params=None, metadata=None, operational=None, type_variety=None):
        """Constructor method
        """
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
        """Returns the location of the node."""
        return self.metadata['location']
    loc = location

    @property
    def longitude(self):
        """Returns the longitude of the node."""
        return self.location.longitude
    lng = longitude

    @property
    def latitude(self):
        """Returns the latitude of the node."""
        return self.location.latitude
    lat = latitude


class Transceiver(_Node):
    """Represents a logical start for propagation in the optical network.

    :ivar osnr_ase_01nm: OSNR in 0.1 nm bandwidth per carrier in the spectrum.
    :vartype osnr_ase_01nm: numpy.ndarray
    :ivar osnr_ase: OSNR ASE value per carrier in the spectrum.
    :vartype osnr_ase: numpy.ndarray
    :ivar osnr_nli: OSNR NLI value per carrier in the spectrum.
    :vartype osnr_nli: numpy.ndarray
    :ivar snr: Generalized Signal-to-noise ratio per carrier in the spectrum.
    :vartype snr: numpy.ndarray
    :ivar passive: Indicates if the system is passive (default is False).
    :vartype passive: bool
    :ivar baud_rate: Baud rate of each carrier of the emitted spectrum.
    :vartype baud_rate: numpy.ndarray
    :ivar chromatic_dispersion: Chromatic dispersion value per carrier in the spectrum.
    :vartype chromatic_dispersion: numpy.ndarray
    :ivar pmd: PMD value per carrier in the spectrum.
    :vartype pmd: numpy.ndarray
    :ivar pdl: PDL value per carrier in the spectrum.
    :vartype pdl: numpy.ndarray
    :ivar latency: Latency value per carrier in the spectrum.
    :vartype latency: numpy.ndarray
    :ivar penalties: Penalties for various impairments.
    :vartype penalties: Dict[str, float]
    :ivar total_penalty: Total penalty value per carrier in the spectrum.
    :vartype total_penalty: numpy.ndarray
    :ivar propagated_labels: Labels propagated by the transceiver.
    :vartype propagated_labels: numpy.ndarray[str]
    :ivar tx_power: Transmit power.
    :vartype tx_power: numpy.ndarray
    :ivar design_bands: Design bands parameters.
    :vartype design_bands: list
    :ivar per_degree_design_bands: Per degree design bands parameters.
    :vartype per_degree_design_bands: dict
    """
    def __init__(self, *args, params=None, **kwargs):
        """Constructor method
        """
        if not params:
            params = {}
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                super().__init__(*args, params=TransceiverParams(**params), **kwargs)
                if caught_warnings:
                    msg = f'In Transceiver {kwargs["uid"]}: {caught_warnings[0].message}'
                    _logger.warning(msg)
        except ParametersError as e:
            msg = f'Config error in {kwargs["uid"]}: {e}'
            _logger.critical(msg)
            raise ParametersError(msg) from e

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
        self.tx_power = None
        self.design_bands = self.params.design_bands
        self.per_degree_design_bands = self.params.per_degree_design_bands

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
        """Computes the SNR penalty given the impairment value.

        :param impairment_value: The impairment value.
        :type impairment_value: float
        :param boundary_list: The boundary list for penalties.
        :type boundary_list: Dict[str, Any]

        :return float: The computed penalty.
        """
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
            if s is not None:
                snr_added += db2lin(-s)
        snr_added = -lin2db(snr_added)
        self.osnr_ase = snr_sum(self.raw_osnr_ase, self.baud_rate, snr_added)
        self.snr = snr_sum(self.raw_snr, self.baud_rate, snr_added)
        self.osnr_ase_01nm = snr_sum(self.raw_osnr_ase_01nm, 12.5e9, snr_added)
        self.snr_01nm = snr_sum(self.raw_snr_01nm, 12.5e9, snr_added)

    @property
    def to_json(self):
        """Converts the transceiver's state to a JSON-compatible dictionary.

        :return Dict[str, Any]: JSON representation of the transceiver.
        """
        return {'uid': self.uid,
                'type': type(self).__name__,
                'metadata': {
                    'location': self.metadata['location']._asdict()
                }
                }

    def __repr__(self):
        """Returns a string representation of the transceiver.

        :return str: String representation of the transceiver.
        """
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
        """Returns a formatted string representation of the transceiver.

        :return str: Formatted string representation of the transceiver.
        """
        if self.snr is None or self.osnr_ase is None:
            return f'{type(self).__name__} {self.uid}'

        snr = per_label_average(self.snr, self.propagated_labels)
        osnr_ase = per_label_average(self.osnr_ase, self.propagated_labels)
        osnr_ase_01nm = per_label_average(self.osnr_ase_01nm, self.propagated_labels)
        snr_01nm = per_label_average(self.snr_01nm, self.propagated_labels)
        tx_power_dbm = per_label_average(watt2dbm(self.tx_power), self.propagated_labels)
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
                            f'  Latency (ms):              {latency:.2f}',
                            f'  Actual pch out (dBm):      {pretty_summary_print(tx_power_dbm)}'])

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
        """Propagates spectral information through the transceiver:
        i) computes the accumulated impairments and convert them into penalties for each cariier,
        ii) computes the resulting OSNR and GSNR per carrier and records the values into the attributes

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInfomation)
        :return SpectralInformation: The updated spectral information object.
        """
        self.tx_power = spectral_info.tx_power
        self._calc_snr(spectral_info)
        self._calc_cd(spectral_info)
        self._calc_pmd(spectral_info)
        self._calc_pdl(spectral_info)
        self._calc_latency(spectral_info)
        return spectral_info


class Roadm(_Node):
    """Represents a Reconfigurable Optical Add-Drop Multiplexer (ROADM).

    :ivar ref_pch_out_dbm: Reference output power in dBm.
    :vartype ref_pch_out_dbm: float
    :ivar loss: Total loss experienced by the ROADM.
    :vartype loss: float
    :ivar loss_pch_db: Loss per channel in dB.
    :vartype loss_pch_db: numpy.ndarray
    :ivar ref_effective_loss: Effective loss for a reference carrier.
    :vartype ref_effective_loss: float
    :ivar passive: True, indicates that the ROADM is passive.
    :vartype passive: bool
    :ivar restrictions: Restrictions on the ROADM.
    :vartype restrictions: dict
    :ivar propagated_labels: Labels propagated by the ROADM.
    :vartype propagated_labels: numpy.ndarray[str]
    :ivar target_pch_out_dbm: Target output power in dBm.
    :vartype target_pch_out_dbm: float
    :ivar target_psd_out_mWperGHz: Target PSD output in mW/GHz.
    :vartype target_psd_out_mWperGHz: float
    :ivar target_out_mWperSlotWidth: Target output power per slot width.
    :vartype target_out_mWperSlotWidth: float
    :ivar per_degree_pch_out_dbm: Per degree target output power.
    :vartype per_degree_pch_out_dbm: Dict[str, float]
    :ivar per_degree_pch_psd: Per degree target PSD output.
    :vartype per_degree_pch_psd: Dict[str, float]
    :ivar per_degree_pch_psw: Per degree target output per slot width.
    :vartype per_degree_pch_psw: Dict[str, float]
    :ivar ref_pch_in_dbm: Reference input power in dBm.
    :vartype ref_pch_in_dbm: Dict[str, float]
    :ivar ref_carrier: Reference carrier.
    :vartype ref_carrier: ReferenceCarrier
    :ivar roadm_paths: Internal paths for the ROADM.
    :vartype roadm_paths: Dict[str, Any]
    :ivar roadm_path_impairments: Impairment profiles for the ROADM paths.
    :vartype roadm_path_impairments: Dict
    :ivar per_degree_impairments: Per degree impairments.
    :vartype per_degree_impairments: Dict[str, Any]
    :ivar design_bands: Design bands parameters.
    :vartype design_bands: List[Dict]
    :ivar per_degree_design_bands: Per degree design bands parameters.
    :vartype per_degree_design_bands: Dict[str, Dict]
    """
    def __init__(self, *args, params=None, **kwargs):
        """Constructor method
        """
        # pylint: disable=C0103
        if not params:
            params = {}
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                super().__init__(*args, params=RoadmParams(**params), **kwargs)
                if caught_warnings:
                    msg = f'In ROADM {kwargs["uid"]}: {caught_warnings[0].message}'
                    _logger.warning(msg)
        except ParametersError as e:
            msg = f'Config error in {kwargs["uid"]}: {e}'
            raise ParametersError(msg) from e

        # Target output power for the reference carrier, can only be computed on the fly, because it depends
        # on the path, since it depends on the equalization definition on the degree.
        self.ref_pch_out_dbm = None
        self.loss = 0  # auto-design interest
        self.loss_pch_db = None

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
        # Define the nature of from-to internal connection: express-path, drop-path, add-path
        # roadm_paths contains a list of RoadmPath object for each path crossing the ROADM
        self.roadm_paths = []
        # roadm_path_impairments contains a dictionnary of impairments profiles corresponding to type_variety
        # first listed add, drop an express constitute the default
        self.roadm_path_impairments = self.params.roadm_path_impairments
        # per degree definitions, in case some degrees have particular deviations with respect to default.
        self.per_degree_impairments = {f'{i["from_degree"]}-{i["to_degree"]}': {"from_degree": i["from_degree"],
                                                                                "to_degree": i["to_degree"],
                                                                                "impairment_id": i["impairment_id"]}
                                       for i in self.params.per_degree_impairments}
        self.design_bands = deepcopy(self.params.design_bands)
        self.per_degree_design_bands = deepcopy(self.params.per_degree_design_bands)

    @property
    def to_json(self):
        """Converts the ROADM's state to a JSON-compatible dictionary.

        :return Dict[str, Any]: JSON representation of the ROADM.
        """
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
            'type_variety': self.type_variety,
            'params': {
                equalisation: value,
                'restrictions': self.restrictions
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
        if self.per_degree_impairments:
            to_json['params']['per_degree_impairments'] = list(self.per_degree_impairments.values())

        if self.params.design_bands is not None:
            if len(self.params.design_bands) > 1:
                to_json['params']['design_bands'] = self.params.design_bands
        if self.params.per_degree_design_bands:
            to_json['params']['per_degree_design_bands'] = self.params.per_degree_design_bands
        return to_json

    def __repr__(self):
        """Returns a string representation of the ROADM.

        :return str: String representation of the ROADM.
        """
        return f'{type(self).__name__}(uid={self.uid!r}, loss={self.loss!r})'

    def __str__(self):
        """Returns a formatted string representation of the ROADM.

        :return str: Formatted string representation of the ROADM.
        """
        if self.ref_effective_loss is None:
            return f'{type(self).__name__} {self.uid}'

        total_pch = pretty_summary_print(per_label_average(self.pch_out_dbm, self.propagated_labels))
        total_loss = pretty_summary_print(per_label_average(self.loss_pch_db, self.propagated_labels))
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  Type_variety:            {self.type_variety}',
                          f'  Reference loss (dB):     {self.ref_effective_loss:.2f}',
                          f'  Actual loss (dB):        {total_loss}',
                          f'  Reference pch out (dBm): {self.ref_pch_out_dbm:.2f}',
                          f'  Actual pch out (dBm):    {total_pch}'])

    def get_roadm_target_power(self, spectral_info: SpectralInformation = None) -> Union[float, ndarray]:
        """Computes the power in dBm for a reference carrier or for a spectral information.
        power is computed based on equalization target.
        if spectral_info baud_rate is baud_rate = [32e9, 42e9, 64e9, 42e9, 32e9], and
        target_pch_out_dbm is defined to -20 dbm, then the function returns an array of powers
        [-20, -20, -20, -20, -20]
        if target_psd_out_mWperGHz is defined instead with 3.125e-4mW/GHz then it returns
        [-20, -18.819, -16.9897, -18.819, -20]
        if instead a reference_baud_rate is defined, the functions computes the result for a
        single reference carrier whose baud_rate is reference_baudrate.

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation, optional
        :return: Target power in dBm.
        :rtype: Union[float, ndarray]
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

        :param degree: The degree identifier.
        :type degree: str

        :return float: Target power in dBm
        """
        if degree in self.per_degree_pch_out_dbm:
            return self.per_degree_pch_out_dbm[degree]
        if degree in self.per_degree_pch_psd:
            return psd2powerdbm(self.per_degree_pch_psd[degree], self.ref_carrier.baud_rate)
        if degree in self.per_degree_pch_psw:
            return psd2powerdbm(self.per_degree_pch_psw[degree], self.ref_carrier.slot_width)
        return self.get_roadm_target_power()

    def get_per_degree_power(self, degree, spectral_info):
        """Get the target power in dBm out of ROADM degree for the spectral information
        If no equalization is defined on this degree use the ROADM level one.

        :param degree: The degree identifier.
        :type degree: str
        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation

        :return float: Target power in dBm.
        """
        if degree in self.per_degree_pch_out_dbm:
            return self.per_degree_pch_out_dbm[degree]
        if degree in self.per_degree_pch_psd:
            return psd2powerdbm(self.per_degree_pch_psd[degree], spectral_info.baud_rate)
        if degree in self.per_degree_pch_psw:
            return psd2powerdbm(self.per_degree_pch_psw[degree], spectral_info.slot_width)
        return self.get_roadm_target_power(spectral_info=spectral_info)

    def propagate(self, spectral_info, degree, from_degree):
        """Equalization targets are read from topology file if defined and completed with default
        definition of the library.
        If the input power is lower than the target one, use the input power minus the ROADM loss
        if is exists, because a ROADM doesn't amplify, it can only attenuate.
        There is no difference for add or express : the same target is applied.
        For the moment propagate operates with spectral info carriers all having the same source or destination.

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation
        :param degree: The egress degree.
        :type degree: str
        :param from_degree: The ingress degree.
        :type from_degree: str
        """
        # record input powers to compute the actual loss at the end of the process
        input_power_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        # apply min ROADM loss if it exists
        roadm_maxloss_db = self.get_impairment('roadm-maxloss', spectral_info.frequency, from_degree, degree)
        spectral_info.apply_attenuation_db(roadm_maxloss_db)
        # records the total power after applying minimum loss
        net_input_power_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)
        # find the target power for the reference carrier
        ref_per_degree_pch = self.get_per_degree_ref_power(degree)
        # find the target powers for each signal carrier
        per_degree_pch = self.get_per_degree_power(degree, spectral_info=spectral_info)

        # Definition of ref_pch_out_dbm for the reference channel:
        # Depending on propagation upstream from this ROADM, the input power might be smaller than
        # the target power out configured for this ROADM degree's egress. Since ROADM does not amplify,
        # the power out of the ROADM for the ref channel is the min value between target power and input power.
        ref_pch_in_dbm = self.ref_pch_in_dbm[from_degree]
        # Calculate the output power for the reference channel (only for visualization)
        self.ref_pch_out_dbm = min(ref_pch_in_dbm - max(roadm_maxloss_db), ref_per_degree_pch)

        # Definition of effective_loss:
        # Optical power of carriers are equalized by the ROADM, so that the experienced loss is not the same for
        # different carriers. effective_loss records the loss for the reference carrier.
        # Calculate the effective loss for the reference channel
        self.ref_effective_loss = ref_pch_in_dbm - self.ref_pch_out_dbm

        # Calculate the target power per channel according to the equalization policy
        target_power_per_channel = per_degree_pch + spectral_info.delta_pdb_per_channel
        # Computation of the correction according to equalization policy
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
        correction = calculate_absolute_min_or_zero(net_input_power_dbm - target_power_per_channel)
        new_target = target_power_per_channel - correction
        delta_power = net_input_power_dbm - new_target

        spectral_info.apply_attenuation_db(delta_power)

        # Update the PMD information
        pmd_impairment = self.get_impairment('roadm-pmd', spectral_info.frequency, from_degree, degree)
        spectral_info.pmd = sqrt(spectral_info.pmd ** 2 + pmd_impairment ** 2)

        # Update the PMD information
        pdl_impairment = self.get_impairment('roadm-pdl', spectral_info.frequency, from_degree, degree)
        spectral_info.pdl = sqrt(spectral_info.pdl ** 2 + pdl_impairment ** 2)

        # Update the per channel power with the result of propagation
        self.pch_out_dbm = watt2dbm(spectral_info.signal + spectral_info.nli + spectral_info.ase)

        # Update the loss per channel and the labels
        self.loss_pch_db = input_power_dbm - self.pch_out_dbm
        self.propagated_labels = spectral_info.label

    def set_roadm_paths(self, from_degree, to_degree, path_type, impairment_id=None):
        """set internal path type: express, drop or add with corresponding impairment

        If no impairment id is defined, then use the first profile that matches the path_type in the
        profile dictionnary.

        :param from_degree: The ingress degree.
        :type from_degree: str
        :param to_degree: The egress degree.
        :type to_degree: str
        :param path_type: Type of the path (express, drop, add).-
        :type path_type: str
        :param impairment_id: Impairment profile ID. This parameter is optional.
        :type impairment_id: int, optional
        """
        # initialize impairment with params.pmd, params.cd
        # if more detailed parameters are available for the Roadm, the use them instead
        roadm_global_impairment = {
            'impairment': [{
                'roadm-pmd': self.params.pmd,
                'roadm-pdl': self.params.pdl,
                'frequency-range': {
                    'lower-frequency': None,
                    'upper-frequency': None
                }}]}
        if path_type in ['add', 'drop']:
            # without detailed imparments, we assume that add OSNR contribution is the same as drop contribution
            # add_drop_osnr_db = - 10log10(1/add_osnr + 1/drop_osnr) with add_osnr = drop_osnr
            # = add_osnr_db + 10log10(2)
            roadm_global_impairment['impairment'][0]['roadm-osnr'] = self.params.add_drop_osnr + lin2db(2)
        impairment = RoadmImpairment(roadm_global_impairment)

        if impairment_id is None:
            # get the first item in the type variety that matches the path_type
            for path_impairment_id, path_impairment in self.roadm_path_impairments.items():
                if path_impairment.path_type == path_type:
                    impairment = path_impairment
                    impairment_id = path_impairment_id
                    break
            # at this point, path_type is not part of roadm_path_impairment, impairment and impairment_id are None
        else:
            if impairment_id in self.roadm_path_impairments:
                impairment = self.roadm_path_impairments[impairment_id]
            else:
                msg = f'ROADM {self.uid}: impairment profile id {impairment_id} is not defined in library'
                raise NetworkTopologyError(msg)
        # print(from_degree, to_degree, path_type)
        self.roadm_paths.append(RoadmPath(from_degree=from_degree, to_degree=to_degree, path_type=path_type,
                                          impairment_id=impairment_id, impairment=impairment))

    def get_roadm_path(self, from_degree: str, to_degree: str):
        """Get internal path type impairment.

        :param from_degree: The ingress degree.
        :type from_degree: str
        :param to_degree: The egress degree.
        :type to_degree: str

        :return Any: The roadm path object.
        """
        for roadm_path in self.roadm_paths:
            if roadm_path.from_degree == from_degree and roadm_path.to_degree == to_degree:
                return roadm_path
        msg = f'Could not find from_degree-to_degree {from_degree}-{to_degree} path in ROADM {self.uid}'
        raise NetworkTopologyError(msg)

    def get_per_degree_impairment_id(self, from_degree: str, to_degree: str) -> Union[int, None]:
        """returns the id of the impairment if the degrees are in the per_degree tab.

        :param from_degree: The ingress degree.
        :type from_degree: str
        :param to_degree: The egress degree.
        :type to_degree: str

        :return Union[int, None]: The impairment ID or None if not found.
        """
        if f'{from_degree}-{to_degree}' in self.per_degree_impairments.keys():
            return self.per_degree_impairments[f'{from_degree}-{to_degree}']["impairment_id"]
        return None

    def get_path_type_per_id(self, impairment_id: int) -> Union[str, None]:
        """returns the path_type of the impairment if the id is defined

        :param impairment_id: The impairment ID.
        :type impairment_id: int

        :return Union[str, None]: The path type or None if not found.
        """
        if impairment_id in self.roadm_path_impairments.keys():
            return self.roadm_path_impairments[impairment_id].path_type
        return None

    def get_impairment(self, impairment: str, frequency_array: array, from_degree: str, degree: str) \
            -> array:
        """
        Retrieves the specified impairment values for the given frequency array.

        :param impairment: The type of impairment to retrieve (e.g., roadm-pmd, roadm-maxloss).
        :type impairment: str
        :param frequency_array: The frequencies at which to check for impairments.
        :type frequency_array: numpy.ndarray
        :param from_degree: The ingress degree for the ROADM internal path.
        :type from_degree: str
        :param degree: The egress degree for the ROADM internal path.
        :type degree: str

        :return array: An array of impairment values for the specified frequencies.
        """
        result = []
        impairment_per_band = self.get_roadm_path(from_degree, degree).impairment.impairments
        for frequency in frequency_array:
            for item in impairment_per_band:
                f_min = item['frequency-range']['lower-frequency']
                f_max = item['frequency-range']['upper-frequency']
                if (f_min is None or f_min <= frequency <= f_max):
                    item[impairment] = item.get(impairment, RoadmImpairment.default_values[impairment])
                    if item[impairment] is not None:
                        result.append(item[impairment])
                        break  # Stop searching after the first match for this frequency
        if result:
            return array(result)

    def __call__(self, spectral_info: SpectralInformation, degree: str, from_degree: str) -> SpectralInformation:
        """Propagate from_degree to degree in the ROADM

        param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation
        :param degree: The egress degree.
        :type degree: str
        :param from_degree: The ingress degree.
        :type from_degree: str

        :return SpectralInformation: The updated spectral information object.
        """
        self.propagate(spectral_info, degree=degree, from_degree=from_degree)
        return spectral_info


class Fused(_Node):
    """Represents a fused optical element in the network.

    :ivar loss (float): Loss experienced by the fused element.
    :ivar passive (bool): Indicates if the fused element is passive.
    """
    def __init__(self, *args, params=None, **kwargs):
        """Constructor method
        """
        if not params:
            params = {}
        super().__init__(*args, params=FusedParams(**params), **kwargs)
        self.loss = self.params.loss
        self.passive = True

    @property
    def to_json(self):
        """Converts the fused element's state to a JSON-compatible dictionary.

        :return Dict[str, Any]: JSON representation of the fused element.
        """
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
        """Returns a string representation of the fused element.

        :return str: String representation of the fused element.
        """
        return f'{type(self).__name__}(uid={self.uid!r}, loss={self.loss!r})'

    def __str__(self):
        """Returns a formatted string representation of the fused element.

        :return str: Formatted string representation of the fused element.
        """
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  loss (dB): {self.loss:.2f}'])

    def propagate(self, spectral_info):
        """Applies loss to the spectral information.

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation
        """
        spectral_info.apply_attenuation_db(self.loss)

    def __call__(self, spectral_info):
        """Propagates spectral information through the fused element.

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation

        :return SpectralInformation: The updated spectral information object.
        """
        self.propagate(spectral_info)
        return spectral_info


class Fiber(_Node):
    """Represents an optical fiber element in the network.

    :ivar pch_out_db: Output power per channel in dBm.
    :vartype pch_out_db: float
    :ivar passive: Indicates if the fiber is passive.
    :vartype passive: bool
    :ivar propagated_labels: Labels propagated by the fiber.
    :vartype propagated_labels: List[str]
    """
    def __init__(self, *args, params=None, **kwargs):
        """Constructor method
        """
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
        """Converts the fiber's state to a JSON-compatible dictionary.
        Adapts the json export: scalar or vector.

        :return Dict[str, Any]: JSON representation of the fiber.
        """
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
        """Returns a string representation of the fiber.

        :return str: String representation of the fiber.
        """
        return f'{type(self).__name__}(uid={self.uid!r}, ' \
            f'length={round(self.params.length * 1e-3, 1)!r}km, ' \
            f'loss={round(self.loss, 1)!r}dB)'

    def __str__(self):
        """Returns a formatted string representation of the fiber.

        :return str: Formatted string representation of the fiber.
        """
        if self.pch_out_db is None:
            return f'{type(self).__name__} {self.uid}'

        total_pch = pretty_summary_print(per_label_average(self.pch_out_dbm, self.propagated_labels))
        return '\n'.join([f'{type(self).__name__}          {self.uid}',
                          f'  type_variety:                {self.type_variety}',
                          f'  length (km):                 {self.params.length * 1e-3:.2f}',
                          f'  pad att_in (dB):             {self.params.att_in:.2f}',
                          f'  total loss (dB):             {self.loss:.2f}',
                          f'  (includes conn loss (dB) in: {self.params.con_in:.2f} out: {self.params.con_out:.2f})',
                          '  (conn loss out includes EOL margin defined in eqpt_config.json)',
                          f'  reference pch out (dBm):     {self.pch_out_db:.2f}',
                          f'  actual pch out (dBm):        {total_pch}'])

    def interpolate_parameter_over_spectrum(self, parameter, ref_frequency, spectrum_frequency, name):
        """Interpolates loss coefficient value given the input frequency.

        :param parameter (ndarray): The parameter to interpolate.
        :param ref_frequency (ndarray): Reference frequencies.
        :param spectrum_frequency (ndarray): Frequencies of the spectrum.
        :param name (str): Name of the parameter for error messages.

        :return ndarray: Interpolated values.
        """
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
        """
        Returns the loss coefficient (of a fibre) which can be uniform,
        or made frequency-dependent defined via a dictionnary-model per instance
        in the topology-file, or via a list-model ('LUT') in the equipment-file.

        If a list-model is declared, then the legacy 'loss_coef' scalar is used to
        offset the provided LUT-model for the given self.params.ref_frequency' such
        that at the 'self.params.ref_frequency' the offset model values the 'loss_coef' scalar;
        in case the 'loss_coef' scalar is not defined then no offset is applied.

        In case a dictionary-model as well as a list-model are declared, then the legacy
        dictionary-based model has priority.

        :param frequency (Union[float, ndarray]): Frequency at which to compute the loss coefficient.

        :return ndarray: Loss coefficient values.
        """
        frequency = asarray(frequency)
        if self.params.loss_coef.size > 1:
            loss_coef = self.interpolate_parameter_over_spectrum(self.params.loss_coef, self.params.f_loss_ref,
                                                                 frequency, 'Loss Coefficient')
        else:
            loss_coef = full(frequency.size, self.params.loss_coef)
        return squeeze(loss_coef)

    @property
    def loss(self):
        """total loss including padding att_in: useful for polymorphism with roadm loss

        :return float: Total loss in dB.
        """
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
                            self.beta2(self.params.f_dispersion_ref), 2)[1] / (2 * pi)
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
        """Propagate through the fiber

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation

        :return SpectralInformation: The updated spectral information object.
        """
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
    """Class representing a Raman fiber in a network.

    Inherits from the Fiber class and adds specific parameters and methods for Raman amplification.

    :ivar raman_pumps: A tuple of pump parameters for the Raman amplification.
    :vartype raman_pumps: Tuple[PumpParams]
    :ivar temperature: The operational temperature of the Raman fiber.
    :vartype temperature: float
    :raises NetworkTopologyError: If the fiber is defined as a RamanFiber without operational parameters,
                                  or if required operational parameters are missing.
    """
    def __init__(self, *args, params=None, **kwargs):
        """Constructor method
        """
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
        """Converts the RamanFiber's state to a JSON-compatible dictionary.
        Adapts the json export: scalar or vector.

        :return Dict[str, Any]: JSON representation of the RamanFiber.
        """
        return dict(super().to_json, operational=self.operational)

    def __str__(self):
        return super().__str__() + f'\n  reference gain (dB):         {round(self.estimated_gain, 2)}' \
            + f'\n  actual gain (dB):            {round(self.actual_raman_gain, 2)}'

    def propagate(self, spectral_info: SpectralInformation):
        """Modifies the spectral information computing the attenuation, the non-linear interference generation,
        the CD and PMD accumulation.
        """
        # apply the attenuation due to the input connector loss
        pin = watt2dbm(sum(spectral_info.signal))
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
        pout = watt2dbm(sum(spectral_info.signal))
        self.actual_raman_gain = self.loss + pout - pin


class Edfa(_Node):
    """Class representing an Erbium-Doped Fiber Amplifier (EDFA).

    This class models the behavior of an EDFA, including its parameters, operational characteristics,
    and methods for propagation of spectral information.

    :ivar variety_list: A list of type_variety associated with the amplifier.
    :vartype variety_list: Union[List[str], None]
    :ivar interpol_dgt: Interpolated dynamic gain tilt defined per frequency on the amplifier band.
    :vartype interpol_dgt: numpy.ndarray
    :ivar interpol_gain_ripple: Interpolated gain ripple.
    :vartype interpol_gain_ripple: numpy.ndarray
    :ivar interpol_nf_ripple: Interpolated noise figure ripple.
    :vartype interpol_nf_ripple: numpy.ndarray
    :ivar channel_freq: SI channel frequencies.
    :vartype channel_freq: numpy.ndarray
    :ivar nf: Noise figure in dB at the operational gain target.
    :vartype nf: numpy.ndarray
    :ivar gprofile: Gain profile of the amplifier.
    :vartype gprofile: numpy.ndarray
    :ivar pin_db: Input power in dBm.
    :vartype pin_db: float
    :ivar nch: Number of channels.
    :vartype nch: int
    :ivar pout_db: Output power in dBm.
    :vartype pout_db: float
    :ivar target_pch_out_dbm: Target output power per channel in dBm.
    :vartype target_pch_out_dbm: float
    :ivar effective_pch_out_db: Effective output power per channel in dBm.
    :vartype effective_pch_out_db: float
    :ivar passive: Indicates if the fiber is passive.
    :vartype passive: bool
    :ivar att_in: Input attenuation in dB.
    :vartype att_in: float
    :ivar effective_gain: Effective gain of the amplifier.
    :vartype effective_gain: float
    :ivar delta_p: Delta power defined by the operational parameters.
    :vartype delta_p: float
    :ivar _delta_p: Computed delta power during design.
    :vartype _delta_p: float
    :ivar tilt_target: Target tilt defined per wavelength on the amplifier band.
    :vartype tilt_target: float
    :ivar out_voa: Output variable optical attenuator setting.
    :vartype out_voa: float
    :ivar in_voa: Input variable optical attenuator setting.
    :vartype in_voa: float
    :ivar propagated_labels: Labels propagated by the amplifier.
    :vartype propagated_labels: numpy.ndarray
    :raises ParametersError: If there are conflicting amplifier definitions for the same frequency
        band during initialization.
    :raises ValueError: If the input spectral information does not match any defined amplifier bands
        during propagation.
    """
    def __init__(self, *args, params=None, operational=None, **kwargs):
        """Constructor method for initializing the EDFA.

        :param args: Positional arguments for the parent class.
        :param params: Parameters for the EDFA, defaults to an empty dictionary if None.
        :type params: dict, optional
        :param operational: Operational parameters for the EDFA, defaults to an empty dictionary if None.
        :type operational: dict, optional
        """
        if params is None:
            params = {}
        if operational is None:
            operational = {}
        self.variety_list = kwargs.pop('variety_list', None)
        try:
            super().__init__(*args, params=EdfaParams(**params), operational=EdfaOperational(**operational), **kwargs)
        except ParametersError as e:
            raise ParametersError(f'{kwargs["uid"]}: {e}') from e
        self.interpol_dgt = None  # interpolated dynamic gain tilt defined per frequency on amp band
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
        self.tilt_target = self.operational.tilt_target  # defined per lambda on the amp band
        self.out_voa = self.operational.out_voa
        self.propagated_labels = [""]

    @property
    def to_json(self):
        """Converts the Edfa's state to a JSON-compatible dictionary.
        Adapts the json export: scalar or vector.

        :return Dict[str, Any]: JSON representation of the Edfa.
        """
        _to_json = {
            'uid': self.uid,
            'type': type(self).__name__,
            'type_variety': self.params.type_variety,
            'operational': {
                'gain_target': round(self.effective_gain, 6) if self.effective_gain else None,
                'delta_p': self.delta_p,
                'tilt_target': round(self.tilt_target, 5) if self.tilt_target is not None else None,
                # defined per lambda on the amp band
                'out_voa': self.out_voa
            },
            'metadata': {
                'location': self.metadata['location']._asdict()
            }
        }
        return _to_json

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
                          '  (before att_in and before output VOA)',
                          f'  tilt-target(dB)         {self.tilt_target if self.tilt_target else 0:.2f}',
                          # avoids -0.00 value for tilt_target
                          f'  noise figure (dB):      {nf:.2f}',
                          '  (including att_in)',
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

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation
        """
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

        # check power saturation and correct effective gain & power accordingly:
        # Compute the saturation accounting for actual power at the input of the amp
        self.effective_gain = min(
            self.effective_gain,
            self.params.p_max - self.pin_db
        )

        # check power saturation and correct target_gain accordingly:
        self.nf = self._calc_nf()
        self.gprofile = self._gain_profile(pin)

        pout = (pin + self.noise_profile(spectral_info)) * db2lin(self.gprofile)
        self.pout_db = lin2db(sum(pout * 1e3))
        # ase & nli are only calculated in signal bandwidth
        #    pout_db is not the absolute full output power (negligible if sufficient channels)

    def _nf(self, type_def, nf_model, nf_fit_coeff, gain_min, gain_flatmax, gain_target):
        # if hybrid raman, use edfa_gain_flatmax attribute, else use gain_flatmax
        # gain_flatmax = getattr(params, 'edfa_gain_flatmax', params.gain_flatmax)
        # pylint: disable=C0103
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
        return self.interpol_nf_ripple + nf_avg  # input VOA = 1 for 1 NF degradation

    def noise_profile(self, spectral_info: SpectralInformation):
        """Computes amplifier ASE noise integrated over the signal bandwidth. This is calculated at amplifier input.

        :return: the ASE power in W in the signal bandwidth bw for 96 channels
        :rtype: numpy.ndarray

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

        # Calculate the target slope defined per frequency on the amp band
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
        """add ASE noise to the propagating carriers of :class:`.info.SpectralInformation`

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation
        """
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
        """Propagate through the amplifier.

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation

        :return SpectralInformation: The updated spectral information object.
        """
        # filter out carriers outside the amplifier band
        band = next(b for b in self.params.bands)
        spectral_info = demuxed_spectral_information(spectral_info, band)
        if spectral_info.carriers:
            self.propagate(spectral_info)
            return spectral_info
        raise ValueError(f'Amp {self.uid} Defined propagation band does not match amplifiers band.')


class Multiband_amplifier(_Node):
    """Represents a multiband amplifier that manages multiple amplifiers across different frequency bands.

    This class allows for the initialization and management of amplifiers, each associated with a specific
    frequency band. It provides methods for signal propagation through the amplifiers and for exporting
    to JSON format.

    :param amplifiers: list of dictionaries, each containing parameters for setting an individual amplifier.
    :type amplifiers: List[Dict]
    :param params: dictionary of parameters for the multiband amplifier, which must include necessary
        configuration settings.
    :type params: Dict
    :param args: Additional positional and keyword arguments passed to the parent class `_Node`.
    :param kwargs: Additional positional and keyword arguments passed to the parent class `_Node`.
    :ivar variety_list: A list of varieties associated with the amplifier.
    :vartype variety_list: list
    :ivar amplifiers: A dictionary mapping band names to their corresponding amplifier instances.
    :vartype amplifiers: dict
    :raises ParametersError: If there are conflicting amplifier definitions for the same frequency
        band during initialization.
    :raises ValueError: If the input spectral information does not match any defined amplifier bands
        during propagation.
    """
    # pylint: disable=C0103
    # separate the top level type_variety from kwargs to avoid having multiple type_varieties on each element processing
    def __init__(self, *args, amplifiers: List[dict], params: dict, **kwargs):
        """Constructor method
        """
        self.variety_list = kwargs.pop('variety_list', None)
        try:
            super().__init__(params=MultiBandParams(**params), **kwargs)
        except ParametersError as e:
            raise ParametersError(f'{kwargs["uid"]}: {e}') from e
        self.amplifiers = {}
        if 'type_variety' in kwargs:
            kwargs.pop('type_variety')
        self.passive = False
        for amp_dict in amplifiers:
            # amplifiers dict uses default names as key to represent the band
            amp = Edfa(**amp_dict, **kwargs)
            band = next(b for b in amp.params.bands)
            band_name = find_band_name(FrequencyBand(f_min=band["f_min"], f_max=band["f_max"]))
            if band_name not in self.amplifiers and band not in self.params.bands:
                self.params.bands.append(band)
                self.amplifiers[band_name] = amp
            elif band_name not in self.amplifiers and band in self.params.bands:
                self.amplifiers[band_name] = amp
            else:
                raise ParametersError(f'{kwargs["uid"]}: has more than one amp defined for the same band')

    def __call__(self, spectral_info: SpectralInformation):
        """propagates in each amp and returns the muxed spectrum

        :param spectral_info: The spectral information object.
        :type spectral_info: SpectralInformation

        :return SpectralInformation: The updated spectral information object.
        """
        out_si = []
        for _, amp in self.amplifiers.items():
            si = demuxed_spectral_information(spectral_info, amp.params.bands[0])
            # if spectral_info frequencies are outside amp band, si is None
            if si:
                si = amp(si)
                out_si.append(si)
        if not out_si:
            raise ValueError('Defined propagation band does not match amplifiers band.')
        return muxed_spectral_information(out_si)

    @property
    def to_json(self):
        """Converts the MultibandAmplifier's state to a JSON-compatible dictionary.

        :return Dict[str, Any]: JSON representation of the MultibandAmplifier.
        """
        return {'uid': self.uid,
                'type': type(self).__name__,
                'type_variety': self.type_variety,
                'amplifiers': [{
                    'type_variety': amp.type_variety,
                    'operational': {
                        'gain_target': round(amp.effective_gain, 6),
                        'delta_p': amp.delta_p,
                        'tilt_target': amp.tilt_target,
                        'out_voa': amp.out_voa
                    }} for amp in self.amplifiers.values()
                ],
                'metadata': {
                    'location': self.metadata['location']._asdict()
                }
                }

    def __repr__(self):
        """Returns a string representation of the MultibandAmplifier.

        :return str: String representation of the MultibandAmplifier.
        """
        return (f'{type(self).__name__}(uid={self.uid!r}, '
                f'type_variety={self.type_variety!r}, ')

    def __str__(self):
        """Returns a formatted string representation of the MultibandAmplifier.

        :return str: Formatted string representation of the MultibandAmplifier.
        """
        amp_str = [f'{type(self).__name__} {self.uid}',
                   f'  type_variety:           {self.type_variety}']
        multi_str_data = []
        max_width = 0
        for amp in self.amplifiers.values():
            lines = amp.__str__().split('\n')
            # start at index 1 to remove uid from each amp list of strings
            # records only if amp is used ie si has frequencies in amp) otherwise there is no other string than the uid
            if len(lines) > 1:
                max_width = max(max_width, max([len(line) for line in lines[1:]]))
                multi_str_data.append(lines[1:])
        # multi_str_data contains lines with each amp str, instead we want to print per column: transpose the string
        transposed_data = list(map(list, zip(*multi_str_data)))
        return '\n'.join(amp_str) + '\n' + nice_column_str(data=transposed_data, max_length=max_width + 2, padding=3)
