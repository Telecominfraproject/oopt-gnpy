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

from numpy import abs, arange, array, divide, errstate, ones, interp, mean, pi, polyfit, polyval, sum, sqrt
from scipy.constants import h, c
from collections import namedtuple

from gnpy.core.utils import lin2db, db2lin, arrange_frequencies, snr_sum
from gnpy.core.parameters import FiberParams, PumpParams
from gnpy.core.science_utils import NliSolver, RamanSolver, propagate_raman_fiber, _psi


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
        self.chromatic_dispersion = [carrier.chromatic_dispersion * 1e3 for carrier in spectral_info.carriers]

    def _calc_pmd(self, spectral_info):
        """Updates the Transceiver property with the PMD of the received channels. PMD in ps.
        """
        self.pmd = [carrier.pmd*1e12 for carrier in spectral_info.carriers]

    def _calc_snr(self, spectral_info):
        with errstate(divide='ignore'):
            self.baud_rate = [c.baud_rate for c in spectral_info.carriers]
            ratio_01nm = [lin2db(12.5e9 / b_rate) for b_rate in self.baud_rate]
            # set raw values to record original calculation, before update_snr()
            self.raw_osnr_ase = [lin2db(divide(c.power.signal, c.power.ase))
                                 for c in spectral_info.carriers]
            self.raw_osnr_ase_01nm = [ase - ratio for ase, ratio
                                      in zip(self.raw_osnr_ase, ratio_01nm)]
            self.raw_osnr_nli = [lin2db(divide(c.power.signal, c.power.nli))
                                 for c in spectral_info.carriers]
            self.raw_snr = [lin2db(divide(c.power.signal, c.power.nli + c.power.ase))
                            for c in spectral_info.carriers]
            self.raw_snr_01nm = [snr - ratio for snr, ratio
                                 in zip(self.raw_snr, ratio_01nm)]

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
        self.osnr_ase = list(map(lambda x, y: snr_sum(x, y, snr_added),
                                 self.raw_osnr_ase, self.baud_rate))
        self.snr = list(map(lambda x, y: snr_sum(x, y, snr_added),
                            self.raw_snr, self.baud_rate))
        self.osnr_ase_01nm = list(map(lambda x: snr_sum(x, 12.5e9, snr_added),
                                      self.raw_osnr_ase_01nm))
        self.snr_01nm = list(map(lambda x: snr_sum(x, 12.5e9, snr_added),
                                 self.raw_snr_01nm))

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

        snr = round(mean(self.snr), 2)
        osnr_ase = round(mean(self.osnr_ase), 2)
        osnr_ase_01nm = round(mean(self.osnr_ase_01nm), 2)
        snr_01nm = round(mean(self.snr_01nm), 2)
        cd = mean(self.chromatic_dispersion)
        pmd = mean(self.pmd)

        return '\n'.join([f'{type(self).__name__} {self.uid}',

                          f'  GSNR (0.1nm, dB):          {snr_01nm:.2f}',
                          f'  GSNR (signal bw, dB):      {snr:.2f}',
                          f'  OSNR ASE (0.1nm, dB):      {osnr_ase_01nm:.2f}',
                          f'  OSNR ASE (signal bw, dB):  {osnr_ase:.2f}',
                          f'  CD (ps/nm):                {cd:.2f}',
                          f'  PMD (ps):                  {pmd:.2f}'])

    def __call__(self, spectral_info):
        self._calc_snr(spectral_info)
        self._calc_cd(spectral_info)
        self._calc_pmd(spectral_info)
        return spectral_info


RoadmParams = namedtuple('RoadmParams', 'target_pch_out_db add_drop_osnr pmd restrictions per_degree_pch_out_db')


class Roadm(_Node):
    def __init__(self, *args, params, **kwargs):
        if 'per_degree_pch_out_db' not in params.keys():
            params['per_degree_pch_out_db'] = {}
        super().__init__(*args, params=RoadmParams(**params), **kwargs)
        self.loss = 0  # auto-design interest
        self.effective_loss = None
        self.effective_pch_out_db = self.params.target_pch_out_db
        self.passive = True
        self.restrictions = self.params.restrictions
        self.per_degree_pch_out_db = self.params.per_degree_pch_out_db

    @property
    def to_json(self):
        return {'uid': self.uid,
                'type': type(self).__name__,
                'params': {
                    'target_pch_out_db': self.effective_pch_out_db,
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
                          f'  pch out (dBm):        {self.effective_pch_out_db:.2f}'])

    def propagate(self, pref, *carriers, degree):
        # pin_target and loss are read from eqpt_config.json['Roadm']
        # all ingress channels in xpress are set to this power level
        # but add channels are not, so we define an effective loss
        # in the case of add channels
        # find the target power on this degree:
        # if a target power has been defined for this degree use it else use the global one.
        # if the input power is lower than the target one, use the input power instead because
        # a ROADM doesn't amplify, it can only attenuate
        # TODO maybe add a minimum loss for the ROADM
        per_degree_pch = self.per_degree_pch_out_db[degree] if degree in self.per_degree_pch_out_db.keys() else self.params.target_pch_out_db
        self.effective_pch_out_db = min(pref.p_spani, per_degree_pch)
        self.effective_loss = pref.p_spani - self.effective_pch_out_db
        carriers_power = array([c.power.signal + c.power.nli + c.power.ase for c in carriers])
        carriers_att = list(map(lambda x: lin2db(x * 1e3) - per_degree_pch, carriers_power))
        exceeding_att = -min(list(filter(lambda x: x < 0, carriers_att)), default=0)
        carriers_att = list(map(lambda x: db2lin(x + exceeding_att), carriers_att))
        for carrier_att, carrier in zip(carriers_att, carriers):
            pwr = carrier.power
            pwr = pwr._replace(signal=pwr.signal / carrier_att,
                               nli=pwr.nli / carrier_att,
                               ase=pwr.ase / carrier_att)
            pmd = sqrt(carrier.pmd**2 + self.params.pmd**2)
            yield carrier._replace(power=pwr, pmd=pmd)

    def update_pref(self, pref):
        return pref._replace(p_span0=pref.p_span0, p_spani=self.effective_pch_out_db)

    def __call__(self, spectral_info, degree):
        carriers = tuple(self.propagate(spectral_info.pref, *spectral_info.carriers, degree=degree))
        pref = self.update_pref(spectral_info.pref)
        return spectral_info._replace(carriers=carriers, pref=pref)


FusedParams = namedtuple('FusedParams', 'loss')


class Fused(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if params is None:
            # default loss value if not mentioned in loaded network json
            params = {'loss': 1}
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

    def propagate(self, *carriers):
        attenuation = db2lin(self.loss)

        for carrier in carriers:
            pwr = carrier.power
            pwr = pwr._replace(signal=pwr.signal / attenuation,
                               nli=pwr.nli / attenuation,
                               ase=pwr.ase / attenuation)
            yield carrier._replace(power=pwr)

    def update_pref(self, pref):
        return pref._replace(p_span0=pref.p_span0, p_spani=pref.p_spani - self.loss)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref)
        return spectral_info._replace(carriers=carriers, pref=pref)


class Fiber(_Node):
    def __init__(self, *args, params=None, **kwargs):
        if not params:
            params = {}
        super().__init__(*args, params=FiberParams(**params), **kwargs)
        self.pch_out_db = None
        self.nli_solver = NliSolver(self)

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

        return '\n'.join([f'{type(self).__name__}          {self.uid}',
                          f'  type_variety:                {self.type_variety}',
                          f'  length (km):                 {self.params.length * 1e-3:.2f}',
                          f'  pad att_in (dB):             {self.params.att_in:.2f}',
                          f'  total loss (dB):             {self.loss:.2f}',
                          f'  (includes conn loss (dB) in: {self.params.con_in:.2f} out: {self.params.con_out:.2f})',
                          f'  (conn loss out includes EOL margin defined in eqpt_config.json)',
                          f'  pch out (dBm): {self.pch_out_db:.2f}'])

    @property
    def loss(self):
        """total loss including padding att_in: useful for polymorphism with roadm loss"""
        return self.params.loss_coef * self.params.length + self.params.con_in + self.params.con_out + self.params.att_in

    @property
    def passive(self):
        return True

    def alpha(self, frequencies):
        """It returns the values of the series expansion of attenuation coefficient alpha(f) for all f in frequencies

        :param frequencies: frequencies of series expansion [Hz]
        :return: alpha: power attenuation coefficient for f in frequencies [Neper/m]
        """
        if type(self.params.loss_coef) == dict:
            alpha = interp(frequencies, self.params.f_loss_ref, self.params.lin_loss_exp)
        else:
            alpha = self.params.lin_loss_exp * ones(frequencies.shape)

        return alpha

    def alpha0(self, f_ref=193.5e12):
        """It returns the zero element of the series expansion of attenuation coefficient alpha(f) in the
        reference frequency f_ref

        :param f_ref: reference frequency of series expansion [Hz]
        :return: alpha0: power attenuation coefficient in f_ref [Neper/m]
        """
        return self.alpha(f_ref * ones(1))[0]

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

    def _gn_analytic(self, carrier, *carriers):
        r"""Computes the nonlinear interference power on a single carrier.
        The method uses eq. 120 from `arXiv:1209.0394 <https://arxiv.org/abs/1209.0394>`__.

        :param carrier: the signal under analysis
        :param \*carriers: the full WDM comb
        :return: carrier_nli: the amount of nonlinear interference in W on the under analysis
        """

        g_nli = 0
        for interfering_carrier in carriers:
            psi = _psi(carrier, interfering_carrier, beta2=self.params.beta2,
                       asymptotic_length=self.params.asymptotic_length)
            g_nli += (interfering_carrier.power.signal / interfering_carrier.baud_rate)**2 \
                * (carrier.power.signal / carrier.baud_rate) * psi

        g_nli *= (16 / 27) * (self.params.gamma * self.params.effective_length)**2 \
            / (2 * pi * abs(self.params.beta2) * self.params.asymptotic_length)

        carrier_nli = carrier.baud_rate * g_nli
        return carrier_nli

    def propagate(self, *carriers):
        r"""Generator that computes the fiber propagation: attenuation, non-linear interference generation, CD
        accumulation and PMD accumulation.

        :param: \*carriers: the channels at the input of the fiber
        :yield: carrier: the next channel at the output of the fiber
        """

        # apply connector_att_in on all carriers before computing gn analytics  premiere partie pas bonne
        attenuation = db2lin(self.params.con_in + self.params.att_in)

        chan = []
        for carrier in carriers:
            pwr = carrier.power
            pwr = pwr._replace(signal=pwr.signal / attenuation,
                               nli=pwr.nli / attenuation,
                               ase=pwr.ase / attenuation)
            carrier = carrier._replace(power=pwr)
            chan.append(carrier)

        carriers = tuple(f for f in chan)

        # propagate in the fiber and apply attenuation out
        attenuation = db2lin(self.params.con_out)
        for carrier in carriers:
            pwr = carrier.power
            carrier_nli = self._gn_analytic(carrier, *carriers)
            pwr = pwr._replace(signal=pwr.signal / self.params.lin_attenuation / attenuation,
                               nli=(pwr.nli + carrier_nli) / self.params.lin_attenuation / attenuation,
                               ase=pwr.ase / self.params.lin_attenuation / attenuation)
            chromatic_dispersion = carrier.chromatic_dispersion + self.chromatic_dispersion(carrier.frequency)
            pmd = sqrt(carrier.pmd**2 + self.pmd**2)
            yield carrier._replace(power=pwr, chromatic_dispersion=chromatic_dispersion, pmd=pmd)

    def update_pref(self, pref):
        self.pch_out_db = round(pref.p_spani - self.loss, 2)
        return pref._replace(p_span0=pref.p_span0, p_spani=self.pch_out_db)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref)
        return spectral_info._replace(carriers=carriers, pref=pref)


class RamanFiber(Fiber):
    def __init__(self, *args, params=None, **kwargs):
        super().__init__(*args, params=params, **kwargs)
        if self.operational and 'raman_pumps' in self.operational:
            self.raman_pumps = tuple(PumpParams(p['power'], p['frequency'], p['propagation_direction'])
                                     for p in self.operational['raman_pumps'])
        else:
            self.raman_pumps = None
        self.raman_solver = RamanSolver(self)

    @property
    def to_json(self):
        return dict(super().to_json, operational=self.operational)

    def update_pref(self, pref, *carriers):
        pch_out_db = lin2db(mean([carrier.power.signal for carrier in carriers])) + 30
        self.pch_out_db = round(pch_out_db, 2)
        return pref._replace(p_span0=pref.p_span0, p_spani=self.pch_out_db)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(*spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref, *carriers)
        return spectral_info._replace(carriers=carriers, pref=pref)

    def propagate(self, *carriers):
        for propagated_carrier in propagate_raman_fiber(self, *carriers):
            chromatic_dispersion = propagated_carrier.chromatic_dispersion + \
                                   self.chromatic_dispersion(propagated_carrier.frequency)
            pmd = sqrt(propagated_carrier.pmd**2 + self.pmd**2)
            propagated_carrier = propagated_carrier._replace(chromatic_dispersion=chromatic_dispersion, pmd=pmd)
            yield propagated_carrier


class EdfaParams:
    def __init__(self, **params):
        self.update_params(params)
        if params == {}:
            self.type_variety = ''
            self.type_def = ''
            # self.gain_flatmax = 0
            # self.gain_min = 0
            # self.p_max = 0
            # self.nf_model = None
            # self.nf_fit_coeff = None
            # self.nf_ripple = None
            # self.dgt = None
            # self.gain_ripple = None
            # self.out_voa_auto = False
            # self.allowed_for_design = None

    def update_params(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, self.update_params(**v) if isinstance(v, dict) else v)


class EdfaOperational:
    default_values = {
        'gain_target': None,
        'delta_p': None,
        'out_voa': None,
        'tilt_target': 0
    }

    def __init__(self, **operational):
        self.update_attr(operational)

    def update_attr(self, kwargs):
        clean_kwargs = {k: v for k, v in kwargs.items() if v != ''}
        for k, v in self.default_values.items():
            setattr(self, k, clean_kwargs.get(k, v))

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'gain_target={self.gain_target!r}, '
                f'tilt_target={self.tilt_target!r})')


class Edfa(_Node):
    def __init__(self, *args, params=None, operational=None, **kwargs):
        if params is None:
            params = {}
        if operational is None:
            operational = {}
        self.variety_list = kwargs.pop('variety_list', None)
        super().__init__(
            *args,
            params=EdfaParams(**params),
            operational=EdfaOperational(**operational),
            **kwargs
        )
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
        return '\n'.join([f'{type(self).__name__} {self.uid}',
                          f'  type_variety:           {self.params.type_variety}',
                          f'  effective gain(dB):     {self.effective_gain:.2f}',
                          f'  (before att_in and before output VOA)',
                          f'  noise figure (dB):      {nf:.2f}',
                          f'  (including att_in)',
                          f'  pad att_in (dB):        {self.att_in:.2f}',
                          f'  Power In (dBm):         {self.pin_db:.2f}',
                          f'  Power Out (dBm):        {self.pout_db:.2f}',
                          f'  Delta_P (dB):           ' + f'{self.delta_p:.2f}' if self.delta_p is not None else 'None',
                          f'  target pch (dBm):       ' + f'{self.target_pch_out_db:.2f}' if self.target_pch_out_db is not None else 'None',
                          f'  effective pch (dBm):    {self.effective_pch_out_db:.2f}',
                          f'  output VOA (dB):        {self.out_voa:.2f}'])

    def interpol_params(self, frequencies, pin, baud_rates, pref):
        """interpolate SI channel frequencies with the edfa dgt and gain_ripple frquencies from JSON
        """
        # TODO|jla: read amplifier actual frequencies from additional params in json
        self.channel_freq = frequencies

        amplifier_freq = arrange_frequencies(len(self.params.dgt), self.params.f_min, self.params.f_max)  # Hz
        self.interpol_dgt = interp(self.channel_freq, amplifier_freq, self.params.dgt)

        amplifier_freq = arrange_frequencies(len(self.params.gain_ripple), self.params.f_min, self.params.f_max)  # Hz
        self.interpol_gain_ripple = interp(self.channel_freq, amplifier_freq, self.params.gain_ripple)

        amplifier_freq = arrange_frequencies(len(self.params.nf_ripple), self.params.f_min, self.params.f_max)  # Hz
        self.interpol_nf_ripple = interp(self.channel_freq, amplifier_freq, self.params.nf_ripple)

        self.nch = frequencies.size
        self.pin_db = lin2db(sum(pin * 1e3))

        """in power mode: delta_p is defined and can be used to calculate the power target
        This power target is used calculate the amplifier gain"""
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

        pout = (pin + self.noise_profile(baud_rates)) * db2lin(self.gprofile)
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

    def noise_profile(self, df):
        """noise_profile(bw) computes amplifier ASE (W) in signal bandwidth (Hz)

        Noise is calculated at amplifier input

        :bw: signal bandwidth = baud rate in Hz
        :type bw: float

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

        ase = h * df * self.channel_freq * db2lin(self.nf)  # W
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

    def propagate(self, pref, *carriers):
        """add ASE noise to the propagating carriers of :class:`.info.SpectralInformation`"""
        pin = array([c.power.signal + c.power.nli + c.power.ase for c in carriers])  # pin in W
        freq = array([c.frequency for c in carriers])
        brate = array([c.baud_rate for c in carriers])
        # interpolate the amplifier vectors with the carriers freq, calculate nf & gain profile
        self.interpol_params(freq, pin, brate, pref)

        gains = db2lin(self.gprofile)
        carrier_ases = self.noise_profile(brate)
        att = db2lin(self.out_voa)

        for gain, carrier_ase, carrier in zip(gains, carrier_ases, carriers):
            pwr = carrier.power
            pwr = pwr._replace(signal=pwr.signal * gain / att,
                               nli=pwr.nli * gain / att,
                               ase=(pwr.ase + carrier_ase) * gain / att)
            yield carrier._replace(power=pwr)

    def update_pref(self, pref):
        return pref._replace(p_span0=pref.p_span0,
                             p_spani=pref.p_spani + self.effective_gain - self.out_voa)

    def __call__(self, spectral_info):
        carriers = tuple(self.propagate(spectral_info.pref, *spectral_info.carriers))
        pref = self.update_pref(spectral_info.pref)
        return spectral_info._replace(carriers=carriers, pref=pref)
