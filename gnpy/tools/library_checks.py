# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.library_checks: pure checks of equipment libraries (no logging, no I/O)
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""Checks on equipment libraries. Functions return issues and never log nor modify input data."""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from statistics import median
from typing import Dict, List, Iterable, Tuple, Callable

import oopt_gnpy_libyang as ly
from gnpy.tools.yang_convert_utils import dump_data
from gnpy.tools.convert_legacy_yang import legacy_to_yang, yang_to_legacy
from gnpy.core.utils import lin2db, db2lin, automatic_nch
from gnpy.core.science_utils import estimate_nf_model
from gnpy.core.exceptions import EquipmentConfigError

LEGACY_TYPE = Dict[str, List | int | float | str | Dict | None]
YANG_TYPE = Dict[str, List | int | str | Dict | None]


LIBRARY = 'Library'
TRANSCEIVER = 'Transceiver'
AMPLIFIER = 'Edfa'
ROADM = 'Roadm'

ZERO_SCORE_ISSUE_CODES: set[str] = {
    'BIT_RATE_INVALID',
    'INVALID_PENALTY_VALUE_NOT_MONOTONIC',
    'INVALID_NF_RANGE_NOT_MATCHING_GAIN_RANGE',
    'INVALID_MULTIBAND_SINGLE_BAND'
}


class Axis(Enum):
    """Validation axes used to assess an equipment library."""
    FORMAT = 'format'
    CONSISTENCY = 'consistency'
    COMPLETENESS = 'completeness'


WEIGHTS = {Axis.FORMAT: 0.4, Axis.CONSISTENCY: 0.4, Axis.COMPLETENESS: 0.2}


@dataclass(frozen=True)
class Issue:
    """Describe a validation issue found in a library.

    :param axis: Validation axis associated with the issue.
    :param code: Machine-readable issue code.
    :param message: Human-readable description of the issue.
    :param section: Library section where the issue was found.
    :param item: Item concerned by the issue, such as a type variety.
    :param sub_item: Sub-item concerned by the issue, such as a mode.
    :param weight: Penalty applied to the corresponding score.
    """
    axis: Axis
    code: str
    message: str
    section: str                   # 'Transceiver', 'Edfa', 'Roadm', ...
    item: str | None = None        # type_variety
    sub_item: str | None = None    # mode, ...
    weight: float = 1.0


@dataclass
class Report:
    """Collect validation issues and the number of performed checks.

    :ivar issues: Issues found during validation.
    :ivar checked: Number of checks grouped by section and validation axis.
    :ivar legacy_data: Library converted to the legacy representation, if valid.
    """
    issues: list = field(default_factory=list)
    checked: dict = field(default_factory=lambda: defaultdict(int))  # (section, axis) -> n
    legacy_data: dict | None = None

    def extend(self, section: str, axis: Axis, n_checks: int, issues: Iterable[Issue]):
        """Add validation results to the report.

        :param section: Library section associated with the checks.
        :param axis: Validation axis associated with the checks.
        :param n_checks: Number of checks performed.
        :param issues: Issues produced by the checks.
        :return: None.
        """
        self.checked[(section, axis)] += n_checks
        self.issues.extend(issues)

    @property
    def sections(self) -> List[str]:
        """Return the sections containing at least one check.

        :return: Sorted list of section names.
        """
        return sorted({s[0] for s in self.checked})


@dataclass(frozen=True)
class ModeEntry:
    """Store the values required for equalization-offset checks.

    :param transceiver: Transceiver type variety.
    :param mode: Mode identifier.
    :param baud_rate: Mode baud rate in symbols per second.
    :param min_spacing: Minimum channel spacing in hertz.
    :param offset: Equalization offset in decibels.
    :param bit_rate: Bit rate in Hz.
    :param osnr: Minimum required OSNR ind dB@0.1nm.
    :param penalties_pmd: maximum PMD impairment supporté par le mode, en ps.
    :param penalties_pdl: maximum PDL impairment supporté par le mode, en dB.
    :param tx_osnr: OSNR intrudced by the emitter in db@0.1nm
    :param tx_power: maximum per channel power of the emitter
    """
    transceiver: str
    mode: str
    baud_rate: float
    min_spacing: float
    offset: float | None
    offset_defined: bool = True
    bit_rate: float | None = None
    osnr: float | None = None
    penalties_pmd: float | None = None
    penalties_pdl: float | None = None
    tx_osnr: float | None = None
    tx_power: float | None = None
    f_min: float | None = None
    f_max: float | None = None


@dataclass(frozen=True)
class RoadmEntry:
    """Store the values required for roadm checks.

    :param roadm: Roadm type variety.
    :param add_drop_osnr: add_drop_osnr in dB.
    :param pmd: polarization mode dispersion introduced by ROADM in s.
    :param pdl: polarization dependant loss of ROADM in dB.
    :param target_pch_out_db: constant power equalization target
    :param target_psd_out_mWperGHz: constant power spectral density with respect to baud rate target
    :param target_out_mWperSlotWidth: constant power spectral density with respect to slot width target
    :param detailed_add_pmd: minimum PMD impairment for add path
    :param detailed_drop_pmd: minimum PMD impairment for drop path
    :param detailed_add_pdl: minimum PDL impairment for add path
    :param detailed_drop_pdl: minimum PDL impairment for drop path
    :param detailed_add_osnr: minimum OSNR impairment for add path
    :param detailed_drop_osnr: minimum OSNR impairment for drop path
    :param detailed_add_maxloss: minimum loss for add path
    :param detailed_drop_maxloss: minimum loss for drop path
    """
    roadm: str
    add_drop_osnr: str | None = None
    pmd: float | None = None
    pdl: float | None = None
    target_pch_out_db: float | None = None
    target_psd_out_mWperGHz: float | None = None
    target_out_mWperSlotWidth: float | None = None
    detailed_add_pmd: float = 0
    detailed_drop_pmd: float = 0
    detailed_add_pdl: float = 0
    detailed_drop_pdl: float = 0
    detailed_add_osnr: float | None = None
    detailed_drop_osnr: float | None = None
    detailed_add_maxloss: float = 0
    detailed_drop_maxloss: float = 0


@dataclass(frozen=True)
class EdfaEntry:
    """Store the values required for amplfier checks.
    :param f_min:
    :param f_max:
    """
    amplifier: str
    type_def: str
    pmax: float
    f_min: float | None = None
    f_max: float | None = None
    amplifiers: list = field(default_factory=list)
    alias: list = field(default_factory=list)
    raman: bool = False


# ----------------------------------------------------------------------
# Format (YANG)
# ----------------------------------------------------------------------
def _yang_error_to_issue(error: ly.Error) -> Issue:
    """Convert one libyang error into an issue.

    :param error: ErrorMessage with `what` and `where` attributes.
    :return: Format issue with the location in item and the cause in message.
    """
    path = error.where or ''
    parts = path.replace('/gnpy-eqpt-config:equipment/', '', 1).split('/', 1)
    item = parts[0] if parts and parts[0] else None
    sub_item = parts[1] if len(parts) > 1 else None
    return Issue(Axis.FORMAT, 'YANG_INVALID', error.what.strip(), section=LIBRARY,
                 item=item, sub_item=sub_item, weight=1.0)


def check_yang(data: YANG_TYPE | LEGACY_TYPE) -> tuple[LEGACY_TYPE | None, list[Issue]]:
    """Validate library data against the YANG model.

    The function first checks RFC7951/YANG-formatted data. If that fails,
    it attempts to convert legacy-formatted data to the YANG representation.

    :param data: Library data to validate.
    :return: A tuple containing the converted legacy data, if available,
        and the list of format issues.
    """
    try:
        return yang_to_legacy(json.loads(dump_data(data))), []
    except ly.Error:
        pass
    try:
        legacy = yang_to_legacy(legacy_to_yang(data))
        return legacy, [Issue(Axis.FORMAT, 'NOT_RFC7951',
                              'Data does not conform to RFC7951 but is compliant with the YANG model. '
                              'Please use YANG formatted data compliant with RFC7951.', section=LIBRARY,
                              weight=0.3)]
    except ly.Error as err:
        messages = err.args[1] if len(err.args) > 1 else []
        if not messages:
            return None, [Issue(Axis.FORMAT, 'YANG_INVALID',
                                f'Library does not comply to YANG model: {err}', section=LIBRARY)]
        return None, [_yang_error_to_issue(m) for m in messages]


# ----------------------------------------------------------------------
# Mode checks
# ----------------------------------------------------------------------
def _is_valid_bit_rate(value: float) -> bool:
    """Return whether a bit rate is a valid multiple of 50 Gbit/s.

    :param value: Bit rate in bit/s.
    :return: ``True`` if the value is a valid bit rate, otherwise ``False``.
    """
    ratio = value / 50e9
    return ratio >= 1 and math.isclose(ratio, round(ratio), abs_tol=1e-9)


def check_mode(mode: LEGACY_TYPE, trx_name: str) -> Tuple[int, List[Issue], int, List[Issue]]:
    """Check consistency and completeness of one transceiver mode.

    :param mode: Transceiver mode data.
    :param trx_name: Transceiver type variety.
    :return: A tuple containing, in order, the number of consistency checks,
        consistency issues, completeness checks, and completeness issues.
    """
    name = mode['format']
    cons, comp = [], []
    n_cons = n_comp = 0

    # bit_rate
    n_cons += 1
    try:
        if not _is_valid_bit_rate(float(mode['bit_rate'])):
            cons.append(Issue(Axis.CONSISTENCY, 'BIT_RATE_INVALID',
                              f'bit_rate {mode["bit_rate"]} is not a multiple of 50e9 bit/s',
                              TRANSCEIVER,
                              trx_name, name, weight=0.5))
    except (KeyError, TypeError, ValueError):
        cons.append(Issue(Axis.CONSISTENCY, 'BIT_RATE_INVALID', 'bit_rate missing or not numeric',
                          TRANSCEIVER, trx_name, name, weight=0.5))

    # penalties
    n_comp += 1
    penalties = mode.get('penalties')
    if not penalties:
        comp.append(Issue(Axis.COMPLETENESS, 'PENALTIES_MISSING', 'Missing penalties', TRANSCEIVER, trx_name, name))
    else:
        for key in ('chromatic_dispersion', 'pmd', 'pdl'):
            n_comp += 1
            if not any(key in elem for elem in penalties):
                comp.append(Issue(Axis.COMPLETENESS, 'PENALTY_KIND_MISSING',
                                  f'Missing {key} impairment in penalties', TRANSCEIVER, trx_name, name, weight=0.5))
            else:
                n_cons += 1
                cons.extend(
                    check_penalty_values_monotonicity(
                        penalty=key,
                        penalties=penalties,
                        transceiver=trx_name,
                        mode=name,
                    )
                )
    return n_cons, cons, n_comp, comp


def check_edfa(amplifier: LEGACY_TYPE) -> Tuple[int, List[Issue], int, List[Issue]]:
    """Check consistency and completeness of one amplifier.

    :param amplifier: item of the library Edfa list
    :return: A tuple containing, in order, the number of consistency checks,
        consistency issues, completeness checks, and completeness issues.
    """
    name = amplifier['type_variety']
    cons, comp = [], []
    n_cons = n_comp = 0

    # frequency range should be several THz, non blocking but may reveal typos
    f_min, f_max = amplifier.get('f_min'), amplifier.get('f_max')
    if f_min and f_max:
        n_cons += 1
        if f_max - f_min < 1e12:
            cons.append(Issue(Axis.CONSISTENCY, 'FREQUENCY_RANGE_TOO_SMALL',
                              message=f'Frequency range is too small {(f_max - f_min) * 1e-12:.2f}THz',
                              section=AMPLIFIER,
                              item=name,
                              weight=0.5))

    # nf and gain should satisfy GNPy check
    if amplifier.get('type_def') == 'variable_gain':
        n_cons += 1
        try:
            estimate_nf_model(name, amplifier.get('gain_min'), amplifier.get('gain_flatmax'),
                              amplifier.get('nf_min'), amplifier.get('nf_max'))
        except EquipmentConfigError as e:
            cons.append(Issue(Axis.CONSISTENCY, 'INVALID_NF_RANGE_NOT_MATCHING_GAIN_RANGE',
                              message=f'{e}',
                              section=AMPLIFIER,
                              item=name,
                              weight=0.5))

    # frequency range should not be missing although this is tolerated by GNPy
    n_comp += 1
    if amplifier.get('type_def') in ["variable_gain", "fixed_gain"] and (
            not amplifier.get('f_min') or not amplifier.get('f_max')):
        comp.append(Issue(Axis.COMPLETENESS, 'FREQUENCY_RANGE_MISSING',
                          message='Missing frequency range',
                          section=AMPLIFIER,
                          item=name,
                          weight=1))

    # Raman amplifier (fixed gain with <3 nf0) must have allowed_for_design False
    # because autodesign does not know yet how to cascade raman amp with EDFA, and
    # pure Raman amplification can not me modeled with fixed_gain model.
    n_cons += 1
    if amplifier.get('type_def') == 'fixed_gain' and amplifier.get('raman') and amplifier.get('allowed_for_design'):
        cons.append(Issue(Axis.CONSISTENCY, 'INVALID_RAMAN_AMPLIFIER_ALLOWED_FOR_DESIGN',
                          message='Raman amplifier (fixed gain with <3 nf0) can not be used alone by autodesign',
                          section=AMPLIFIER,
                          item=name,
                          weight=0.5))

    return n_cons, cons, n_comp, comp


def _build_clusters(
    entries: List[ModeEntry],
    group_by: Callable[[ModeEntry], float],
    is_compatible: Callable[[ModeEntry, ModeEntry], bool],
) -> list[list[ModeEntry]]:
    """Group mode entries into clusters.

    Entries are first grouped using ``group_by`` and then sorted by baud rate.
    An entry is added to the current cluster when ``is_compatible`` returns
    ``True`` compared with the first entry of that cluster.

    :param entries: Mode entries to group.
    :param group_by: Function returning the grouping key for an entry.
    :param is_compatible: Function checking whether two entries belong to
        the same cluster.
    :return: List of mode-entry clusters.
    """
    grouped = defaultdict(list)

    for entry in entries:
        grouped[group_by(entry)].append(entry)

    clusters: list[list[ModeEntry]] = []

    for group in grouped.values():
        group.sort(key=lambda entry: entry.baud_rate)

        current: list[ModeEntry] = []

        for entry in group:
            if current and is_compatible(current[0], entry):
                current.append(entry)
            else:
                if current:
                    clusters.append(current)
                current = [entry]

        if current:
            clusters.append(current)

    return clusters


# ----------------------------------------------------------------------
# Offset consistency
# ----------------------------------------------------------------------
def _build_offset_clusters(
    entries: List[ModeEntry],
    baud_tol: float,
) -> list[list[ModeEntry]]:
    """Build clusters for equalization-offset checks.

    :param entries: Mode entries to group.
    :param baud_tol: Maximum baud-rate difference within a cluster.
    :return: List of mode-entry clusters.
    """
    return _build_clusters(
        entries,
        group_by=lambda entry: entry.min_spacing,
        is_compatible=lambda first, entry: (
            abs(entry.baud_rate - first.baud_rate) <= baud_tol
        ),
    )


def check_equalization_offset_consistency(
        entries: List[ModeEntry], reference: ModeEntry | None = None, baud_tol: float = 5e9, offset_tol: float = 0.2,
        proportionality_tol: float = 0.75) -> tuple[int, list[Issue]]:
    """Check equalization offsets across transceiver modes.

    Two rules are checked:

    * offsets must be consistent within a cluster of modes sharing the same
      minimum spacing and having close baud rates;
    * cluster offsets must be proportional to the reference channel, using
      either the spacing ratio or the baud-rate ratio.

    If all modes are without an offset this check is skipped.

    :param entries: Mode entries to validate.
    :param reference: Reference SI channel, if available.
    :param baud_tol: Maximum baud-rate difference within a cluster.
    :param offset_tol: Maximum allowed intra-cluster offset difference.
    :param proportionality_tol: Maximum allowed difference from the expected
        reference-based offset.
    :return: A tuple containing the number of performed checks and the issues.
    """
    # if all offsets are 0, then skip this check
    n_checks, issues = 0, []
    if not [e for e in entries if e.offset != 0 and e.mode != 'reference_mode']:
        return n_checks, issues

    # if any offset is defined, then all offsets should be consistent
    clusters = _build_offset_clusters(entries, baud_tol)
    # ------------------------------------------------------------------
    # 1. Offsets must be consistents across modes
    # ------------------------------------------------------------------
    for cluster in clusters:
        n_checks += len(cluster)
        cluster_offset = median(e.offset for e in cluster)
        for entry in cluster:
            if abs(entry.offset - cluster_offset) > offset_tol:
                others = ', '.join(f'{e.transceiver}/{e.mode}={e.offset:.2f} dB' for e in cluster if e is not entry)
                issues.append(Issue(
                    Axis.CONSISTENCY, 'OFFSET_INCONSISTENT',
                    f'Spacing {entry.min_spacing * 1e-9:.2f} GHz, '
                    f'baud_rate={entry.baud_rate * 1e-9:.2f} Gbaud: offset={entry.offset:.2f} dB '
                    f'differs from cluster median {cluster_offset:.2f} dB. Cluster: {others}',
                    TRANSCEIVER, entry.transceiver, entry.mode))

    # without reference, we can not check the proportionality with respect to it.
    if reference is None or reference.offset is None:
        return n_checks, issues

    for cluster in clusters:
        baud = median(e.baud_rate for e in cluster)
        spacing = median(e.min_spacing for e in cluster)
        exp_spacing = reference.offset + 10 * math.log10(spacing / reference.min_spacing)
        exp_baud = reference.offset + 10 * math.log10(baud / reference.baud_rate)
        for entry in cluster:
            n_checks += 1
            if (abs(entry.offset - exp_spacing) > proportionality_tol
                    and abs(entry.offset - exp_baud) > proportionality_tol):
                issues.append(Issue(
                    Axis.CONSISTENCY, 'OFFSET_NOT_PROPORTIONAL',
                    f'Offset {entry.offset:.2f} dB not consistent with reference '
                    f'(baud_rate={entry.baud_rate * 1e-9:.2f} Gbaud, '
                    f'spacing={entry.min_spacing * 1e-9:.2f} GHz). '
                    f'Expected approximately {exp_spacing:.2f} dB from spacing '
                    f'or {exp_baud:.2f} dB from baud rate.',
                    TRANSCEIVER, entry.transceiver, entry.mode))
    return n_checks, issues


# ----------------------------------------------------------------------
# OSNR consistency
# ----------------------------------------------------------------------
def _build_osnr_clusters(
    entries: List[ModeEntry],
    baud_tol: float,
    bit_rate_tol: float,
) -> list[list[ModeEntry]]:
    """Build clusters for OSNR checks.

    :param entries: Mode entries to group.
    :param baud_tol: Maximum baud-rate difference within a cluster.
    :param bit_rate_tol: Maximum bit-rate difference within a cluster.
    :return: List of mode-entry clusters.
    """
    return _build_clusters(
        entries,
        group_by=lambda entry: entry.bit_rate,
        is_compatible=lambda first, entry: (
            abs(entry.baud_rate - first.baud_rate) <= baud_tol
            and abs(entry.bit_rate - first.bit_rate) <= bit_rate_tol
        ),
    )


def check_osnr_consistency(
        entries: List[ModeEntry], baud_tol: float = 15e9, bit_rate_tol: float = 1e9, osnr_tol: float = 5.0,
) -> tuple[int, list[Issue]]:
    """Looks for anomalies in OSNR

    :param entries: Mode entries to validate.
    :param baud_tol: Maximum baud-rate difference within a cluster.
    :param bit_tol: Maximum bit-rate difference within a cluster.
    :param osnr_tol: Maximum OSNR difference within a cluster.
    :return: A tuple containing the number of performed checks and the issues.
    """
    # if all offsets are 0, then skip this check
    n_checks, issues = 0, []

    clusters = _build_osnr_clusters(entries, baud_tol=baud_tol, bit_rate_tol=bit_rate_tol)
    # ------------------------------------------------------------------
    # 1. OSNR must be consistents across modes
    # ------------------------------------------------------------------
    for cluster in clusters:
        n_checks += len(cluster)
        cluster_osnr = median(e.osnr for e in cluster)
        for entry in cluster:
            if abs(entry.osnr - cluster_osnr) > osnr_tol:
                others = ', '.join(f'{e.transceiver}/{e.mode}={e.osnr:.2f} dB' for e in cluster if e is not entry)
                issues.append(Issue(
                    Axis.CONSISTENCY, 'OSNR_INCONSISTENT',
                    f'bit_rate {entry.bit_rate * 1e-9:.2f} Gbit/s, '
                    f'baud_rate={entry.baud_rate * 1e-9:.2f} Gbaud: OSNR={entry.osnr:.2f} dB '
                    f'differs from cluster median {cluster_osnr:.2f} dB. Cluster: {others}',
                    TRANSCEIVER, entry.transceiver, entry.mode))
    return n_checks, issues


def check_penalty_values_monotonicity(
    penalty: str,
    penalties: List[Dict[str, float]],
    transceiver: str,
    mode: str,
) -> list[Issue]:
    """Check that penalty values increase with impairment magnitude.

    PMD and PDL are positive values and are checked in ascending order.
    Chromatic dispersion can be negative. In that case, the penalty must
    increase with the absolute value of the dispersion.

    :param penalties: Penalties grouped by penalty type.
    :param transceiver: Transceiver type variety.
    :param mode: Transceiver mode.
    :return: List of monotonicity issues.
    """
    issues: list[Issue] = []

    valid_entries = [entry for entry in penalties if entry.get(penalty) is not None]

    if penalty == 'chromatic_dispersion':
        # The penalty depends on the magnitude of the dispersion.
        ordered_entries = sorted(valid_entries, key=lambda entry: abs(entry[penalty]))
    else:
        # PMD and PDL are positive values.
        ordered_entries = sorted(valid_entries, key=lambda entry: entry[penalty])

    for previous, current in zip(
        ordered_entries,
        ordered_entries[1:],
        strict=False,
    ):
        if current['penalty_value'] < previous['penalty_value']:
            issues.append(
                Issue(
                    axis=Axis.CONSISTENCY,
                    code='INVALID_PENALTY_VALUE_NOT_MONOTONIC',
                    message=(
                        f'{penalty} penalty values must not decrease '
                        f'when {penalty} increases.'
                    ),
                    section=TRANSCEIVER,
                    item=transceiver,
                    sub_item=mode,
                )
            )
            break

    return issues


# ----------------------------------------------------------------------
# ROADM consistency with Transceivers
# ----------------------------------------------------------------------
def check_roadm_pmd_pdl_consistency_with_transceivers(
    roadm_entries: List[RoadmEntry],
    transceiver_entries: List[ModeEntry],
) -> tuple[int, list[Issue]]:
    """Check that ROADMs are compatible with the transceiver with the lowest penalty tolerance.

    Each ROADM is assumed to be crossed twice. The accumulated PMD and PDL
    are computed using the root-sum-square formula.

    :param roadm_entries: ROADM entries to validate.
    :param transceiver_entries: Transceiver mode entries.
    :return: Number of checks performed and detected issues.
    """
    n_checks = 0
    issues: list[Issue] = []

    transceiver_pmd_tolerances = \
        [entry.penalties_pmd for entry in transceiver_entries if entry.penalties_pmd is not None]

    transceiver_pdl_tolerances = \
        [entry.penalties_pdl for entry in transceiver_entries if entry.penalties_pdl is not None]

    # The worst transceiver is the one with the smallest tolerance.
    min_transceiver_pmd = (min(transceiver_pmd_tolerances) if transceiver_pmd_tolerances else None)
    min_transceiver_pdl = (min(transceiver_pdl_tolerances) if transceiver_pdl_tolerances else None)

    for roadm in roadm_entries:
        if roadm.pmd is not None and min_transceiver_pmd is not None:
            n_checks += 1
            accumulated_pmd = roadm.pmd * math.sqrt(2) \
                if not roadm.detailed_add_pmd and not roadm.detailed_drop_pmd \
                else math.sqrt(roadm.detailed_add_pmd**2 + roadm.detailed_drop_pmd**2)
            if accumulated_pmd > min_transceiver_pmd:
                if not roadm.detailed_add_pmd and not roadm.detailed_drop_pmd:
                    msg = f'ROADM PMD={roadm.pmd:.2f}ps is too high for two ' \
                        + f'ROADMs: accumulated PMD={accumulated_pmd:.2f}ps, ' \
                        + 'while the worst transceiver tolerance is ' \
                        + f'{min_transceiver_pmd:.2f}ps.'
                else:
                    msg = 'ROADM add/drop path PMD ' \
                        + f'({roadm.detailed_add_pmd:.2f}ps/{roadm.detailed_drop_pmd:.2f}ps) ' \
                        + f'are too high for two ROADMs: accumulated PMD={accumulated_pmd:.2f}ps, ' \
                        + f'while the worst transceiver tolerance is {min_transceiver_pmd:.2f}ps.'
                issues.append(
                    Issue(
                        axis=Axis.CONSISTENCY,
                        code='ROADM_PMD_TOO_HIGH',
                        message=msg,
                        section=ROADM,
                        item=roadm.roadm,
                    )
                )

        if roadm.pdl is not None and min_transceiver_pdl is not None:
            n_checks += 1
            accumulated_pdl = roadm.pdl * math.sqrt(2) \
                if not roadm.detailed_add_pdl and not roadm.detailed_drop_pdl \
                else math.sqrt(roadm.detailed_add_pdl**2 + roadm.detailed_drop_pdl**2)
            if accumulated_pdl > min_transceiver_pdl:
                if not roadm.detailed_add_pdl and not roadm.detailed_drop_pdl:
                    msg_pdl = f'ROADM PDL={roadm.pdl:.2f}dB is too high for two ' \
                        + f'ROADMs: accumulated PDL={accumulated_pdl:.2f}dB, ' \
                        + 'while the worst transceiver tolerance is ' \
                        + f'{min_transceiver_pdl:.2f}dB.'
                else:
                    msg_pdl = 'ROADM add/drop path PDL ' \
                        + f'({roadm.detailed_add_pdl:.2f}dB/{roadm.detailed_drop_pdl:.2f}dB) ' \
                        + f'are too high for two ROADMs: accumulated PMD={accumulated_pdl:.2f}dB, ' \
                        + f'while the worst transceiver tolerance is {min_transceiver_pdl:.2f}dB.'
                issues.append(
                    Issue(
                        axis=Axis.CONSISTENCY,
                        code='ROADM_PDL_TOO_HIGH',
                        message=msg_pdl,
                        section=ROADM,
                        item=roadm.roadm,
                    )
                )

    return n_checks, issues


def check_roadm_add_drop_osnr_consistency_with_transceivers(
    roadm_entries: List[RoadmEntry],
    transceiver_entries: List[ModeEntry],
) -> tuple[int, list[Issue]]:
    """Check that ROADMs are compatible with the transceiver with the best sensitivity.

    Each ROADM is assumed to be crossed twice. The accumulated OSNR
    is computed using the tx_osnr and the add_drop_osnr or, the one defined in
    the roadm_impairment_path

    :param roadm_entries: ROADM entries to validate.
    :param transceiver_entries: Transceiver mode entries.
    :return: Number of checks performed and detected issues.
    """
    n_checks = 0
    issues: list[Issue] = []
    best_osnr_sensitivity, tx_osnr = min([(m.osnr, m.tx_osnr) for m in transceiver_entries if m.osnr],
                                         key=lambda m: m[0])

    for roadm in roadm_entries:
        n_checks += 1
        # Minimum accumulated OSNR just crossing 2 ROADMs and including the Transceiver tx_osnr
        accumulated_osnr = lin2db(1 / (1 / db2lin(roadm.add_drop_osnr) + 1 / db2lin(tx_osnr))) \
            if not roadm.detailed_add_osnr and not roadm.detailed_drop_osnr \
            else lin2db(1 / (1 / db2lin(roadm.detailed_add_osnr)
                             + 1 / db2lin(roadm.detailed_drop_osnr)
                             + 1 / db2lin(tx_osnr)))
        if accumulated_osnr < best_osnr_sensitivity:
            if not roadm.detailed_add_osnr and not roadm.detailed_drop_osnr:
                msg = f'ROADM add_drop_osnr={roadm.add_drop_osnr:.2f}dB is too low for two ' \
                    + f'ROADMs: accumulated OSNR={accumulated_osnr:.2f}dB, ' \
                    + 'while the best transceiver sensitivity is ' \
                    + f'{best_osnr_sensitivity:.2f}pdB.'
            else:
                msg = 'ROADM add/drop path OSNR ' \
                    + f'({roadm.detailed_add_osnr:.2f}dB/{roadm.detailed_drop_osnr:.2f}dB) ' \
                    + f'are too low for two ROADMs: accumulated OSNR={accumulated_osnr:.2f}pdB, ' \
                    + f'while the best transceiver sensitivity is {best_osnr_sensitivity:.2f}dB.'
            issues.append(
                Issue(
                    axis=Axis.CONSISTENCY,
                    code='ROADM_ADD_DROP_OSNR_TOO_LOW',
                    message=msg,
                    section=ROADM,
                    item=roadm.roadm,
                )
            )

    return n_checks, issues


def check_roadm_target_power_with_transceivers(
    roadm_entries: List[RoadmEntry],
    transceiver_entries: List[ModeEntry],
    default_loss: float = 5
) -> tuple[int, list[Issue]]:
    """Check that ROADMs target power is reasonably lower than transceivers' max output.

    :param roadm_entries: ROADM entries to validate.
    :param transceiver_entries: Transceiver mode entries.
    :return: Number of checks performed and detected issues.
    """
    n_checks = 0
    issues: list[Issue] = []
    reference_baud_rate, reference_slot_width = \
        next(((entry.baud_rate * 1e-9, entry.min_spacing * 1e-9) for entry in transceiver_entries
             if entry.transceiver == 'reference_channel'), (None, None))
    if not reference_baud_rate:
        # can not check
        return n_checks, issues
    # if SI is part of the library, the following list is never empty.
    maximum_output_power = max([(m.tx_power) for m in transceiver_entries if m.tx_power is not None])
    for roadm in roadm_entries:
        roadm_maxloss = roadm.detailed_add_maxloss if roadm.detailed_add_maxloss else default_loss
        target_power = None
        if roadm.target_pch_out_db is not None:
            target_power = roadm.target_pch_out_db
        elif roadm.target_psd_out_mWperGHz is not None:
            target_power = lin2db(roadm.target_psd_out_mWperGHz * reference_baud_rate)
        elif roadm.target_out_mWperSlotWidth is not None:
            target_power = lin2db(roadm.target_out_mWperSlotWidth * reference_slot_width)
        if target_power is None:
            continue
        n_checks += 1
        if maximum_output_power - roadm_maxloss < target_power:
            if roadm.target_pch_out_db is not None:
                msg = f'ROADM target power={roadm.target_pch_out_db:.2f}dBm is too high ' \
                    + f'compared to maximum output power of transceiver {maximum_output_power:.2f}dBm ' \
                    + f'and estimated ROADM best maxloss {roadm_maxloss:.2f}dB.'
            elif roadm.target_psd_out_mWperGHz is not None:
                msg = f'ROADM target power={roadm.target_psd_out_mWperGHz:.2f}mW/GHz is too high ' \
                    + 'compared to maximum output power spectral density ' \
                    + f'of transceiver {db2lin(maximum_output_power) / reference_baud_rate:.2f}mW/GHz ' \
                    + f'and estimated ROADM best maxloss {roadm_maxloss:.2f}dB.'
            elif roadm.target_out_mWperSlotWidth is not None:
                msg = f'ROADM target power={roadm.target_out_mWperSlotWidth:.2f}mW/GHz is too high ' \
                    + 'compared to maximum output power spectral density ' \
                    + f'of transceiver {db2lin(maximum_output_power) / reference_slot_width:.2f}mW/GHz ' \
                    + f'and estimated ROADM best maxloss {roadm_maxloss:.2f}dB.'
            issues.append(
                Issue(
                    axis=Axis.CONSISTENCY,
                    code='ROADM_TARGET_POWER_TOO_HIGH',
                    message=msg,
                    section=ROADM,
                    item=roadm.roadm,
                )
            )

    return n_checks, issues


def check_consistency_of_roadm_path_impairment(roadm: LEGACY_TYPE,
                                               tolerance: float = 2) -> Tuple[int, List[Issue], int, List[Issue]]:
    """Check that the parameters in the bands are consistent and check completeness

    :param mode: ROADM data.
    :return: A tuple containing, in order, the number of consistency checks,
            consistency issues, completeness checks, and completeness issues.
    """
    name = roadm.get('type_variety', 'default')
    default = {
        'roadm-pmd': roadm.get('pmd'),
        'roadm-pdl': roadm.get('pdl'),
        'roadm-osnr': roadm.get('add_drop_osnr'),
        'roadm-maxloss': None,
        'roadm-pmax': None
    }
    incomplete_value = {
        'roadm-pmd': [None, 0],
        'roadm-pdl': [None, 0],
        'roadm-osnr': [None],
        'roadm-maxloss': [None, 0],
        'roadm-pmax': [None],
    }
    cons, comp = [], []
    n_cons = n_comp = 0
    n_comp += 1
    roadm_impairments = roadm.get('roadm-path-impairments', [])
    if not roadm_impairments:
        comp.append(Issue(Axis.COMPLETENESS, 'PATH_IMPAIRMENTS_MISSING',
                          'Missing  roadm-path-impairments',
                          ROADM, name, weight=0.5))
    for roadm_path in roadm.get('roadm-path-impairments', []):
        # add/drop attributes
        for path in ["roadm-add-path", "roadm-drop-path"]:
            for key in ['roadm-pmd', 'roadm-pdl', 'roadm-maxloss', 'roadm-pmax', 'roadm-osnr']:
                for item in roadm_path.get(path, []):
                    n_comp += 1
                    if default[key] in incomplete_value[key] and item.get(key) in incomplete_value[key]:
                        comp.append(Issue(Axis.COMPLETENESS, 'IMPAIRMENT_KIND_MISSING',
                                          f'Missing {key} in ROADM {path} '
                                          f'{item["frequency-range"]["lower-frequency"] * 1e-12:.2f}-'
                                          f'{item["frequency-range"]["upper-frequency"] * 1e-12:.2f}',
                                          ROADM, name, weight=0.5))
        # express attributes
        for key in ['roadm-pmd', 'roadm-pdl', 'roadm-maxloss']:
            for item in roadm_path.get('roadm-express-path', []):
                n_comp += 1
                if default[key] in incomplete_value[key] and item.get(key) in incomplete_value[key]:
                    comp.append(Issue(Axis.COMPLETENESS, 'IMPAIRMENT_KIND_MISSING',
                                      f'Missing {key} in ROADM roadm-express-path '
                                      f'{item["frequency-range"]["lower-frequency"] * 1e-12:.2f}-'
                                      f'{item["frequency-range"]["upper-frequency"] * 1e-12:.2f}',
                                      ROADM, name, weight=0.5))

    # check that values in different bands for the same roadm-path are consistent
    for roadm_path in roadm.get('roadm-path-impairments', []):
        for path in ["roadm-add-path", "roadm-drop-path", "roadm-express-path"]:
            for key in ['roadm-pmd', 'roadm-pdl', 'roadm-maxloss', 'roadm-pmax', 'roadm-osnr']:
                n_cons += 1
                values = [item.get(key) for item in roadm_path.get(path, []) if item.get(key) is not None]
                bands = [f'{item["frequency-range"]["lower-frequency"] * 1e-12:.2f}-'
                         f'{item["frequency-range"]["upper-frequency"] * 1e-12:.2f}'
                         for item in roadm_path.get(path, [])]
                if values and abs(max(values) - min(values)) > tolerance:
                    cons.append(
                        Issue(Axis.CONSISTENCY, 'IMPAIRMENT_TOO_DIFFERENT_BETWEEN_BANDS',
                              f'Impairment {key} {values} in ROADM {path} is too different between the bands {bands}'
                              f'{item["frequency-range"]["lower-frequency"] * 1e-12:.2f}-'
                              f'{item["frequency-range"]["upper-frequency"] * 1e-12:.2f}',
                              ROADM, name, weight=0.5))

    return n_cons, cons, n_comp, comp


def check_equalization_information(
    roadm_entries: List[RoadmEntry],
    transceiver_entries: List[ModeEntry],
) -> Tuple[int, List[Issue], int]:
    """Check that equalization targets and offsets are not mixed.

    Rules:

    If the ROADMs mix several kinds of equalization targets while transceiver
    offsets are defined, the information is inconsistent.

    :param roadm_entries: ROADM entries to validate.
    :param transceiver_entries: Transceiver mode entries (reference channel included).
    :return: A tuple containing, in order, the number of consistency checks,
        consistency issues, completeness checks, and completeness issues.
    """
    kinds = _equalization_kinds(roadm_entries)
    if not kinds:
        return 0, []

    has_offset = any(e.offset_defined for e in transceiver_entries if e.mode != 'reference_mode')
    if len(kinds) > 1 and has_offset:
        return 1, [Issue(
            Axis.CONSISTENCY, 'EQUALIZATION_MODES_MIXED',
            f'ROADMs mix several equalization targets ({", ".join(sorted(kinds))}) '
            'while transceiver equalization offsets are defined.',
            section=ROADM, weight=0.5)]

    return 1, []


def _equalization_kinds(roadm_entries: List[RoadmEntry]) -> set[str]:
    """Return the equalization target kinds used by the ROADMs.

    :param roadm_entries: ROADM entries.
    :return: Set of target names defined in at least one ROADM.
    """
    kinds = set()
    for roadm in roadm_entries:
        if roadm.target_pch_out_db is not None:
            kinds.add('target_pch_out_db')
        if roadm.target_psd_out_mWperGHz is not None:
            kinds.add('target_psd_out_mWperGHz')
        if roadm.target_out_mWperSlotWidth is not None:
            kinds.add('target_out_mWperSlotWidth')
    return kinds


def check_offset_presence(
    roadm_entries: List[RoadmEntry],
    transceiver_entries: List[ModeEntry],
    reference: ModeEntry | None = None,
    baud_tol: float = 5e9,
) -> Tuple[int, List[Issue]]:
    """Check that modes with a baud rate different from the reference have an offset.

    The check only applies if a ROADM uses target_pch_out_db.

    :param roadm_entries: ROADM entries.
    :param transceiver_entries: Transceiver mode entries (reference included).
    :param reference: Reference channel, if available.
    :param baud_tol: Baud rate difference under which two modes are considered identical.
    :return: Number of checks performed and detected issues.
    """
    n_checks, issues = 0, []
    if 'target_pch_out_db' not in _equalization_kinds(roadm_entries):
        return n_checks, issues

    modes = [e for e in transceiver_entries if e.mode != 'reference_mode']
    if not modes:
        return n_checks, issues
    ref_baud = reference.baud_rate if reference else median(e.baud_rate for e in modes)

    for entry in modes:
        if abs(entry.baud_rate - ref_baud) <= baud_tol:
            continue
        n_checks += 1
        if not entry.offset_defined:
            issues.append(Issue(
                Axis.COMPLETENESS, 'OFFSET_MISSING',
                f'Missing equalization_offset_db: baud_rate={entry.baud_rate * 1e-9:.2f} Gbaud differs '
                f'from reference {ref_baud * 1e-9:.2f} Gbaud and ROADMs use target_pch_out_db.',
                TRANSCEIVER, entry.transceiver, entry.mode, weight=0.5))
    return n_checks, issues


# ----------------------------------------------------------------------
# AMPLIFIERS consistency
# ----------------------------------------------------------------------
def check_multi_band_amplifier_frequency_range(
    edfa_entries: List[EdfaEntry],
) -> Tuple[int, List[Issue]]:
    """Check that the amplifiers referenced by a multiband amplifier cover several bands.

    :param amplifier: item of the library Edfa list.
    :param edfa_entries: Edfa entries indexed by type_variety and other_name.
    :return: Number of checks performed and detected issues.
    """
    n_checks = 0
    issues: list[Issue] = []
    for edfa_entry in edfa_entries:
        name = edfa_entry.amplifier
        if edfa_entry.type_def != 'multi_band':
            continue

        ranges = []
        for ref in edfa_entry.amplifiers:
            entry = next((e for e in edfa_entries if e.amplifier == ref or ref in e.alias), None)
            if entry is None or entry.f_min is None or entry.f_max is None:
                continue
            n_checks += 1
            ranges.append((entry.f_min, entry.f_max))
        # merge bands that overlap
        merged = []
        for f_min, f_max in sorted(ranges):
            if merged and f_min < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], f_max))
            else:
                merged.append((f_min, f_max))
        if len(merged) < 2:
            issues.append(Issue(
                Axis.CONSISTENCY, 'INVALID_MULTIBAND_SINGLE_BAND',
                f'Multiband amplifier references amplifiers covering {len(merged)} distinct '
                f'frequency range(s) ({merged}): at least 2 are expected.',
                section=AMPLIFIER,
                item=name,
                weight=0.5))
    return n_checks, issues


# ----------------------------------------------------------------------
# SI consistency
# ----------------------------------------------------------------------
def check_reference_channel_with_amp_max_power(
    edfa_entries: List[EdfaEntry],
    reference: ModeEntry,
) -> tuple[int, list[Issue]]:
    """Check that reference channel defined in SI is consistent with max power of EDFA are
    compatible with the transceiver with the lowest penalty tolerance.

    :param edfa_entries: amplifiers entries to validate.
    :param reference: reference mode entries.
    :return: Number of checks performed and detected issues.
    """
    n_checks = 0
    issues: list[Issue] = []

    if not reference:
        return n_checks, issues

    total_power = reference.offset + lin2db(automatic_nch(reference.f_min, reference.f_max, reference.min_spacing))
    for edfa in [e for e in edfa_entries if e.pmax is not None]:
        n_checks += 1
        if not edfa.raman and edfa.pmax < total_power:
            issues.append(Issue(
                Axis.CONSISTENCY, 'MAX_POWER_TOO_LOW',
                f'Amplifier {edfa.amplifier} max power {edfa.pmax:.2f}dBm is below reference spectrum power'
                f' {total_power:.2f}dBm.',
                section=AMPLIFIER,
                item=edfa.amplifier,
                weight=0.5))
    return n_checks, issues


# -------------------------------------
# Check library meta data
# -------------------------------------
def check_metadata(data: LEGACY_TYPE, filename: str) -> tuple[int, list[Issue]]:
    """Check that library-information is consistent with the file name.

    The name must match the file name (without date and extension), and the most recent
    revision date must match the date included in the file name.

    :param data: Library data (YANG formatted).
    :param filename: File name or path of the library.
    :param report: Validation report.
    :return: Number of checks performed and detected issues.
    """
    n_checks, issues = 0, []
    library_filename = data.get("library-information", {}).get("name", None)

    if library_filename and library_filename != filename:
        n_checks += 1
        issues.append(Issue(Axis.FORMAT, 'LIBRARY_NAME_MISMATCH',
                            f'library-information/name "{library_filename}" does not match file name "{filename}"',
                            LIBRARY, weight=0.5))
    return n_checks, issues


# -----------------------------------------
# Build entries to check
# -----------------------------------------
def _build_transceiver_entries(
    transceivers: list[LEGACY_TYPE],
    si: Dict[str, LEGACY_TYPE] | None,
    report: Report,
) -> tuple[list[ModeEntry], ModeEntry | None]:
    """Build transceiver entries and validate individual modes.

    :param transceivers: library's Transceiver section
    :param si: library's SI section
    :param report: Validation report.
    """
    entries: list[ModeEntry] = []
    reference = None
    if si:
        reference = ModeEntry(
            transceiver='reference_channel',
            mode='reference_mode',
            baud_rate=si['baud_rate'],
            min_spacing=si['spacing'],
            offset=si['power_dbm'],   # power_dBm is used as offset: 0dBm + offset = power_dbm
            tx_osnr=si['tx_osnr'],
            tx_power=si.get('tx_power_dbm', si['power_dbm']),
            f_min=si['f_min'],
            f_max=si['f_max'])
        entries.append(reference)

    for trx in transceivers:
        trx_name = trx['type_variety']
        for mode in trx['mode']:
            n_cons, cons, n_comp, comp = check_mode(mode, trx_name)
            report.extend(TRANSCEIVER, Axis.CONSISTENCY, n_cons, cons)
            report.extend(TRANSCEIVER, Axis.COMPLETENESS, n_comp, comp)
            penalties = mode.get('penalties') or []
            pmd_values = [elem['pmd'] for elem in penalties if elem.get('pmd') is not None]
            pdl_values = [elem['pdl'] for elem in penalties if elem.get('pdl') is not None]

            entries.append(
                ModeEntry(
                    transceiver=trx_name,
                    mode=mode['format'],
                    baud_rate=mode['baud_rate'],
                    min_spacing=mode['min_spacing'],
                    # no offset key means offset is 0dB in GNPy
                    offset=mode.get('equalization_offset_db', 0),
                    offset_defined=mode.get('equalization_offset_db') is not None,
                    bit_rate=(
                        mode['bit_rate']
                        if mode.get('bit_rate') is not None
                        else None
                    ),
                    osnr=(
                        mode['OSNR']
                        if mode.get('OSNR') is not None
                        else None
                    ),
                    penalties_pmd=max(pmd_values, default=None),
                    penalties_pdl=max(pdl_values, default=None),
                    tx_osnr=mode['tx_osnr'],
                    tx_power=mode.get('tx-channel-power-max-dbm', None)
                )
            )

    return entries, reference


def _build_roadm_entries(
    roadms: list[LEGACY_TYPE],
    report: Report,
) -> list[RoadmEntry]:
    """Build internal ROADM entries from legacy data.

    :param roadms: library's Roadm section
    :param report: Validation report.
    """
    entries = []

    for roadm in roadms:
        add_pmd = []
        drop_pmd = []
        add_pdl = []
        drop_pdl = []
        add_osnr = []
        drop_osnr = []
        add_maxloss = []
        drop_maxloss = []
        n_cons, cons, n_comp, comp = check_consistency_of_roadm_path_impairment(roadm)
        report.extend(ROADM, Axis.CONSISTENCY, n_cons, cons)
        report.extend(ROADM, Axis.COMPLETENESS, n_comp, comp)
        for item in roadm.get('roadm-path-impairments', []):
            for subitem in item.get('roadm-add-path', []):
                add_pmd.append(float(subitem.get('roadm-pmd', 0)))
                add_pdl.append(float(subitem.get('roadm-pdl', 0)))
                add_osnr.append(subitem.get('roadm-osnr', None))
                add_maxloss.append(subitem.get('roadm-maxloss', 0))
            for subitem in item.get('roadm-drop-path', []):
                drop_pmd.append(float(subitem.get('roadm-pmd', 0)))
                drop_pdl.append(float(subitem.get('roadm-pdl', 0)))
                drop_osnr.append(subitem.get('roadm-osnr', None))
                drop_maxloss.append(subitem.get('roadm-maxloss', 0))
        # filter out  None values
        add_osnr = [v for v in add_osnr if v is not None]
        drop_osnr = [v for v in drop_osnr if v is not None]
        entries.append(
            RoadmEntry(
                roadm=roadm.get('type_variety', 'default'),
                add_drop_osnr=roadm.get('add_drop_osnr'),
                pmd=roadm.get('pmd', 0) * 1e12,  # convert in ps as transceivers' impairments are in ps,
                pdl=roadm.get('pdl', 0),
                target_pch_out_db=roadm.get('target_pch_out_db'),
                target_psd_out_mWperGHz=roadm.get('target_psd_out_mWperGHz'),
                target_out_mWperSlotWidth=roadm.get('target_out_mWperSlotWidth'),
                detailed_add_pmd=min(add_pmd, default=0),
                detailed_drop_pmd=min(drop_pmd, default=0),
                detailed_add_pdl=min(add_pdl, default=0),
                detailed_drop_pdl=min(drop_pdl, default=0),
                detailed_add_osnr=min(add_osnr, default=0),
                detailed_drop_osnr=min(drop_osnr, default=0),
                detailed_add_maxloss=min(add_maxloss, default=0),
                detailed_drop_maxloss=min(drop_maxloss, default=0)
            )
        )

    return entries


def _build_edfa_entries(
    edfas: list[LEGACY_TYPE],
    report: Report,
) -> list[EdfaEntry]:
    """Build internal amplifiers entries from legacy data.

    :param roadms: library's Edfa section
    :param report: Validation report.
    """
    entries = []

    for edfa in edfas:
        n_cons, cons, n_comp, comp = check_edfa(edfa)
        report.extend(AMPLIFIER, Axis.CONSISTENCY, n_cons, cons)
        report.extend(AMPLIFIER, Axis.COMPLETENESS, n_comp, comp)
        entries.append(
            EdfaEntry(
                amplifier=edfa.get('type_variety', 'default'),
                f_min=edfa.get('f_min'),
                f_max=edfa.get('f_max'),
                type_def=edfa.get('type_def'),
                amplifiers=edfa.get('amplifiers', []),
                pmax=edfa.get('p_max'),
                alias=edfa.get('other_name', []),
                raman=edfa.get('raman', False) and edfa.get('type_def') == 'fixed_gain'  # then this is a Raman amp
            )
        )

    return entries


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def validate(data: LEGACY_TYPE | YANG_TYPE, filename: str | None = None) -> Report:
    """Validate an equipment library.

    The input dictionary is not modified.

    :param data: Equipment library data already loaded in memory.
    :param filename: file name for the library
    :return: Validation report containing performed checks and issues.
    """
    report = Report()
    legacy, issues = check_yang(data)
    report.extend(LIBRARY, Axis.FORMAT, 1, issues)
    if legacy is None:
        return report
    report.legacy_data = legacy

    if filename:
        n_checks, issues = check_metadata(legacy, filename)
        report.extend(LIBRARY, Axis.FORMAT, n_checks, issues)

    # Transceivers checks
    si = (legacy.get('SI') or [None])[0]
    entries, reference = _build_transceiver_entries(
        transceivers=legacy.get('Transceiver', []),
        si=si,
        report=report,
    )

    n_checks, issues = check_equalization_offset_consistency(entries, reference)
    report.extend(TRANSCEIVER, Axis.CONSISTENCY, n_checks, issues)
    osnr_entries = [e for e in entries if e.mode != 'reference_mode' and e.osnr is not None and e.bit_rate is not None]
    n_checks, issues = check_osnr_consistency(osnr_entries)
    report.extend(TRANSCEIVER, Axis.CONSISTENCY, n_checks, issues)

    # Roadms checks
    roadm_entries = _build_roadm_entries(legacy.get('Roadm', []), report=report)
    n_checks, issues = (check_roadm_pmd_pdl_consistency_with_transceivers(roadm_entries=roadm_entries,
                                                                          transceiver_entries=entries))
    report.extend(ROADM, Axis.CONSISTENCY, n_checks, issues)
    n_checks, issues = (check_roadm_add_drop_osnr_consistency_with_transceivers(roadm_entries=roadm_entries,
                                                                                transceiver_entries=entries))
    report.extend(ROADM, Axis.CONSISTENCY, n_checks, issues)
    n_checks, issues = (check_roadm_target_power_with_transceivers(roadm_entries=roadm_entries,
                                                                   transceiver_entries=entries))
    report.extend(ROADM, Axis.CONSISTENCY, n_checks, issues)

    n_checks, issues = check_offset_presence(roadm_entries, entries, reference)
    report.extend(TRANSCEIVER, Axis.COMPLETENESS, n_checks, issues)
    n_checks, issues = check_equalization_information(roadm_entries, entries)
    report.extend(ROADM, Axis.CONSISTENCY, n_checks, issues)

    # Amplifiers checks
    edfa_entries = _build_edfa_entries(legacy.get('Edfa', []), report=report)
    # frequency range of amplifiers referenced by multiband amplifiers must cover several bands
    n_checks, issues = (check_multi_band_amplifier_frequency_range(edfa_entries=edfa_entries))
    report.extend(AMPLIFIER, Axis.CONSISTENCY, n_checks, issues)

    # max power should be enough to support the reference spectrum
    n_checks, issues = (check_reference_channel_with_amp_max_power(edfa_entries=edfa_entries,
                                                                   reference=reference))
    report.extend(AMPLIFIER, Axis.CONSISTENCY, n_checks, issues)
    # the reference power should be consistent with the reference span
    # reference spectrum should be included in the amplifier largest band (careful about the border)

    return report


# ----------------------------------------------------------------------
# Score (pure computation on a report)
# ----------------------------------------------------------------------
SECTION_WEIGHTS = {            # weight of each section when computing the global score
    'Transceiver': 0.4,
    'Edfa': 0.2,
    'Roadm': 0.2,
    'Fiber': 0.1,
    'SI': 0.1,
}


def axis_score(report: Report, section: str, axis: Axis) -> float | None:
    """Compute the score for one section and one validation axis.

    :param report: Validation report.
    :param section: Library section to score.
    :param axis: Validation axis to score.
    :return: Score between 0 and 100, or ``None`` if no check was performed.
    """
    total = report.checked.get((section, axis), 0)
    if total == 0:
        return None
    penalty = sum(i.weight for i in report.issues
                  if i.section == section and i.axis == axis)
    return max(0.0, 100 * (1 - penalty / total))


def _weighted(scores: Dict, weights: Dict) -> float | None:
    """Compute a weighted average of available scores.

    Scores with a value of ``None`` are excluded from the calculation.

    :param scores: Scores indexed by their corresponding key.
    :param weights: Weight associated with each score key.
    :return: Weighted average, or ``None`` if no score is available.
    """
    valid = {k: s for k, s in scores.items() if s is not None}
    if not valid:
        return None
    total_w = sum(weights[k] for k in valid)
    return sum(weights[k] * s for k, s in valid.items()) / total_w


def section_score(report: Report, section: str) -> tuple[float | None, dict[Axis, float | None]]:
    """Compute the score for one library section.

    :param report: Validation report.
    :param section: Library section to score.
    :return: A tuple containing the section score and the score for each axis.
    """
    scores = {a: axis_score(report, section, a)
              for a in (Axis.CONSISTENCY, Axis.COMPLETENESS)}
    return _weighted(scores, WEIGHTS), scores


@dataclass
class Score:
    """Store global, format, and per-section validation scores.

    :param total: Global library score.
    :param format: Global format score.
    :param sections: Scores grouped by section.
    """
    total: float
    format: float | None
    sections: dict            # section -> (score, {axis: score})


def score(report: Report) -> Score:
    """Compute all scores from a validation report.

    :param report: Validation report.
    :return: Global score, format score, and per-section scores.
    """
    fmt = axis_score(report, LIBRARY, Axis.FORMAT)
    sections = {s: section_score(report, s) for s in report.sections if s != LIBRARY}

    if any(issue.code in ZERO_SCORE_ISSUE_CODES for issue in report.issues):
        return Score(0.0, fmt, sections)

    if fmt == 0:
        return Score(0.0, 0.0, sections)
    content = _weighted({s: v[0] for s, v in sections.items()},
                        {s: SECTION_WEIGHTS.get(s, 1.0) for s in sections})
    if content is None:
        return Score(fmt or 0.0, fmt, sections)
    total = (WEIGHTS[Axis.FORMAT] * (fmt if fmt is not None else 100)
             + (1 - WEIGHTS[Axis.FORMAT]) * content)
    return Score(total, fmt, sections)


def format_score(score_result: Score) -> str:
    """Format validation scores for human-readable output.

    :param score_result: Scores to format.
    :return: Multiline textual representation of the scores.
    """
    lines = [f'Score: {score_result.total: .1f} (format={score_result.format})']
    for section, (value, axes) in score_result.sections.items():
        detail = ', '.join(f'{a.value}={"n/a" if v is None else f"{v: .1f}"}' for a, v in axes.items())
        lines.append(f'  {section}: {"n/a" if value is None else f"{value: .1f}"} ({detail})')
    return '\n'.join(lines)


def _aggregate_axis(report: Report, axis: Axis) -> float:
    """Score of one axis aggregated over all sections (100 if nothing was checked).

    :param report: Validation report.
    :param axis: Validation axis to score.
    """
    checks = sum(n for (_, a), n in report.checked.items() if a == axis)
    if checks == 0:
        return 100.0
    penalty = sum(i.weight for i in report.issues if i.axis == axis)
    return max(0.0, 100 * (1 - penalty / checks))


def is_eligible(report: Report, min_consistency: float = 100.0,
                min_completeness: float = 75.0) -> tuple[bool, list[str]]:
    """Return whether a proof of validation can be issued, and the reasons if not.

    :param report: Validation report.
    :param min_consistency: consistency threshold.
    :param min_completeness: completeness threshold.
    """
    reasons = []
    if score(report).total <= 0:
        reasons.append('Global score is zero')
    consistency = _aggregate_axis(report, Axis.CONSISTENCY)
    if consistency < min_consistency:
        reasons.append(f'consistency {consistency:.1f} < {min_consistency:.1f}')
    completeness = _aggregate_axis(report, Axis.COMPLETENESS)
    if completeness < min_completeness:
        reasons.append(f'completeness {completeness:.1f} < {min_completeness:.1f}')
    return not reasons, reasons
