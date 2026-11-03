# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause# SPDX-License-Identifier: BSD-3-Clause
# test_library_checks
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

import copy
import json
from pathlib import Path

import pytest

from gnpy.tools.library_checks import Axis, LIBRARY, TRANSCEIVER, score, validate
from gnpy.tools.validate_library import to_json
from gnpy.tools.proof import write_proof, SCHEMA_VERSION


DATA_DIR = Path(__file__).parent / 'data' / 'library_checks'
LIBRARIES = ['invalid_yang', 'invalid_bitrate', 'valid_yang', 'valid_legacy']
SNAPSHOT = DATA_DIR / 'proof_schema_snapshot.json'


def load_library(filename: str) -> dict:
    """Load a JSON library fixture.

    :param filename: Fixture filename.
    :return: Parsed library data.
    """
    with (DATA_DIR / filename).open(encoding='utf-8') as stream:
        return json.load(stream)


def issue_codes(report) -> set[str]:
    """Return all issue codes from a validation report.

    :param report: Validation report.
    :return: Set of issue codes.
    """
    return {issue.code for issue in report.issues}


def issues_for(report, code: str):
    """Return all issues having the given code.

    :param report: Validation report.
    :param code: Issue code.
    :return: Matching issues.
    """
    return [issue for issue in report.issues if issue.code == code]


def test_invalid_yang_library_gets_zero_format_score():
    """An invalid YANG library must not receive a perfect score."""
    data = load_library('invalid_yang.json')

    report = validate(data, 'invalid_yang.json')
    result = score(report)

    assert report.checked[(LIBRARY, Axis.FORMAT)] == 1
    assert 'YANG_INVALID' in issue_codes(report)

    yang_issues = issues_for(report, 'YANG_INVALID')
    assert len(yang_issues) == 22
    assert yang_issues[0].section == LIBRARY

    assert result.format == 0
    assert result.total == 0
    assert result.sections == {}


def test_valid_yang_library_passes_format_check():
    """A valid YANG library must pass the format check."""
    data = load_library('valid_yang.json')

    report = validate(data, 'valid_yang.json')
    result = score(report)

    assert report.checked[(LIBRARY, Axis.FORMAT)] == 1
    assert 'YANG_INVALID' not in issue_codes(report)
    assert 'NOT_RFC7951' not in issue_codes(report)

    assert report.legacy_data is not None
    assert result.format == 100


def test_legacy_library_is_reported_but_remains_usable():
    """A valid legacy library must be usable with a reduced format score."""
    data = load_library('valid_legacy.json')

    report = validate(data, 'valid_legacy.json')
    result = score(report)

    assert 'NOT_RFC7951' in issue_codes(report)
    assert 'YANG_INVALID' not in issue_codes(report)

    assert report.legacy_data is not None
    assert result.format == pytest.approx(60.0)


def test_validation_does_not_modify_input_data():
    """Validation must not modify the input dictionary."""
    data = load_library('valid_legacy.json')
    original = copy.deepcopy(data)

    validate(data, 'valid_legacy.json')

    assert data == original


def test_invalid_bit_rate_is_reported_on_the_transceiver_section():
    """An invalid bit rate must affect the transceiver consistency score."""
    data = load_library('invalid_bitrate.json')

    report = validate(data, 'invalid_bitrate.json')
    result = score(report)

    issues = issues_for(report, 'BIT_RATE_INVALID')

    assert len(issues) == 1
    assert issues[0].section == TRANSCEIVER
    assert issues[0].item is not None
    assert issues[0].sub_item is not None

    assert report.checked[(TRANSCEIVER, Axis.CONSISTENCY)] >= 1

    consistency_score = result.sections[TRANSCEIVER][1][Axis.CONSISTENCY]
    assert consistency_score == pytest.approx(82.291, abs=1e-3)


def test_missing_completeness_information_is_reported():
    """Missing penalties and offsets must affect completeness scoring."""
    data = load_library('valid_yang.json')

    report = validate(data, 'valid_yang.json')
    result = score(report)

    codes = issue_codes(report)

    assert 'PENALTIES_MISSING' in codes
    assert 'OFFSET_MISSING' in codes

    completeness_score = result.sections[TRANSCEIVER][1][Axis.COMPLETENESS]
    assert completeness_score is not None
    assert completeness_score < 100


def test_inconsistent_offsets_are_reported_for_each_affected_mode():
    """Each inconsistent mode must receive its own issue."""
    data = load_library('valid_yang.json')

    report = validate(data, 'valid_yang.json')

    issues = issues_for(report, 'OFFSET_INCONSISTENT')

    assert len(issues) == 2
    assert {issue.section for issue in issues} == {TRANSCEIVER}
    assert {issue.item for issue in issues} == {'Voyager'}
    assert {issue.sub_item for issue in issues} == {
        'mode 2 - fake',
        'mode 2',
    }

    issues = issues_for(report, 'OFFSET_NOT_PROPORTIONAL')
    assert len(issues) == 4
    assert {issue.section for issue in issues} == {TRANSCEIVER}
    assert {issue.item for issue in issues} == {'vendorA_trx-type1', 'Voyager'}
    assert {issue.sub_item for issue in issues} == {'PS_SP64_2', 'mode 4', 'mode 3', 'mode 2'}


def test_inconsistent_osnr_are_reported_for_each_affected_mode():
    """Each inconsistent mode must receive its own issue."""
    data = load_library('valid_legacy.json')

    report = validate(data, 'valid_legacy.json')

    issues = issues_for(report, 'OSNR_INCONSISTENT')

    assert len(issues) == 1
    assert {issue.section for issue in issues} == {TRANSCEIVER}
    assert {issue.item for issue in issues} == {'vendorA_trx-type1'}
    assert {issue.sub_item for issue in issues} == {'mode 2'}


def test_sections_are_unique_and_sorted():
    """Report sections must not contain duplicates."""
    data = load_library('valid_legacy.json')

    report = validate(data, 'valid_legacy.json')

    assert report.sections == sorted(set(report.sections))
    assert len(report.sections) == len(set(report.sections))


def test_score_contains_transceiver_axis_scores():
    """The score must expose the transceiver section and its axes."""
    data = load_library('valid_legacy.json')

    report = validate(data, 'valid_legacy.json')
    result = score(report)

    assert TRANSCEIVER in result.sections

    section_score, axis_scores = result.sections[TRANSCEIVER]

    assert section_score is not None
    assert Axis.CONSISTENCY in axis_scores
    assert Axis.COMPLETENESS in axis_scores


@pytest.mark.parametrize('name', LIBRARIES)
def test_report_matches_expected(name):
    """The report of each library must match its expected file."""
    data = load_library(f'{name}.json')
    report = validate(data, f'{name}.json')
    actual = json.loads(to_json(report))

    expected_file = DATA_DIR / f'{name}_expected.json'
    with expected_file.open(encoding='utf-8') as stream:
        expected = json.load(stream)

    assert actual == expected


def _shape(value, prefix=''):
    """Return the sorted list of 'path: type' entries describing a JSON structure."""
    if isinstance(value, dict):
        out = []
        for key, sub in value.items():
            out.extend(_shape(sub, f'{prefix}/{key}'))
        return out
    return [f'{prefix}: {type(value).__name__}']


def _current_schema(tmp_path):
    profile = {'global_min_exclusive': 0, 'consistency_min': 98.0, 'completeness_min': 75.0}
    _, proof_path = write_proof(b'{}', 'equipment.json', 87.3, profile,
                                'Vendor', 'name@vendor.com', tmp_path)
    content = json.loads(proof_path.read_text(encoding='utf-8'))
    return sorted(_shape(content))


def test_proof_schema_version_follows_format(tmp_path):
    """The proof format must not change without a change of SCHEMA_VERSION."""
    current = {'schema_version': SCHEMA_VERSION, 'schema': _current_schema(tmp_path)}
    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8')
        raise AssertionError(f'Snapshot created: {SNAPSHOT}. Review it and commit it.')

    expected = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    if current['schema'] != expected['schema']:
        if current['schema_version'] == expected['schema_version']:
            raise AssertionError(
                'The proof format changed but SCHEMA_VERSION is still '
                f'{current["schema_version"]}. Increase SCHEMA_VERSION in proof.py '
                'and update the snapshot.')
        raise AssertionError(
            'The proof format and SCHEMA_VERSION changed. Update the snapshot '
            f'{SNAPSHOT.name} with the new schema and version.')
    assert current['schema_version'] == expected['schema_version'], \
        'SCHEMA_VERSION changed without any change of the proof format.'
