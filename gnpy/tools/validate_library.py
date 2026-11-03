#!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.validate_library: CLI for library validation
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""Validate equipment libraries"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import hashlib
import sys

from gnpy.tools.library_checks import Report, score, validate, format_score, is_eligible
from gnpy.tools.proof import write_proof


def log_report(report: Report):
    """Log one warning for each validation issue.

    :param report: Validation report to log.
    :return: None.
    """
    for issue in report.issues:
        where = '/'.join(x for x in (issue.section, issue.item, issue.sub_item) if x)
        logging.warning('[%s] %s%s', issue.code, f'{where}: ' if where else '', issue.message)


def to_json(report: Report) -> str:
    """Serialize a validation report and its scores to JSON.

    :param report: Validation report to serialize.
    :return: JSON-formatted validation summary.
    """
    sc = score(report)
    return json.dumps({
        'score': round(sc.total, 1),
        'format': None if sc.format is None else round(sc.format, 1),
        'sections': {
            s: {'score': None if v is None else round(v, 1),
                'axes': {a.value: None if x is None else round(x, 1) for a, x in axes.items()}}
            for s, (v, axes) in sc.sections.items()
        },
        'issues': [{'axis': i.axis.value, 'code': i.code, 'section': i.section,
                    'item': i.item, 'sub_item': i.sub_item,
                    'message': i.message, 'weight': i.weight}
                   for i in report.issues],
    }, indent=2)


def main():
    """Run the equipment-library validation command-line interface.
    """
    parser = ArgumentParser()
    parser.add_argument('library_filename', type=Path, help='Input JSON file')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Increase verbosity (can be specified several times)')
    parser.add_argument('--json', action='store_true', help='Print a JSON summary')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Exit with code 1 if the global score is below this value')
    parser.add_argument('--proof-dir', type=Path,
                        help='Write the certified copy and the proof of validation in this directory')
    parser.add_argument('--organization', help='Organization name (required with --proof-dir)')
    parser.add_argument('--contact', help='Contact email (required with --proof-dir)')
    parser.add_argument('--min-completeness', type=float, default=75.0,
                        help='Minimum completeness required for a proof (default: 75)')
    parser.add_argument('--min-consistency', type=float, default=98.0,
                        help='Minimum consistency required for a proof (default: 100)')
    args = parser.parse_args()
    logging.basicConfig(level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
                        format='%(message)s')

    raw = args.library_filename.read_bytes()
    try:
        data = json.loads(raw.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        logging.exception('File %s has json issues: %s. Analysis stopped.', args.library_filename, err)
        sys.exit(2)

    report = validate(data, args.library_filename.name)
    log_report(report)
    sc = score(report)
    logging.warning('\n%s', format_score(sc))
    if args.json:
        print(to_json(report))
    if args.min_score is not None and sc.total < args.min_score:
        sys.exit(1)
    logging.warning('SHA-256: %s', hashlib.sha256(raw).hexdigest())

    if args.proof_dir:
        if not (args.organization and args.contact):
            parser.error('--organization and --contact are required with --proof-dir')
        eligible, reasons = is_eligible(report, args.min_consistency, args.min_completeness)
        if not eligible:
            logging.error('No proof of validation generated: %s', '; '.join(reasons))
            sys.exit(1)
        profile = {'global_min_exclusive': 0, 'consistency_min': args.min_consistency,
                   'completeness_min': args.min_completeness}
        library_path, proof_path = write_proof(raw, args.library_filename.name, sc.total, profile,
                                               args.organization, args.contact, args.proof_dir)
        logging.warning('Certified library: %s\nProof of validation: %s', library_path, proof_path)


if __name__ == '__main__':
    main()
