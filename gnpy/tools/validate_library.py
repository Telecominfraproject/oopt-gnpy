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
import sys

from gnpy.tools.library_checks import Report, score, validate, format_score


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
    args = parser.parse_args()
    logging.basicConfig(level={0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG),
                        format='%(message)s')

    try:
        with open(args.library_filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as err:
        logging.error('File %s has json issues: %s. Analysis stopped.', args.library_filename, err)
        sys.exit(2)

    report = validate(data, args.library_filename.name)
    log_report(report)
    sc = score(report)
    logging.warning('\n%s', format_score(sc))
    if args.json:
        print(to_json(report))
    if args.min_score is not None and sc.total < args.min_score:
        sys.exit(1)


if __name__ == '__main__':
    main()
