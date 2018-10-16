#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_service_sheet.py
========================

XLS parser that can be called to create a JSON request file in accordance with
Yang model for requesting path computation.

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from argparse import ArgumentParser
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps

from gnpy.core.service_sheet import Request, Element, Request_element
from gnpy.core.service_sheet import parse_row, parse_excel, convert_service_sheet

logger = getLogger(__name__)

if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={2: DEBUG, 1: INFO, 0: CRITICAL}.get(args.verbose, CRITICAL))
    logger.info(f'Converting Service sheet {args.workbook!r} into gnpy JSON format')
    if args.output is None:
        data = convert_service_sheet(args.workbook,'eqpt_config.json')
        print(dumps(data, indent=2))
    else:
        data = convert_service_sheet(args.workbook,'eqpt_config.json',args.output)
