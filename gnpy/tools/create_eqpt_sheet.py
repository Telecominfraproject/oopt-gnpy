#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# Utility functions that creates an Eqpt sheet template
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
create_eqpt_sheet.py
====================

XLS parser that can be called to create a "City" column in the "Eqpt" sheet.

If not present in the "Nodes" sheet, the "Type" column will be implicitly
determined based on the topology.
"""

from argparse import ArgumentParser
from pathlib import Path
import csv
from typing import List, Dict, Optional
from logging import getLogger
import dataclasses

from gnpy.core.exceptions import NetworkTopologyError
from gnpy.tools.xls_utils import generic_open_workbook, get_sheet, XLS_EXCEPTIONS, all_rows, fast_get_sheet_rows, \
    WorkbookType, SheetType


logger = getLogger(__name__)
EXAMPLE_DATA_DIR = Path(__file__).parent.parent / 'example-data'

PARSER = ArgumentParser()
PARSER.add_argument('workbook', type=Path, nargs='?', default=f'{EXAMPLE_DATA_DIR}/meshTopologyExampleV2.xls',
                    help='create the mandatory columns in Eqpt sheet')
PARSER.add_argument('-o', '--output', type=Path, help='Store CSV file')


@dataclasses.dataclass
class Node:
    """Represents a network node with a unique identifier, connected nodes, and equipment type.

    :param uid: Unique identifier of the node.
    :type uid: str
    :param to_node: List of connected node identifiers.
    :type to_node: List[str.]
    :param eqpt: Equipment type associated with the node (ROADM, ILA, FUSED).
    :type eqpt: str
    """
    def __init__(self, uid: str, to_node: List[str], eqpt: str = None):
        self.uid = uid
        self.to_node = to_node
        self.eqpt = eqpt


def open_sheet_with_error_handling(wb: WorkbookType, sheet_name: str, is_xlsx: bool) -> SheetType:
    """Opens a sheet from the workbook with error handling.

    :param wb: The opened workbook.
    :type wb: WorkbookType
    :param sheet_name: Name of the sheet to open.
    :type sheet_name: str
    :param is_xlsx: Boolean indicating if the file is XLSX format.
    :type is_xlsx: bool
    :return: The worksheet object.
    :rtype: SheetType
    :raises NetworkTopologyError: If the sheet is not found.
    """
    try:
        sheet = get_sheet(wb, sheet_name, is_xlsx)
        return sheet
    except XLS_EXCEPTIONS as exc:
        msg = f'Error: no {sheet_name} sheet in the file.'
        raise NetworkTopologyError(msg) from exc


def read_excel(input_filename: Path) -> Dict[str, Node]:
    """Reads the 'Nodes' and 'Links' sheets from an Excel file to build a network graph.

    :param input_filename: Path to the Excel file.
    :type input_filename: Path
    :return: Dictionary of nodes with their connectivity and equipment type.
    :rtype: Dict[str, Node]
    """
    wobo, is_xlsx = generic_open_workbook(input_filename)
    links_sheet = open_sheet_with_error_handling(wobo, 'Links', is_xlsx)
    get_rows_links = fast_get_sheet_rows(links_sheet) if is_xlsx else None

    nodes = {}
    for row in all_rows(links_sheet, is_xlsx, start=5, get_rows=get_rows_links):
        node_a, node_z = row[0].value, row[1].value
        # Add connection in both directions
        for node1, node2 in [(node_a, node_z), (node_z, node_a)]:
            if node1 in nodes:
                nodes[node1].to_node.append(node2)
            else:
                nodes[node1] = Node(node1, [node2])

    nodes_sheet = open_sheet_with_error_handling(wobo, 'Nodes', is_xlsx)
    get_rows_nodes = fast_get_sheet_rows(nodes_sheet) if is_xlsx else None

    for row in all_rows(nodes_sheet, is_xlsx, start=5, get_rows=get_rows_nodes):
        node = row[0].value
        eqpt = row[6].value
        if node not in nodes:
            raise NetworkTopologyError(f'Error: node {node} is not listed on the links sheet.')
        if eqpt == 'ILA' and len(nodes[node].to_node) != 2:
            degree = len(nodes[node].to_node)
            raise NetworkTopologyError(f'Error: node {node} has an incompatible node degree ({degree}) '
                                       + 'for its equipment type (ILA).')
        if eqpt == '' and len(nodes[node].to_node) == 2:
            nodes[node].eqpt = 'ILA'
        elif eqpt == '' and len(nodes[node].to_node) != 2:
            nodes[node].eqpt = 'ROADM'
        else:
            nodes[node].eqpt = eqpt
    return nodes


def create_eqpt_template(nodes: Dict[str, Node], input_filename: Path, output_filename: Optional[Path] = None):
    """Creates a CSV template to help users populate equipment types for nodes.

    :param nodes: Dictionary of nodes.
    :type nodes: Dict[str, Node]
    :param input_filename: Path to the original Excel file.
    :type input_filename: Path
    :param output_filename: Path to save the CSV file; generated if None.
    :type output_filename: Optional(Path)
    """
    if output_filename is None:
        output_filename = input_filename.parent / (input_filename.with_suffix('').stem + '_eqpt_sheet.csv')
    with open(output_filename, mode='w', encoding='utf-8', newline='') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        amp_header = ['amp_type', 'att_in', 'amp_gain', 'tilt', 'att_out', 'delta_p']
        output_writer.writerow(['node_a', 'node_z'] + amp_header + amp_header)
        for node in nodes.values():
            if node.eqpt == 'ILA':
                output_writer.writerow([node.uid, node.to_node[0]])
            if node.eqpt == 'ROADM':
                for to_node in node.to_node:
                    output_writer.writerow([node.uid, to_node])
    msg = f'File {output_filename} successfully created.'
    logger.info(msg)


if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    create_eqpt_template(read_excel(ARGS.workbook), ARGS.workbook, ARGS.output)
