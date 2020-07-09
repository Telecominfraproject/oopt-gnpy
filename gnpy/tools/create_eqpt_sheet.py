#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_eqpt_sheet.py
====================

XLS parser that can be called to create a "City" column in the "Eqpt" sheet.

If not present in the "Nodes" sheet, the "Type" column will be implicitly
determined based on the topology.
"""

from argparse import ArgumentParser
from xlrd import open_workbook
import csv

EXAMPLE_DATA_DIR = '../example_data/'

PARSER = ArgumentParser()
PARSER.add_argument('workbook', nargs='?', default=f'{EXAMPLE_DATA_DIR}meshTopologyExampleV2.xls',
                    help='create the mandatory columns in Eqpt sheet')


def ALL_ROWS(sheet, start=0):
    return (sheet.row(x) for x in range(start, sheet.nrows))


class Node:
    """Node element contains uid, list of connected nodes and eqpt type."""
    def __init__(self, uid, to_node, eqpt=None):
        self.uid = uid
        self.to_node = to_node
        self.eqpt = eqpt

    def __repr__(self):
        return f'Node(uid={self.uid}, to_node={self.to_node}, eqpt={self.eqpt})'

    def __str__(self):
        return {'uid': self.uid, 'to_node': self.to_node, 'eqpt': self.eqpt}
        

def read_excel(input_filename):
    """Read excel Nodes and Links sheets and create a dict of nodes with
    their 'to_node' and 'eqpt'.
    """
    with open_workbook(input_filename) as wobo:
        try:
            links_sheet = wobo.sheet_by_name('Links')
        except Exception:
            print(f'Error: no Links sheet on file {input_filename}.')
            exit()
        
        nodes = {}
        for row in ALL_ROWS(links_sheet, start=5):
            try:
                nodes[row[0].value].to_node.append(row[1].value)
            except KeyError:
                nodes[row[0].value] = Node(row[0].value, [row[1].value])
            try:
                nodes[row[1].value].to_node.append(row[0].value)
            except KeyError:
                nodes[row[1].value] = Node(row[1].value, [row[0].value])

        try:
            nodes_sheet = wobo.sheet_by_name('Nodes')
        except Exception:
            print(f'Error: no Nodes sheet on file {input_filename}.')
            exit()
        for row in ALL_ROWS(nodes_sheet, start=5):
            node = row[0].value
            eqpt = row[6].value
            try:
                if eqpt == 'ILA' and len(nodes[node].to_node) != 2:
                    print(f'Error: node {node} has an incompatible node degree ({len(nodes[node].to_node)}) for its equipment type (ILA).')
                    nodes = {}
                    break
                if eqpt == '' and len(nodes[node].to_node) == 2:
                    nodes[node].eqpt = 'ILA'
                elif eqpt == '' and len(nodes[node].to_node) != 2:
                    nodes[node].eqpt = 'ROADM'
                else:
                    nodes[node].eqpt = eqpt
            except KeyError:
                print(f'Error: node {node} is not listed on the links sheet.')
                nodes = {}
                break
        
        return nodes


def create_eqpt_template(nodes, input_filename):
    """Write list of node A node Z corresponding to Nodes and Links sheets in order
    to help user populate Eqpt.
    """
    if not nodes:
        exit()
    output_filename = f'{input_filename[:-4]}_eqpt_sheet.csv'
    with open(output_filename, mode='w', encoding='utf-8') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['node_a', 'node_z', 'amp_type', 'att_in', 'amp_gain', 'tilt', 'att_out', 'amp_type', 'att_in', 'amp_gain', 'tilt', 'att_out'])
        for node in nodes.values():
            if node.eqpt == 'ILA':
                output_writer.writerow([node.uid, node.to_node[0]])
            if node.eqpt == 'ROADM':
                for to_node in node.to_node:
                    output_writer.writerow([node.uid, to_node])
    print(f'File {output_filename} successfully created.')


if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    create_eqt_template(read_excel(ARGS.workbook), ARGS.workbook)
