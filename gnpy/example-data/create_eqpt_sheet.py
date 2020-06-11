#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_eqpt_sheet.py
====================

XLS parser that can be called to create a "City" column in the "Eqpt" sheet.

If not present in the "Nodes" sheet, the "Type" column will be implicitly
determined based on the topology.
"""

from xlrd import open_workbook
from argparse import ArgumentParser

PARSER = ArgumentParser()
PARSER.add_argument('workbook', nargs='?', default='meshTopologyExampleV2.xls',
                    help='create the mandatory columns in Eqpt sheet')


def ALL_ROWS(sh, start=0):
    return (sh.row(x) for x in range(start, sh.nrows))


class Node:
    """ Node element contains uid, list of connected nodes and eqpt type
    """

    def __init__(self, uid, to_node):
        self.uid = uid
        self.to_node = to_node
        self.eqpt = None

    def __repr__(self):
        return f'uid {self.uid} \nto_node {[node for node in self.to_node]}\neqpt {self.eqpt}\n'

    def __str__(self):
        return f'uid {self.uid} \nto_node {[node for node in self.to_node]}\neqpt {self.eqpt}\n'


def read_excel(input_filename):
    """ read excel Nodes and Links sheets and create a dict of nodes with
    their to_nodes and type of eqpt
    """
    with open_workbook(input_filename) as wobo:
        # reading Links sheet
        links_sheet = wobo.sheet_by_name('Links')
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

        nodes_sheet = wobo.sheet_by_name('Nodes')
        for row in ALL_ROWS(nodes_sheet, start=5):
            node = row[0].value
            eqpt = row[6].value
            try:
                if eqpt == 'ILA' and len(nodes[node].to_node) != 2:
                    print(f'Inconsistancy ILA node with degree > 2: {node} ')
                    exit()
                if eqpt == '' and len(nodes[node].to_node) == 2:
                    nodes[node].eqpt = 'ILA'
                elif eqpt == '' and len(nodes[node].to_node) != 2:
                    nodes[node].eqpt = 'ROADM'
                else:
                    nodes[node].eqpt = eqpt
            except KeyError:
                print(f'inconsistancy between nodes and links sheet: {node} is not listed in links')
                exit()
        return nodes


def create_eqt_template(nodes, input_filename):
    """ writes list of node A node Z corresponding to Nodes and Links sheets in order
    to help user populating Eqpt
    """
    output_filename = f'{input_filename[:-4]}_eqpt_sheet.txt'
    with open(output_filename, 'w', encoding='utf-8') as my_file:
        # print header similar to excel
        my_file.write('OPTIONAL\n\n\n\
           \t\tNode a egress amp (from a to z)\t\t\t\t\tNode a ingress amp (from z to a) \
           \nNode A \tNode Z \tamp type \tatt_in \tamp gain \ttilt \tatt_out\
           amp type   \tatt_in \tamp gain   \ttilt   \tatt_out\n')

        for node in nodes.values():
            if node.eqpt == 'ILA':
                my_file.write(f'{node.uid}\t{node.to_node[0]}\n')
            if node.eqpt == 'ROADM':
                for to_node in node.to_node:
                    my_file.write(f'{node.uid}\t{to_node}\n')

        print(f'File {output_filename} successfully created with Node A - Node Z entries for Eqpt sheet in excel file.')


if __name__ == '__main__':
    ARGS = PARSER.parse_args()
    create_eqt_template(read_excel(ARGS.workbook), ARGS.workbook)
