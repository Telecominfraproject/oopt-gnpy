#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gnpy.core.service_sheet
========================

XLS parser that can be called to create a JSON request file in accordance with
Yang model for requesting path computation.

See: draft-ietf-teas-yang-path-computation-01.txt
"""

from sys import exit
try:
    from xlrd import open_workbook, XL_CELL_EMPTY
except ModuleNotFoundError:
    exit('Required: `pip install xlrd`')
from collections import namedtuple
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps
from pathlib import Path
from gnpy.core.equipment import load_equipment
from gnpy.core.utils import db2lin, lin2db

SERVICES_COLUMN = 11
#EQPT_LIBRARY_FILENAME = Path(__file__).parent / 'eqpt_config.json'

all_rows = lambda sheet, start=0: (sheet.row(x) for x in range(start, sheet.nrows))
logger = getLogger(__name__)

# Type for input data
class Request(namedtuple('Request', 'request_id source destination trx_type mode \
    spacing power nb_channel disjoint_from nodes_list is_loose')):
    def __new__(cls, request_id, source, destination, trx_type,  mode , spacing , power , nb_channel , disjoint_from ='' ,  nodes_list = None, is_loose = ''):
        return super().__new__(cls, request_id, source, destination, trx_type, mode, spacing, power, nb_channel, disjoint_from,  nodes_list, is_loose)

# Type for output data:  // from dutc
class Element:
    def __eq__(self, other):
        return type(self) == type(other) and self.uid == other.uid
    def __hash__(self):
        return hash((type(self), self.uid))

class Request_element(Element):
    def __init__(self,Request,eqpt_filename):
        # request_id is str
        # excel has automatic number formatting that adds .0 on integer values
        # the next lines recover the pure int value, assuming this .0 is unwanted
        if not isinstance(Request.request_id,str):
            value = str(int(Request.request_id))
            if value.endswith('.0'):
                value = value[:-2]
            self.request_id = value
        else:
            self.request_id = Request.request_id
        self.source = Request.source
        self.destination = Request.destination
        self.srctpid = f'trx {Request.source}'
        self.dsttpid = f'trx {Request.destination}'
        # test that trx_type belongs to eqpt_config.json
        # if not replace it with a default
        equipment = load_equipment(eqpt_filename)
        try :
            if equipment['Transceiver'][Request.trx_type]:
                self.trx_type = Request.trx_type
            if [mode for mode in equipment['Transceiver'][Request.trx_type].mode]:
                self.mode = Request.mode
        except KeyError:
            msg = f'could not find tsp : {Request.trx_type} with mode: {Request.mode} in eqpt library \nComputation stopped.'
            #print(msg)
            logger.critical(msg)
            exit()
        # excel input are in GHz and dBm
        self.spacing = Request.spacing * 1e9
        self.power =  db2lin(Request.power) * 1e-3
        self.nb_channel = int(Request.nb_channel)
        if not isinstance(Request.disjoint_from,str):
            value = str(int(Request.disjoint_from))
            if value.endswith('.0'):
                value = value[:-2]
        else:
            value = Request.disjoint_from
        self.disjoint_from = [n for n in value.split()]
        self.nodes_list = []
        if Request.nodes_list :
            self.nodes_list = Request.nodes_list.split(' | ')
        try :
            self.nodes_list.remove(self.source)
            msg = f'{self.source} removed from explicit path node-list'
            logger.info(msg)
            # print(msg)
        except ValueError:
            msg = f'{self.source} already removed from explicit path node-list'
            logger.info(msg)
            # print(msg)
        try :
            self.nodes_list.remove(self.destination)
            msg = f'{self.destination} removed from explicit path node-list'
            logger.info(msg)
            # print(msg)
        except ValueError:
            msg = f'{self.destination} already removed from explicit path node-list'
            logger.info(msg)
            # print(msg)

        self.loose = 'loose'
        if Request.is_loose == 'no' :
            self.loose = 'strict'

    uid = property(lambda self: repr(self))
    @property
    def pathrequest(self):
        return {
                    'request-id':self.request_id,
                    'source':    self.source,
                    'destination':  self.destination,
                    'src-tp-id': self.srctpid,
                    'dst-tp-id': self.dsttpid,
                    'path-constraints':{
                        'te-bandwidth': {
                            'technology': 'flexi-grid',
                            'trx_type'  : self.trx_type,
                            'trx_mode'  : self.mode,
                            'effective-freq-slot':[{'n': 'null','m': 'null'}] ,
                            'spacing'   : self.spacing,
                            'max-nb-of-channel'  : self.nb_channel,
                            'output-power'       : self.power
                        }
                    },
                    'optimizations': {
                        'explicit-route-include-objects': [
                        {
                            'index': self.nodes_list.index(node),
                            'unnumbered-hop':{
                                'node-id': f'{node}',
                                'link-tp-id': 'link-tp-id is not used',
                                'hop-type': 'loose',
                                'direction': 'direction is not used'
                            },
                            'label-hop':{
                                'te-label': {
                                    'generic': 'generic is not used',
                                    'direction': 'direction is not used'
                                }
                            }
                        }
                        for node in self.nodes_list
                    ]

                }
            }
    @property
    def pathsync(self):
        if self.disjoint_from :
            return {'synchonization-id':self.request_id,
                'svec': {
                    'relaxable' : 'False',
                    'link-diverse': 'True',
                    'node-diverse': 'True',
                    'request-id-number': [self.request_id]+ [n for n in self.disjoint_from]
                }
            }
        # TO-DO: avoid multiple entries with same synchronisation vectors
    @property
    def json(self):
        return self.pathrequest , self.pathsync

def convert_service_sheet(input_filename, eqpt_filename, output_filename='', filter_region=[]):
    service = parse_excel(input_filename)
    req = [Request_element(n,eqpt_filename) for n in service]
    # dumps the output into a json file with name
    # split_filename = [input_filename[0:len(input_filename)-len(suffix_filename)] , suffix_filename[1:]]
    if output_filename=='':
        output_filename = f'{str(input_filename)[0:len(str(input_filename))-len(str(input_filename.suffixes[0]))]}_services.json'
    # for debug
    # print(json_filename)
    data = {
        'path-request': [n.json[0] for n in req],
        'synchronisation': [n.json[1] for n in req
        if n.json[1] is not None]
    }
    with open(output_filename, 'w') as f:
            f.write(dumps(data, indent=2))
    return data

# to be used from dutc
def parse_row(row, fieldnames):
    return {f: r.value for f, r in zip(fieldnames, row[0:SERVICES_COLUMN])
            if r.ctype != XL_CELL_EMPTY}
#

def parse_excel(input_filename):
    with open_workbook(input_filename) as wb:
        service_sheet = wb.sheet_by_name('Service')
        services = list(parse_service_sheet(service_sheet))
    return services

def parse_service_sheet(service_sheet):
        logger.info(f'Validating headers on {service_sheet.name!r}')
        header = [x.value.strip() for x in service_sheet.row(4)[0:SERVICES_COLUMN]]
        expected = ['route id', 'Source', 'Destination', 'TRX type', \
         'Mode', 'System: spacing', 'System: input power (dBm)', 'System: nb of channels',\
         'routing: disjoint from', 'routing: path', 'routing: is loose?']
        if header != expected:
            msg = f'Malformed header on Service sheet: {header} != {expected}'
            logger.critical(msg)
            raise ValueError(msg)

        service_fieldnames = 'request_id source destination trx_type mode spacing power nb_channel disjoint_from nodes_list is_loose'.split()
        # Important Note: it reads all colum on each row so that
        # it is not possible to write annotation in the excel sheet
        # outside the SERVICES_COLUMN ...  TO BE IMPROVED
        # request_id should be unique for disjunction constraints (not used yet)
        for row in all_rows(service_sheet, start=5):
            yield Request(**parse_row(row[0:SERVICES_COLUMN], service_fieldnames))
