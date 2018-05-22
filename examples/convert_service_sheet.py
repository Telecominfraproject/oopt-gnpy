#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
# Module name : convert_service_sheet.py
# Version : 
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
@author: jeanluc-auge
xls parser, that can be called to create a json request file in accordance with:
    Yang model for requesting Path Computation
    draft-ietf-teas-yang-path-computation-01.txt. 
 

"""
from sys import exit
try:
    from xlrd import open_workbook, XL_CELL_EMPTY
except ModuleNotFoundError:
    exit('Required: `pip install xlrd`')
from argparse import ArgumentParser
from collections import namedtuple
from logging import getLogger, basicConfig, CRITICAL, DEBUG, INFO
from json import dumps
from pathlib import Path
from gnpy.core.equipment import get_eqpt_params, eqpt_exists, read_eqpt_library

SERVICES_COLUMN = 11
#EQPT_LIBRARY_FILENAME = Path(__file__).parent / 'eqpt_config.json'

all_rows = lambda sheet, start=0: (sheet.row(x) for x in range(start, sheet.nrows))
logger = getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', default='meshTopologyExampleV2.xls')
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-o', '--output', default=None)

# Type for input data
class Request(namedtuple('Request', 'request_id source destination trx_type mode \
    spacing power nb_channel disjoint_from nodes_list is_loose')):
    def __new__(cls, request_id, source, destination, trx_type,  mode , spacing , power , nb_channel , disjoint_from ='' ,  nodes_list = '', is_loose = ''):
        return super().__new__(cls, request_id, source, destination, trx_type, mode, spacing, power, nb_channel, disjoint_from,  nodes_list, is_loose)

# Type for output data:  // from dutc
class Element:
    def __eq__(self, other):
        return type(self) == type(other) and self.uid == other.uid
    def __hash__(self):
        return hash((type(self), self.uid))

class Request_element(Element):
    def __init__(self,Request,eqpt_filename):
        self.request_id = int(Request.request_id)
        self.source = Request.source
        self.destination = Request.destination
        self.srctpid = f'trx {Request.source}'
        self.dsttpid = f'trx {Request.destination}'
        # test that trx_type belongs to eqpt_config.json
        # if not replace it with a default 
        read_eqpt_library(eqpt_filename)
        if eqpt_exists(Request.trx_type):
            self.trx_type = Request.trx_type
            self.mode = Request.mode
        else:
            #TODO : this case must raise an error instead of using Voyager
            self.trx_type = 'Voyager_16QAM'
            print(f'Transceiver type {Request.trx_type} is not defined in {eqpt_filename}')
            print('replaced by Voyager_16QAM')

        if isinstance(Request.disjoint_from,str):
            self.disjoint_from = [int(n) for n in Request.disjoint_from.split()]
        else:
            self.disjoint_from = [int(Request.disjoint_from)]

    uid = property(lambda self: repr(self))
    @property
    def pathrequest(self):
        return {'request-id':self.request_id,
                'source':    self.source,
                'destination':  self.destination,
                'src-tp-id': self.srctpid,
                'dst-tp-id': self.dsttpid,
                'path-constraints':{
                    'te-bandwidth': {
                        'technology': 'flexi-grid',
                        'trx_type'  : self.trx_type,
                        'trx_mode'  : self.mode,
                        'effective-freq-slot':[{'n': 'null','m': 'null'}]
                    }
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

def convert_service_sheet(input_filename, eqpt_filename, filter_region=[]):
    service = parse_excel(input_filename)
    req = [Request_element(n,eqpt_filename) for n in service]
    # todo : maybe dumps the output into a json file ?
    return {
        'path-request': [n.json[0] for n in req],
        'synchronisation': [n.json[1] for n in req 
        if n.json[1] is not None]
    }

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
         'routing: disjoint from', 'routing : path', 'routing : is loose ?']
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

if __name__ == '__main__':
    args = parser.parse_args()
    basicConfig(level={2: DEBUG, 1: INFO, 0: CRITICAL}.get(args.verbose, CRITICAL))
    logger.info(f'Converting Service sheet {args.workbook!r} into gnpy JSON format')
    data = convert_service_sheet(args.workbook,'eqpt_config.json')
    if args.output is None:
        print(dumps(data, indent=2))
    else:
        with open(args.output, 'w') as f:
            f.write(dumps(data, indent=2))