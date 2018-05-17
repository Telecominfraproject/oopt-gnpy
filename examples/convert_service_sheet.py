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

SERVICES_COLUMN = 5
all_rows = lambda sheet, start=0: (sheet.row(x) for x in range(start, sheet.nrows))
logger = getLogger(__name__)

parser = ArgumentParser()
parser.add_argument('workbook', nargs='?', default='meshTopologyExampleV2.xls')
parser.add_argument('-v', '--verbose', action='count')
parser.add_argument('-o', '--output', default=None)

# Type for input data
class Request(namedtuple('Request', 'request_id source destination trx_type disjoint_from')):
    def __new__(cls, request_id, source, destination, trx_type, disjoint_from = ''):
        return super().__new__(cls, request_id, source, destination, trx_type, disjoint_from)

# Type for output data:  // from dutc
class Element:
    def __eq__(self, other):
        return type(self) == type(other) and self.uid == other.uid
    def __hash__(self):
        return hash((type(self), self.uid))

class Request_element(Element):
    def __init__(self,Request):
        self.request_id = int(Request.request_id)
        self.source = Request.source
        self.destination = Request.destination
        self.srctpid = f'trx {Request.source}'
        self.dsttpid = f'trx {Request.destination}'
        self.trx_type = Request.trx_type
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

def convert_service_sheet(input_filename, filter_region=[]):
    service = parse_excel(input_filename)

    return {
        'path-request': [Request_element(n).json[0] for n in service],
        'synchronisation': [Request_element(n).json[1] for n in service if Request_element(n).json[1] is not None]
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
        expected = ['route id', 'Source', 'Destination', 'TRX type', 'routing: disjoint from']
        if header != expected:
            msg = f'Malformed header on Service sheet: {header} != {expected}'
            logger.critical(msg)
            raise ValueError(msg)

        service_fieldnames = 'request_id source destination trx_type disjoint_from'.split()
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
    data = convert_service_sheet(args.workbook)
    if args.output is None:
        print(dumps(data, indent=2))
    else:
        with open(args.output, 'w') as f:
            f.write(dumps(data, indent=2))