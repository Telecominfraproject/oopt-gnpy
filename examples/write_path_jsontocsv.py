#!/usr/bin/env python3
# TelecomInfraProject/gnpy/examples
# Module name : write_path_jsontoxls.py
# Version : 
# License : BSD 3-Clause Licence
# Copyright (c) 2018, Telecom Infra Project

"""
@author: esther.lerouzic
read json path result file in accordance with:
    Yang model for requesting Path Computation
    draft-ietf-teas-yang-path-computation-01.txt. 
and write results in an CSV file

"""

from argparse import ArgumentParser
from pathlib import Path
from json import loads
from gnpy.core.equipment  import load_equipment
from gnpy.core.request  import jsontocsv


parser = ArgumentParser(description = 'A function that writes json path results in an excel sheet.')
parser.add_argument('filename', nargs='?', type = Path)
parser.add_argument('output_filename', nargs='?', type = Path)
parser.add_argument('eqpt_filename', nargs='?', type = Path, default=Path(__file__).parent / 'eqpt_config.json')

if __name__ == '__main__':
    args = parser.parse_args()
    
    with open(args.output_filename,"w") as file :
        with open(args.filename) as f:
            print(f'Reading {args.filename}')
            json_data = loads(f.read())
            equipment = load_equipment(args.eqpt_filename)
            print(f'Writing in {args.output_filename}')
            jsontocsv(json_data,equipment,file)

