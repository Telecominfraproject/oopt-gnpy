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

from sys import exit
from csv import writer
from argparse import ArgumentParser
from pathlib import Path
from json import dumps, loads
from gnpy.core.equipment  import load_equipment
from gnpy.core.utils import lin2db

START_LINE = 5


parser = ArgumentParser(description = 'A function that writes json path results in an excel sheet.')
parser.add_argument('filename', nargs='?', type = Path)
parser.add_argument('eqpt_filename', nargs='?', type = Path)

parser.add_argument('output_filename', nargs='?', type = Path)

if __name__ == '__main__':
    args = parser.parse_args()
    print(f'coucou {args.output_filename}')
    with open(args.output_filename,"w") as file :
        mywriter = writer(file)
        mywriter.writerow(('path-id','source','destination','transponder-type',\
            'transponder-mode','baud rate (Gbaud)', 'input power (dBm)','path','OSNR@bandwidth','OSNR@0.1nm','SNR@bandwidth','SNR@0.1nm','Pass?'))

        with open(args.filename) as f:
            json_data = loads(f.read())
            equipment = load_equipment(args.eqpt_filename)
            tspjsondata = equipment['Transceiver']
            #print(tspjsondata)
            for p in json_data['path']:
                path_id     = int(p['path-id'])
                source      = p['path-properties']['path-route-objects'][0]\
                ['path-route-object']['unnumbered-hop']['node-id']
                destination = p['path-properties']['path-route-objects'][-1]\
                ['path-route-object']['unnumbered-hop']['node-id']
                pth        = ' | '.join([ e['path-route-object']['unnumbered-hop']['node-id'] 
                         for e in p['path-properties']['path-route-objects']])

                [tsp,mode] = p['path-properties']['path-route-objects'][0]\
                ['path-route-object']['unnumbered-hop']['hop-type'].split(' - ')
                
                # find the min  acceptable OSNR, baud rate from the eqpt library based on tsp (tupe) and mode (format)
                try:
                    [minosnr, baud_rate] = next([m['OSNR'] , m['baud_rate']]  
                        for m in equipment['Transceiver'][tsp].mode if  m['format']==mode)
    
                # for debug
                # print(f'coucou {baud_rate}')
                except IndexError:
                    msg = f'could not find tsp : {self.tsp} with mode: {self.tsp_mode} in eqpt library'
                    
                    raise ValueError(msg)
                output_snr = next(e['accumulative-value'] 
                    for e in p['path-properties']['path-metric'] if e['metric-type'] == 'SNR@0.1nm')
                output_snrbandwidth = next(e['accumulative-value'] 
                    for e in p['path-properties']['path-metric'] if e['metric-type'] == 'SNR@bandwidth')
                output_osnr = next(e['accumulative-value'] 
                    for e in p['path-properties']['path-metric'] if e['metric-type'] == 'OSNR@0.1nm')
                output_osnrbandwidth = next(e['accumulative-value'] 
                    for e in p['path-properties']['path-metric'] if e['metric-type'] == 'OSNR@bandwidth')
                power = next(e['accumulative-value'] 
                    for e in p['path-properties']['path-metric'] if e['metric-type'] == 'reference_power')
                mywriter.writerow((path_id,
                    source,
                    destination,
                    tsp,
                    mode,
                    baud_rate*1e-9,
                    round(lin2db(power)+30,2),
                    pth,
                    output_osnrbandwidth,
                    output_osnr,
                    output_snrbandwidth,
                    output_snr,
                    output_snr >= minosnr
                    ))
