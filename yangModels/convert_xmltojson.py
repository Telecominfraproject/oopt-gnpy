

from argparse import ArgumentParser
from pathlib import Path
from json import dumps, loads

import xmltodict

parser = ArgumentParser(description = 'A function that converts xml into json instance.')
parser.add_argument('xml_filename', nargs='?', type = Path)
parser.add_argument('-o', '--output')
 
args = parser.parse_args()
if args.output:
    json_filename =  args.output
else:
    if args.xml_filename.suffix.lower() == '.xml':
        json_filename =  f'{str(args.xml_filename)[0:len(str(args.xml_filename))-3]}json'
    else:
        json_filename =  f'{str(args.xml_filename)}.json'


with open(args.xml_filename, 'r') as f:
    xmlString = f.read()

jsonString = loads(dumps(xmltodict.parse(xmlString), indent=2))


print("\nJSON output(output.json):")

print(dumps(jsonString['data']))

 

with open(json_filename, 'w') as f:

    f.write(dumps(jsonString['data'], indent=2))