from json import loads
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-e', '--elements', action='store_true', default=False)
parser.add_argument('-c', '--connections', action='store_true', default=False)
parser.add_argument('-u', '--uids-only', action='store_true', default=False)
parser.add_argument('expected_output', type=Path, metavar='FILE')
parser.add_argument('actual_output', type=Path, metavar='FILE')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.expected_output) as f:
        expected_json = f.read()
    with open(args.actual_output) as f:
        actual_json = f.read()

    expected_data = loads(expected_json)
    actual_data   = loads(actual_json)

    print(f'Comparing:'
          f'\n\t{args.expected_output}'
          f'\n\t{args.actual_output}')
    if args.elements:
        expected_elements = expected_data['elements']
        actual_elements   = actual_data['elements']
        expected_elements = {el['uid']: el for el in expected_elements}
        actual_elements   = {el['uid']: el for el in actual_elements}
        missing   = set(expected_elements) - set(actual_elements)
        extra     = set(actual_elements) - set(expected_elements)
        different = [(expected_elements[x], actual_elements[x]) for
                     x in set(expected_elements) & set(actual_elements)
                     if expected_elements[x] != actual_elements[x]]

        if missing:
            print(f'Missing Elements (by uid): {len(missing)}/{len(expected_elements)}')
            for x in sorted(missing):
                print(f'\t{x}' if args.uids_only else f'\t{expected_elements[x]}')
        if extra:
            print(f'Extra Elements (by uid): {len(extra)}/{len(expected_elements)}')
            for x in sorted(extra):
                print(f'\t{x}' if args.uids_only else f'\t{actual_elements[x]}')
        if different:
            print(f'Different Elements: {len(different)}/{len(expected_elements)}')
            for x, y in sorted(different, key=lambda xy: xy[0]['uid']):
                print(f'\t{x["uid"]}' if args.uids_only else f'\t- Expected: {x}\n\t  Actual:   {y}')

        if not missing and not extra and not different:
            print('All elements match!')

    if args.connections:
        expected_connections = expected_data['connections']
        actual_connections   = actual_data['connections']

        missing = []
        for x in sorted(expected_connections, key=lambda d: (d['from_node'], d['to_node'])):
            if x not in actual_connections:
                missing.append(x)
        extra = []
        for x in sorted(actual_connections, key=lambda d: (d['from_node'], d['to_node'])):
            if x not in expected_connections:
                extra.append(x)

        if missing:
            print(f'Missing Connections: {len(missing)}/{len(expected_connections)}')
            for x in missing:
                print(f'\t{x}')
        if extra:
            print(f'Extra Connections: {len(extra)}/{len(expected_connections)}')
            for x in extra:
                print(f'\t{x}')

        if not missing and not extra:
            print('All connections match!')
