from json import loads
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('expected_output', type=Path, metavar='FILE')
parser.add_argument('actual_output', type=Path, metavar='FILE')
parser.add_argument('-o', '--output', default=None)

def compare_network_file(expected_output, actual_output):
    with open(expected_output) as f:
        expected_json = f.read()
    with open(actual_output) as f:
        actual_json = f.read()

    expected_data = loads(expected_json)
    actual_data   = loads(actual_json)

    print(f'Comparing:'
          f'\n\t{expected_output}'
          f'\n\t{actual_output}')
    expected_elements = expected_data['elements']
    actual_elements   = actual_data['elements']
    expected_elements = {el['uid']: el for el in expected_elements}
    actual_elements   = {el['uid']: el for el in actual_elements}
    missing   = set(expected_elements) - set(actual_elements)
    extra     = set(actual_elements) - set(expected_elements)
    different = [(expected_elements[x], actual_elements[x]) for
                 x in set(expected_elements) & set(actual_elements)
                 if expected_elements[x] != actual_elements[x]]
    identical = True
    msg = []
    if missing:
        msg.append(f'Missing Elements (by uid): {len(missing)}/{len(expected_elements)}')
        for x in sorted(missing):
            msg.append(f'\t{expected_elements[x]}')
        identical = False
    if extra:
        msg.append(f'Extra Elements (by uid): {len(extra)}/{len(expected_elements)}')
        for x in sorted(extra):
            msg.append(f'\t{actual_elements[x]}')
        identical = False
    if different:
        msg.append(f'Different Elements: {len(different)}/{len(expected_elements)}')
        for x, y in sorted(different, key=lambda xy: xy[0]['uid']):
            msg.append(f'\t- Expected: {x}\n\t  Actual:   {y}')
        identical = False
    if not missing and not extra and not different:
        msg.append('All elements match!')
        identical = True

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
        msg.append(f'Missing Connections: {len(missing)}/{len(expected_connections)}')
        for x in missing:
            msg.append(f'\t{x}')
        identical = False
    if extra:
        msg.append(f'Extra Connections: {len(extra)}/{len(expected_connections)}')
        for x in extra:
            msg.append(f'\t{x}')
        identical = False

    if not missing and not extra:
        msg.append('All connections match!')
    
    return identical,msg

def compare_service_file(expected_output, actual_output):
    with open(expected_output) as f:
        expected_json = f.read()
    with open(actual_output) as f:
        actual_json = f.read()

    expected_data = loads(expected_json)
    actual_data   = loads(actual_json)

    print(f'Comparing:'
          f'\n\t{expected_output}'
          f'\n\t{actual_output}')
    expected_elements = expected_data['path-request']
    actual_elements   = actual_data['path-request']
    expected_elements = {el['request-id']: el for el in expected_elements}
    actual_elements   = {el['request-id']: el for el in actual_elements}
    missing   = set(expected_elements) - set(actual_elements)
    extra     = set(actual_elements) - set(expected_elements)
    different = [(expected_elements[x], actual_elements[x]) for
                 x in set(expected_elements) & set(actual_elements)
                 if expected_elements[x] != actual_elements[x]]
    identical = True
    msg = []
    if missing:
        msg.append(f'Missing requests (by request-id): {len(missing)}/{len(expected_elements)}')
        for x in sorted(missing):
            msg.append(f'\t{expected_elements[x]}')
        identical = False
    if extra:
        msg.append(f'Extra requests (by request-id): {len(extra)}/{len(expected_elements)}')
        for x in sorted(extra):
            msg.append(f'\t{actual_elements[x]}')
        identical = False
    if different:
        msg.append(f'Different requests: {len(different)}/{len(expected_elements)}')
        for x, y in sorted(different, key=lambda xy: xy[0]['request-id']):
            msg.append(f'\t- Expected: {x}\n\t  Actual:   {y}')
        identical = False
    if not missing and not extra and not different:
        msg.append('All elements match!')

    expected_elements = expected_data['synchronisation']
    actual_elements   = actual_data['synchronisation']
    expected_elements = {el['synchonization-id']: el for el in expected_elements}
    actual_elements   = {el['synchonization-id']: el for el in actual_elements}
    missing   = set(expected_elements) - set(actual_elements)
    extra     = set(actual_elements) - set(expected_elements)
    different = [(expected_elements[x], actual_elements[x]) for
                 x in set(expected_elements) & set(actual_elements)
                 if expected_elements[x] != actual_elements[x]]

    if missing:
        msg.append(f'Missing synchronisations (by synchronisation-id): {len(missing)}/{len(expected_elements)}')
        for x in sorted(missing):
            msg.append(f'\t{expected_elements[x]}')
        identical = False
    if extra:
        msg.append(f'Extra synchronisations (by synchronisation-id): {len(extra)}/{len(expected_elements)}')
        for x in sorted(extra):
            msg.append(f'\t{actual_elements[x]}')
        identical = False
    if different:
        msg.append(f'Different synchronisations: {len(different)}/{len(expected_elements)}')
        for x, y in sorted(different, key=lambda xy: xy[0]['synchonization-id']):
            msg.append(f'\t- Expected: {x}\n\t  Actual:   {y}')
        identical = False

    if not missing and not extra and not different:
        msg.append('All synchronisations match!')
    
    return identical,msg

def compare_result_file(expected_output, actual_output):
    with open(expected_output) as f:
        expected_json = f.read()
    with open(actual_output) as f:
        actual_json = f.read()

    expected_data = loads(expected_json)
    actual_data   = loads(actual_json)

    print(f'Comparing:'
          f'\n\t{expected_output}'
          f'\n\t{actual_output}')
    expected_elements = expected_data['path']
    actual_elements   = actual_data['path']
    expected_elements = {el['path-id']: el for el in expected_elements}
    actual_elements   = {el['path-id']: el for el in actual_elements}
    missing   = set(expected_elements) - set(actual_elements)
    extra     = set(actual_elements) - set(expected_elements)
    different = [(expected_elements[x], actual_elements[x]) for
                 x in set(expected_elements) & set(actual_elements)
                 if expected_elements[x] != actual_elements[x]]
    identical = True
    msg = []
    if missing:
        msg.append(f'Missing paths (by path-id): {len(missing)}/{len(expected_elements)}')
        for x in sorted(missing):
            msg.append(f'\t{expected_elements[x]}')
        identical = False
    if extra:
        msg.append(f'Extra paths (by path-id): {len(extra)}/{len(expected_elements)}')
        for x in sorted(extra):
            msg.append(f'\t{actual_elements[x]}')
        identical = False
    if different:
        msg.append(f'Different paths: {len(different)}/{len(expected_elements)}')
        for x, y in sorted(different, key=lambda xy: xy[0]['path-id']):
            msg.append(f'\t- Expected: {x}\n\t  Actual:   {y}')
        identical = False
    if not missing and not extra and not different:
        msg.append('All elements match!')

    return identical,msg


if __name__ == '__main__':
    args = parser.parse_args()
    identical,msg = compare_result_file(args.expected_output, args.actual_output)
    print(identical)
    if args.output:
        with open(args.output,"w") as f:
            f.write('\n'.join(msg))

