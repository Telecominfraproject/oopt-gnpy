#!/usr/bin/env python3
from json import dump
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
from gnpy.tools.json_io import load_json


class Results(namedtuple('Results', 'missing extra different expected actual')):
    def _asdict(self):
        """
        Return a dict representation of this object.

        Args:
            self: (todo): write your description
        """
        return {'missing': self.missing,
                'extra': self.extra,
                'different': self.different}

    def __str__(self):
        """
        Return a string representation of this environment.

        Args:
            self: (todo): write your description
        """
        rv = []
        if self.missing:
            rv.append('Missing: {len(self.missing)}/{len(self.expected)}')
            rv.extend(f'\t{x}' for x in sorted(self.missing))
        if self.extra:
            rv.append('Extra: {len(self.extra)}/{len(self.expected)}')
            rv.extend(f'\t{x}' for x in sorted(self.extra))
        if self.different:
            rv.append('Different: {len(self.different)}/{len(self.expected)}')
            rv.extend(f'\tExpected: {x}\n\tActual:   {y}' for x, y in self.different)
        if not self.missing and not self.extra and not self.different:
            rv.append('All match!')
        return '\n'.join(rv)


class NetworksResults(namedtuple('NetworksResult', 'elements connections')):
    def _asdict(self):
        """
        Return a dictionary as a dict.

        Args:
            self: (todo): write your description
        """
        return {'elements': self.elements._asdict(),
                'connections': self.connections._asdict()}

    def __str__(self):
        """
        Return a string representation of the element.

        Args:
            self: (todo): write your description
        """
        return '\n'.join([
            'Elements'.center(40, '='),
            str(self.elements),
            'Connections'.center(40, '='),
            str(self.connections),
        ])


class ServicesResults(namedtuple('ServicesResult', 'requests synchronizations')):
    def _asdict(self):
        """
        Return a dict representation of the dict.

        Args:
            self: (todo): write your description
        """
        return {'requests': self.requests.asdict(),
                'synchronizations': self.synchronizations.asdict()}

    def __str__(self):
        """
        Returns a string representation of the object.

        Args:
            self: (todo): write your description
        """
        return '\n'.join([
            'Requests'.center(40, '='),
            str(self.requests),
            'Synchronizations'.center(40, '='),
            str(self.synchronizations),
        ])


class PathsResults(namedtuple('PathsResults', 'paths')):
    def _asdict(self):
        """
        Return a dict representation of this object.

        Args:
            self: (todo): write your description
        """
        return {'paths': self.paths.asdict()}

    def __str__(self):
        """
        Str : str of the path

        Args:
            self: (todo): write your description
        """
        return '\n'.join([
            'Paths'.center(40, '='),
            str(self.paths),
        ])


def compare(expected, actual, key=lambda x: x):
    """
    Compares two dictionaries.

    Args:
        expected: (list): write your description
        actual: (todo): write your description
        key: (str): write your description
        x: (todo): write your description
        x: (todo): write your description
    """
    expected = {key(el): el for el in expected}
    actual = {key(el): el for el in actual}
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    different = [(expected[x], actual[x]) for
                 x in set(expected) & set(actual)
                 if expected[x] != actual[x]]
    return Results(missing, extra, different, expected, actual)


def compare_networks(expected, actual):
    """
    Compares two networks of networks.

    Args:
        expected: (todo): write your description
        actual: (float): write your description
    """
    elements = compare(expected['elements'], actual['elements'],
                       key=lambda el: el['uid'])
    connections = compare(expected['connections'], actual['connections'],
                          key=lambda el: (el['from_node'], el['to_node']))
    return NetworksResults(elements, connections)


def compare_services(expected, actual):
    """
    Compares the expected services.

    Args:
        expected: (list): write your description
        actual: (str): write your description
    """
    requests = compare(expected['path-request'], actual['path-request'],
                       key=lambda el: el['request-id'])
    synchronizations = compare(expected['path-request'], expected['path-request'],
                               key=lambda el: el['request-id'])
    if 'synchronization' in expected.keys():
        synchronizations = compare(expected['synchronization'], actual['synchronization'],
                                   key=lambda el: el['synchronization-id'])
    return ServicesResults(requests, synchronizations)


def compare_paths(expected_output, actual_output):
    """
    Compare the expected paths.

    Args:
        expected_output: (str): write your description
        actual_output: (todo): write your description
    """
    paths = compare(expected['path'], actual['path'], key=lambda el: el['path-id'])
    return PathsResults(paths)


COMPARISONS = {
    'networks': compare_networks,
    'services': compare_services,
    'paths': compare_paths,
}

parser = ArgumentParser()
parser.add_argument('expected_output', type=Path, metavar='FILE')
parser.add_argument('actual_output', type=Path, metavar='FILE')
parser.add_argument('-o', '--output', default=None)
parser.add_argument('-c', '--comparison', choices=COMPARISONS, default='networks')


def encode_sets(obj):
    """
    Encodes a list into a list.

    Args:
        obj: (todo): write your description
    """
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f'{obj!r} is not JSON serializable!')


if __name__ == '__main__':
    args = parser.parse_args()
    expected = load_json(args.expected_output)
    actual = load_json(args.actual_output)

    result = COMPARISONS[args.comparison](expected, actual)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            dump(result, f, default=encode_sets, indent=2, ensure_ascii=False)
    else:
        print(str(result))
