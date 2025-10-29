#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_create_eqpt_sheet
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
Checks create_eqpt_sheet.py: verify that output is as expected
"""
from pathlib import Path
from os import symlink, unlink, link
from shutil import copy2

import pytest
from gnpy.tools.create_eqpt_sheet import Node, read_excel, create_eqpt_template
from gnpy.core.exceptions import NetworkTopologyError

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data' / 'create_eqpt_sheet'

TEST_FILE_NO_ERR = DATA_DIR / 'test_ces_no_err.xls'

TEST_FILE = 'testTopology.xls'
EXPECTED_OUTPUT = DATA_DIR / 'testTopology_eqpt_sheet.csv'

TEST_FILE_NODE_DEGREE_ERR = DATA_DIR / 'test_ces_node_degree_err.xls'
TEST_FILE_KEY_ERR = DATA_DIR / 'test_ces_key_err.xls'
TEST_OUTPUT_FILE_CSV = DATA_DIR / 'test_create_eqpt_sheet.csv'
PYTEST_OUTPUT_FILE_NAME = 'test_ces_pytest_output.csv'
EXPECTED_OUTPUT_CSV_NAME = 'testTopology_eqpt_sheet.csv'


@pytest.fixture()
def test_node():
    """Fixture of simple Node."""
    return Node(1, ['A', 'B'], 'ROADM')


@pytest.fixture()
def test_nodes_list():
    """Fixture of nodes list parsing."""
    return read_excel(TEST_FILE_NO_ERR)


def test_node_append(test_node):
    """Test Node's append method."""
    expected = {'uid': 1, 'to_node': ['A', 'B', 'C'], 'eqpt': 'ROADM'}
    test_node.to_node.append('C')
    assert test_node.__dict__ == expected


def test_read_excel(test_nodes_list):
    """Test method read_excel()."""
    expected = {}
    expected['a'] = Node('a', ['b', 'd', 'e'], 'ROADM')
    expected['b'] = Node('b', ['a', 'c'], 'FUSED')
    expected['c'] = Node('c', ['b', 'd', 'e'], 'ROADM')
    expected['d'] = Node('d', ['c', 'a'], 'ILA')
    expected['e'] = Node('e', ['a', 'c'], 'ILA')
    assert set(test_nodes_list) == set(expected)


def test_read_excel_node_degree_err():
    """Test node degree error (eqpt == 'ILA' and len(nodes[node].to_node) != 2)."""
    with pytest.raises(NetworkTopologyError):
        _ = read_excel(TEST_FILE_NODE_DEGREE_ERR)


def test_read_excel_key_err():
    """Test node not listed on the links sheets."""
    with pytest.raises(NetworkTopologyError):
        _ = read_excel(TEST_FILE_KEY_ERR)


def test_create_eqpt_template(tmpdir, test_nodes_list):
    """Test method create_eqt_template()."""
    create_eqpt_template(test_nodes_list, DATA_DIR / TEST_FILE_NO_ERR,
                         tmpdir / PYTEST_OUTPUT_FILE_NAME)
    with open((tmpdir / PYTEST_OUTPUT_FILE_NAME).strpath, 'r') as actual, \
            open(TEST_OUTPUT_FILE_CSV, 'r') as expected:
        assert set(actual.readlines()) == set(expected.readlines())
    unlink(tmpdir / PYTEST_OUTPUT_FILE_NAME)


def test_create_eqpt(tmp_path):
    def safe_link_or_copy(src_path: Path, dst_path: Path) -> None:
        try:
            symlink(src_path, dst_path)
            return
        except (OSError, NotImplementedError):
            pass
        try:
            link(src_path, dst_path)
            return
        except OSError:
            pass
        copy2(src_path, dst_path)

    src = DATA_DIR / TEST_FILE
    dst = tmp_path / TEST_FILE
    safe_link_or_copy(src, dst)
    nodes = read_excel(dst)
    create_eqpt_template(nodes, dst)
    actual = (tmp_path / EXPECTED_OUTPUT_CSV_NAME).read_text().splitlines(True)
    expected = (DATA_DIR / EXPECTED_OUTPUT_CSV_NAME).read_text().splitlines(True)
    assert set(actual) == set(expected)
    (tmp_path / EXPECTED_OUTPUT_CSV_NAME).unlink()
