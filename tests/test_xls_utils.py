#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# test_xls_utils
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
checks all possibilities of xls_utils.py

"""

from pathlib import Path
from typing import Tuple
from numpy.testing import assert_allclose
import pytest

from gnpy.tools.xls_utils import (
    generic_open_workbook, get_sheet, get_cell_value, get_row, get_row_slice,
    get_num_rows, get_sheet_name, all_rows, get_all_sheets, get_sheet_names,
    WorkbookType, fast_get_sheet_rows
)


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'
XLS_FILE = DATA_DIR / 'testTopology.xls'
XLSX_FILE = DATA_DIR / 'CORONET_Global_Topology.xlsx'


@pytest.mark.parametrize("file_path, is_xlsx", [
    (XLS_FILE, False),
    (XLSX_FILE, True),
])
def test_generic_open_workbook(file_path, is_xlsx):
    """Check that workbooks are correctly opened and type recognized
    """
    _, result_is_xlsx = generic_open_workbook(file_path)
    assert result_is_xlsx == is_xlsx


@pytest.mark.parametrize("file_path", [
    XLS_FILE,
    XLSX_FILE,
])
def test_get_sheet(file_path):
    """Checks that the get_sheet function correctly returns the selected sheet
    """
    workbook, is_xlsx = generic_open_workbook(file_path)
    sheet = get_sheet(workbook, "Nodes", is_xlsx)
    assert get_sheet_name(sheet, is_xlsx) == "Nodes"


@pytest.mark.parametrize("file_path, expected_value", [
    (XLS_FILE, 'Lannion_CAS'),
    (XLSX_FILE, 'Abilene'),
])
def test_get_cell_value(file_path, expected_value):
    """Checks that the get_cell_value function correctly returns the expected value
    """
    workbook, is_xlsx = generic_open_workbook(file_path)
    sheet = get_sheet(workbook, "Nodes", is_xlsx)
    value = get_cell_value(sheet, 5, 0, is_xlsx)
    assert value == expected_value


@pytest.mark.parametrize("file_path", [
    XLS_FILE,
    XLSX_FILE,
])
def test_get_row(file_path):
    """Checks that the get_row function correctly returns the selected row
    """
    workbook, is_xlsx = generic_open_workbook(file_path)
    sheet = get_sheet(workbook, "Nodes", is_xlsx)
    row = get_row(sheet, 4, is_xlsx)
    assert row[0].value == 'City'


def test_get_row_slice_xlsx():
    """Checks that the get_row_slice correctly get the selected slice, also when using the fast_get_sheet_rows closure
    """
    workbook, _ = generic_open_workbook(XLSX_FILE)
    sheet = get_sheet(workbook, "Links", True)
    get_rows = fast_get_sheet_rows(sheet)
    slice_cells = get_row_slice(sheet, 6, 0, 10, True, get_rows)
    print(type(slice_cells))
    assert isinstance(slice_cells, Tuple)
    assert_allclose(slice_cells[2].value, 761.209077632861, rtol=1e-9)
    assert len(slice_cells) == 10


@pytest.mark.parametrize("file_path, expected_count", [
    (XLS_FILE, 24),
    (XLSX_FILE, 105)
])
def test_get_num_rows(file_path, expected_count):
    """Checks that get_num_rows function returns the expected number of rows
    """
    workbook, is_xlsx = generic_open_workbook(file_path)
    sheet = get_sheet(workbook, "Nodes", is_xlsx)
    count = get_num_rows(sheet, is_xlsx)
    assert count == expected_count


def test_all_rows_xlsx():
    """Checks that the all_row function correctly returns all rows when using the closure
    """
    workbook, _ = generic_open_workbook(XLSX_FILE)
    sheet = get_sheet(workbook, "Links", True)
    get_rows = fast_get_sheet_rows(sheet)
    rows_gen = all_rows(sheet, True, get_rows=get_rows)
    rows = list(rows_gen)
    assert len(rows) == 141


@pytest.mark.parametrize("file_path, expected_sheet_names", [
    (XLS_FILE, ["Nodes", "Links", "Eqpt", "Service"]),
    (XLSX_FILE, ["Nodes", "Links"])
])
def test_get_all_sheets(file_path, expected_sheet_names):
    """Checks that the get_all_sheets correctly returns all sheets
    """
    workbook, is_xlsx = generic_open_workbook(file_path)
    sheets = list(get_all_sheets(workbook, is_xlsx))
    assert all((hasattr(s, 'title') and s.title in expected_sheet_names)
               or (hasattr(s, 'name') and s.name in expected_sheet_names) for s in sheets)


@pytest.mark.parametrize("file_path, expected_names", [
    (XLS_FILE, ["Nodes", "Links", "Eqpt", "Service"]),
    (XLSX_FILE, ["Nodes", "Links"])
])
def test_get_sheet_names(file_path, expected_names):
    """Checks get_sheet_names
    """
    workbook, is_xlsx = generic_open_workbook(file_path)
    names = get_sheet_names(workbook, is_xlsx)
    assert names == expected_names
