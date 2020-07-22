import pytest
from gnpy.tools.create_eqpt_sheet import Node, \
                                         read_excel, \
                                         create_eqpt_template
from pathlib import Path

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / 'data'

TEST_FILE_NO_ERR = f'{DATA_DIR}/test_ces_no_err.xls'
TEST_FILE_NODE_DEGREE_ERR = f'{DATA_DIR}/test_ces_node_degree_err.xls'
TEST_FILE_KEY_ERR = f'{DATA_DIR}/test_ces_key_err.xls'
TEST_OUTPUT_FILE_CSV = f'{DATA_DIR}/test_create_eqpt_sheet.csv'
PYTEST_OUTPUT_FILE_NAME = 'test_ces_pytest_output.csv'
PYTEST_OUTPUT_FILE_CSV = 'test_ces_pytest_output_eqpt_sheet.csv'


@pytest.fixture()
def test_node():
    """Fixture of simple Node."""
    return Node(1, ['A', 'B'], 'ROADM')


@pytest.fixture()
def test_nodes_list():
    """Fixture of nodes list parsing."""
    return read_excel(TEST_FILE_NO_ERR)


def test_node_repr(test_node):
    """Test node representation."""
    expected = {'uid': 1, 'to_node': ['A', 'B'], 'eqpt': 'ROADM'}
    tn_repr = test_node.__repr__()
    assert tn_repr == expected


def test_node_str(test_node):
    """Test Node string."""
    expected = "Node(uid=1, to_node=['A', 'B'], eqpt=ROADM)"
    tn_str = test_node.__str__()
    assert tn_str == expected


def test_node_append(test_node):
    """Test Node's append method."""
    expected = {'uid': 1, 'to_node': ['A', 'B', 'C'], 'eqpt': 'ROADM'}
    test_node.to_node.append('C')
    tn_after = test_node.__repr__()
    assert tn_after == expected


def test_read_excel(test_nodes_list):
    """Test method read_excel()."""
    expected = {}
    expected['a'] = Node('a', ['b', 'd', 'e'], 'ROADM')
    expected['b'] = Node('b', ['a', 'c'], 'FUSED')
    expected['c'] = Node('c', ['b', 'd', 'e'], 'ROADM')
    expected['d'] = Node('d', ['c', 'a'], 'ILA')
    expected['e'] = Node('e', ['a', 'c'], 'ILA')
    assert set(test_nodes_list) == set(expected)

@pytest.mark.parametrize('xls_err_file, expected',
                         {TEST_FILE_NODE_DEGREE_ERR: {},
                          TEST_FILE_KEY_ERR: {}}.items())
def test_read_excel_node_degree_and_key_err(xls_err_file, expected):
    """Test node degree error (node with incompatile node degree)
    and key error (node not listed on links sheet).
    """
    err_result = read_excel(xls_err_file)
    assert set(err_result) == set(expected)


def test_create_eqpt_template(tmpdir, test_nodes_list):
    """Test method create_eqt_template()."""
    create_eqpt_template(test_nodes_list, 
                         (tmpdir / PYTEST_OUTPUT_FILE_NAME).strpath)
    with open((tmpdir / PYTEST_OUTPUT_FILE_CSV).strpath, 'r') as f1, \
         open(TEST_OUTPUT_FILE_CSV, 'r') as f2:
        output = f1.readlines()
        expected = f2.readlines()
    assert set(output) == set(expected)
