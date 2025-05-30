#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# testing commits compliancy
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""
checks compliance to opensource rules:
- list of authors
- presence of headers
"""
import subprocess
import re
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent


def get_author_emails_from_file():
    # Read AUTHOR.rst content
    with open(SRC_DIR / 'AUTHORS.rst', 'r', encoding='utf-8') as f:
        author_rst = f.read()

    # Extract all author emails from AUTHOR.rst
    return set(re.findall(r'<([^>]+)>', author_rst))


def get_commit_author_emails(since_days=None):
    # Get list of commit authors, optionally limited to recent commits
    cmd = ['git', 'log', '--pretty=format:%aE']

    if since_days:
        cmd.append(f'--since={since_days} days ago')

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Return unique emails from git log
    return set(result.stdout.splitlines())


def test_commit_authors_in_author_rst():
    author_emails = get_author_emails_from_file()
    commit_emails = get_commit_author_emails(since_days=365)
    allowed_missing = {'andrea.damico@polito.it'
                       }

    # Check that each commit author email is listed in AUTHOR.rst
    missing_emails = commit_emails - author_emails - allowed_missing

    assert not missing_emails, f"The following commit author emails are missing in AUTHOR.rst: {missing_emails}"


def get_python_files(root_dir):
    """Get all Python files in the project."""
    return list(root_dir.glob('**/*.py'))


def check_header(file_path):
    """Check if a file has the correct header."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(500)  # Read just the beginning of the file

    # Check for shebang
    has_shebang = content.startswith('#!/usr/bin/env python')

    # Check for encoding
    has_encoding = '# -*- coding: utf-8 -*-' in content[:100]

    # Check for SPDX license
    has_license = 'SPDX-License-Identifier: BSD-3-Clause' in content

    # Check for copyright
    has_copyright = re.search(r'Copyright \(C\) .*Telecom Infra Project', content) is not None

    # Check for AUTHORS reference
    has_authors_ref = 'see AUTHORS.rst' in content

    return {
        'file': file_path,
        'has_shebang': has_shebang,
        'has_encoding': has_encoding,
        'has_license': has_license,
        'has_copyright': has_copyright,
        'has_authors_ref': has_authors_ref,
        'is_compliant': has_shebang and has_encoding and has_license and has_copyright and has_authors_ref
    }


def test_file_headers():
    """Test that all Python files have the correct header."""
    # Get all Python files
    python_files = get_python_files(SRC_DIR / 'gnpy') + get_python_files(SRC_DIR / 'tests')

    # Files that can be excluded from header checks (e.g., third-party code, generated files)
    excluded_files = [
        SRC_DIR / 'gnpy' / 'topology' / '__init__.py',
        SRC_DIR / 'gnpy' / 'tools' / '__init__.py',
        SRC_DIR / 'gnpy' / 'core' / '__init__.py',
        SRC_DIR / 'gnpy' / '__init__.py',
        SRC_DIR / 'tests' / '__init__.py'
    ]

    # Filter out excluded files
    python_files = [f for f in python_files if f not in excluded_files]

    # Check headers
    results = [check_header(f) for f in python_files]
    non_compliant = [r for r in results if not r['is_compliant']]

    # Generate detailed error message
    error_msg = ""
    if non_compliant:
        error_msg = "The following files have non-compliant headers:\n"
        for result in non_compliant:
            file_path = result['file'].relative_to(SRC_DIR)
            error_msg += f"\n{file_path}:\n"
    print(non_compliant)
    assert len(non_compliant) == 0, error_msg
