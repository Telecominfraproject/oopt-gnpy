# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# testing commits compliancy
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

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
