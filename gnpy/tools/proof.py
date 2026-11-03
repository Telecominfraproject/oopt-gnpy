# -*- coding: utf-8 -*-

# SPDX-License-Identifier: BSD-3-Clause
# gnpy.tools.proof: pure checks of equipment libraries (no logging, no I/O)
# Copyright (C) 2025 Telecom Infra Project and GNPy contributors
# see AUTHORS.rst for a list of contributors

"""Generation of the proof of validation and of the certified copy of a library."""

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SCHEMA_VERSION = '1.0'
HASH_LENGTH = 12


def glv_version() -> str:
    """Return the installed GNPy version."""
    try:
        return metadata.version('gnpy')
    except metadata.PackageNotFoundError:
        return 'unknown'


def write_proof(raw: bytes, original_name: str, global_score: float, profile: dict,
                organization: str, contact: str, out_dir: Path) -> tuple[Path, Path]:
    """Write the byte-exact certified copy and the proof of validation.

    :return: Paths of the certified library and of the proof.
    """
    digest = hashlib.sha256(raw).hexdigest()
    short = digest[:HASH_LENGTH]
    original = Path(original_name)
    certified_name = f'{original.stem}.{short}{original.suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)

    library_path = out_dir / certified_name
    library_path.write_bytes(raw)

    proof = {
        'schema_version': SCHEMA_VERSION,
        'organization': organization,
        'contact_email': contact,
        'library': {
            'original_filename': original_name,
            'certified_filename': certified_name,
            'sha256': digest,
        },
        'validation': {
            'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'glv_version': glv_version(),
            'profile': profile,
            'score': round(global_score, 1),
        },
    }
    proof_path = out_dir / f'{original.stem}.{short}.proof.json'
    proof_path.write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    return library_path, proof_path
