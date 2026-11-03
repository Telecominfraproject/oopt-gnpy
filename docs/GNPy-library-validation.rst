GNPy Library Validator (GLV): User Guide
========================================

1. Purpose
----------

GLV checks an equipment library (JSON) used by GNPy and reports its quality.
It can be used to:

* check that a library is valid before sharing or using it;
* find typos and inconsistent values (OSNR, offsets, amplifier bands, ROADM impairments);
* measure how complete the data is;
* obtain a proof of validation for a library that meets the quality criteria.

2. What is checked
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Axis
     - Meaning
     - Examples
   * - Format
     - The library complies with the YANG model ``gnpy-eqpt-config`` and is
       encoded as RFC7951.
     - Value out of range, ``nf_min`` > ``nf_max``, unknown amplifier reference
   * - Consistency
     - Values are coherent with each other.
     - OSNR very different from similar modes, multiband amplifier covering
       one band, ROADM PMD too high
   * - Completeness
     - Useful data is present although GNPy tolerates its absence.
     - Missing penalties, missing frequency range, missing ROADM path impairments

The complete list of checks, with rules, thresholds and penalties, is given in
the check reference table (TBD).

3. Running the tool
-------------------

Validate a library::

    python validate_library.py equipment.json

Options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-v``, ``-vv``
     - Increase verbosity (info, debug).
   * - ``--json``
     - Print a JSON summary (scores and issues) on standard output.
   * - ``--min-score VALUE``
     - Exit with code 1 if the global score is below ``VALUE``. Useful in CI.
   * - ``--proof-dir DIR``
     - Write the certified copy and the proof of validation in ``DIR``.
   * - ``--organization NAME``
     - Organization name (required with ``--proof-dir``).
   * - ``--contact EMAIL``
     - Contact email (required with ``--proof-dir``).
   * - ``--min-completeness VALUE``
     - Minimum completeness for a proof (default: 75).
   * - ``--min-consistency VALUE``
     - Minimum consistency for a proof (default: 98).

The output contains:

* one warning per issue, with its code, the concerned item (section,
  ``type_variety``, mode) and a message;
* the global score, the format score and the scores per section
  (Transceiver, Edfa, Roadm), split by axis;
* the SHA-256 of the input file.

Exit codes: 0 on success, 1 if the score is below ``--min-score`` or if no proof
can be issued when requested, 2 if the file is not valid JSON.

Typical workflow:

#. Run the tool on the library.
#. Fix format issues first. If the format is invalid, no other check is run
   and the score is 0.
#. Fix blocking consistency issues (see section 5).
#. Review the other consistency and completeness issues.
#. Run again until the result is satisfactory.

4. Reading the score
--------------------

* Score of an axis in a section = 100 × (1 − sum of issue penalties / number of checks),
  with a minimum of 0.
* Section score = weighted combination of consistency (0.4) and completeness (0.2),
  normalized.
* Global score = 0.4 × format + 0.6 × weighted average of the sections
  (Transceiver 0.4, Edfa 0.2, Roadm 0.2).
* An axis or section with no applicable check is not counted.

A score below 100 is not necessarily a problem: some issues have a low penalty
and only signal a point to review.

5. Blocking issues
------------------

The following issues force the global score to 0, whatever the rest of the
library contains:

* ``BIT_RATE_INVALID``
* ``INVALID_PENALTY_VALUE_NOT_MONOTONIC``
* ``INVALID_NF_RANGE_NOT_MATCHING_GAIN_RANGE``
* ``INVALID_MULTIBAND_SINGLE_BAND``

Any YANG error also gives a format score of 0, and therefore a global score of 0.

6. Proof of validation
----------------------

A proof can be issued when the library is eligible:

* non-zero global score (no blocking issue, valid format);
* aggregated consistency at or above the threshold (98 by default);
* aggregated completeness at or above the threshold (75 by default).

Command::

    python validate_library.py equipment.json \
        --proof-dir proofs/ \
        --organization "Vendor name" \
        --contact "name@vendor.com" \
        --min-completeness 75

Two files are written in ``proofs/`` when the library is eligible:

* ``<name>.<sha256[:12]>.json``: a byte-exact copy of the validated library,
  named after the first 12 hexadecimal digits of its SHA-256;
* a proof file containing the full SHA-256, the GLV version, the thresholds
  used and the global score.

If the library is not eligible, the reasons are printed and nothing is written.

7. Using the certified copy
---------------------------

* The copy is not modified. Any change to the content gives a different hash,
  which invalidates the proof.
* The name suffix is accepted when comparing ``library-information/name`` with
  the file name. The name stored in the library must be the original file name,
  without the hash.
* To verify a copy, compute its SHA-256 and compare it with the one in the
  proof file.

8. Recommendations for vendors
------------------------------

* Encode the library as RFC7951 (numbers as strings, namespaced identities).
  Legacy encoding is accepted with a small penalty (``NOT_RFC7951``).
* Fill in ``library-information`` (``name``, ``organization``, ``contact``,
  ``revision``).
* Define ``f_min`` and ``f_max`` for every amplifier, and one amplifier per band
  for multiband amplifiers, with non-overlapping ranges.
* Provide penalties for chromatic dispersion, PMD and PDL in every transceiver
  mode, with penalty values that do not decrease as the impairment increases.
* Define ``equalization_offset_db`` for modes whose baud rate differs from the
  reference channel if ROADMs use ``target_pch_out_db``.
* Provide ``roadm-path-impairments`` for add, drop and express paths in every
  frequency band.

9. Confidentiality of the proof
-------------------------------

The proof of validation does not disclose the content of the library.

* The proof contains only the SHA-256 hash of the file, the GLV version, the
  thresholds used, the global score, and the organization and contact
  provided by the vendor.
* SHA-256 is a one-way function: the content of a file cannot be recovered
  from its hash.
* The proof contains no equipment data (no amplifier, transceiver or ROADM
  parameter, and no issue details).
* The hash can only be used to check a file that the verifier already has:
  computing the SHA-256 of that file and comparing it with the one in the
  proof confirms whether it is identical to the validated library.
* A vendor can share the proof alone, and share the library separately and
  only with the intended recipients.

Note that the certified copy (``<name>.<sha256[:12]>.json``) is a full copy
of the library: it carries the same confidentiality as the library itself.
Only the proof file and the hash are safe to publish.

10. Limits
----------

* GLV checks format and internal consistency. It does not verify measured
  performance of the equipment.
* Some checks apply only if the related data is present (for example the ROADM
  and amplifier power checks need the SI section).
* A proof states that the library passed the checks of a given GLV version at
  a given date. It does not endorse its content.
* The proof is not signed and not timestamped: it shows that a file is
  identical to the validated one, but does not prove who produced the proof
  or when. A signature or an external timestamp would be needed for that.
* The 12-digit hash in the file name is meant to distinguish versions. Use
  the full SHA-256 of the proof file for integrity verification.