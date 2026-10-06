#!/usr/bin/env python3
"""Run the whole pipeline end to end: parse, categorize, report.

Usage:
    python3 run.py [--force] <receipt1.jpg> ... <receiptn.jpg> <yaml_report_path>

Steps:
  1. receipt_parse.py      each image       -> <name>.parsed.json
  2. receipt_categorize.py each parsed file -> <name>.classified.json
  3. receipt_report.py     all classified   -> <yaml_report_path>

If any step fails, the run stops immediately (and exits non-zero); output
files written by earlier steps are left in place. Log lines from each script
are printed to stderr as they run.

By default an existing .parsed.json / .classified.json is reused; pass
--force to re-run both steps and overwrite them.

The category schema file is schema.txt next to this script (override with
RECEIPT_SCHEMA).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import receipt_categorize
import receipt_parse
import receipt_report


def main(argv):
    args = list(argv[1:])
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if len(args) < 2:
        sys.exit(__doc__)

    images, report_path = args[:-1], args[-1]
    if not images:
        sys.exit(__doc__)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.environ.get(
        "RECEIPT_SCHEMA", os.path.join(script_dir, "schema.txt"))
    if not os.path.isfile(schema_path):
        sys.exit(f"category schema not found: {schema_path}")
    for img in images:
        if not os.path.isfile(img):
            sys.exit(f"file not found: {img}")

    force_flag = ["--force"] if force else []

    parsed = [os.path.splitext(img)[0] + ".parsed.json" for img in images]
    receipt_parse.main(["receipt_parse.py", *force_flag, *images])

    classified = [receipt_categorize.derive_output_path(p) for p in parsed]
    receipt_categorize.main(
        ["receipt_categorize.py", *force_flag, schema_path, *parsed])

    receipt_report.main(
        ["receipt_report.py", "--output", report_path, *classified])
    print(f"[run] wrote {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
