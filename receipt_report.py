#!/usr/bin/env python3
"""Generate a YAML report from classified receipt JSONs.

Usage:
    python3 receipt_report.py [--output FILE] <receipt.classified.json> [...]

If --output FILE is given the report is written to FILE (overwriting any
existing file) and not printed to stdout. The flag accepts either "--output
FILE" or "--output=FILE"; a short form "-o FILE" is also accepted.

Each input file is grouped by its "currency" field, then by the parts of its
"category" dot-path (e.g. category "iesiri.restaurant" nests as
currency -> iesiri -> restaurant). Under each currency/category path, one
entry is listed per file, a dict with keys:
  - desc:   a short description (falls back from "summary" -> "store" -> filename)
  - amount: the receipt "total"

The report is printed to stdout. Currency and category keys are sorted for
stable output; entries within a group keep the order the files were passed on
the command line. Files missing a category, currency, or numeric total are
skipped with a warning on stderr.

Dependencies: Python stdlib only.
"""

import json
import os
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report(paths):
    report = {}
    for path in paths:
        if not os.path.isfile(path):
            sys.exit(f"file not found: {path}")
        try:
            data = load_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[report] skipping {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"[report] skipping {path}: not a JSON object", file=sys.stderr)
            continue

        category = str(data.get("category") or "").strip()
        currency = str(data.get("currency") or "").strip()
        if not category or not currency:
            print(f"[report] skipping {path}: missing category "
                  f"({category!r}) or currency ({currency!r})", file=sys.stderr)
            continue

        amount = data.get("total")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            print(f"[report] skipping {path}: missing numeric total", file=sys.stderr)
            continue

        desc = (str(data.get("summary") or "").strip()
                or str(data.get("store") or "").strip()
                or os.path.basename(path))

        report.setdefault(currency, {}).setdefault(category, []).append(
            {"desc": desc, "amount": amount})
    return report


def yaml_string(value):
    """A JSON-encoded string is a valid YAML double-quoted scalar, so reusing
    json.dumps guarantees correct quoting/escaping with no extra dependency."""
    return json.dumps(str(value), ensure_ascii=False)


def to_yaml(report):
    lines = []
    for currency in sorted(report):
        lines.append(f"{currency}:")
        categories = report[currency]
        for category in sorted(categories):
            # A category dot-path (e.g. "iesiri.restaurant") becomes one
            # level of YAML nesting per path part.
            parts = category.split(".")
            prefix = "  "
            for part in parts[:-1]:
                lines.append(f"{prefix}{part}:")
                prefix += "  "
            lines.append(f"{prefix}{parts[-1]}:")
            for item in categories[category]:
                lines.append(f"{prefix}  - desc: {yaml_string(item['desc'])}")
                lines.append(f"{prefix}    amount: {json.dumps(item['amount'])}")
    return "\n".join(lines)


def main(argv):
    args = list(argv[1:])
    out_path = None
    # Pull an output-file flag out of the args, supporting both forms:
    # "--output FILE", "--output=FILE", "-o FILE", "-oFILE".
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--output", "-o"):
            if i + 1 >= len(args):
                sys.exit(f"{a} requires a file argument")
            out_path = args[i + 1]
            del args[i:i + 2]
        elif a.startswith("--output="):
            out_path = a.split("=", 1)[1]
            del args[i]
        elif a.startswith("-o") and len(a) > 2:
            out_path = a[2:]
            del args[i]
        else:
            i += 1
    if out_path and (out_path.startswith("-") and len(out_path) > 1):
        sys.exit(f"invalid output path: {out_path!r}")

    if not args:
        sys.exit(__doc__)
    report = build_report(args)
    text = to_yaml(report)
    if not text:
        if out_path:
            print("[report] no entries to report; not writing "
                  f"{out_path}", file=sys.stderr)
        return
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        print(text)


if __name__ == "__main__":
    main(sys.argv)
