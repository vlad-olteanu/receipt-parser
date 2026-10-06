#!/usr/bin/env python3
"""Categorize receipt JSONs using a local Ollama model and a category schema.

Usage:
    python3 receipt_categorize.py [--force] SCHEMA.txt <receipt.parsed.json> [...]

SCHEMA.txt is a plain text file with one category dot-path per line
(e.g. "iesiri.restaurant"). Empty lines and lines starting with # are
ignored.

Reads each receipt JSON (store, items, total, summary), asks the LLM to pick
the best-fitting category from SCHEMA.txt, and writes the result next to the
input JSON as a JSON file with a .classified extension, e.g.
scan_1.parsed.json -> scan_1.classified.json.

Default behaviour: an existing .classified.json is skipped UNLESS its
"category" field is blank. Pass --force to overwrite unconditionally.

If the receipt cannot be placed into any category, a warning is printed to
stderr and the output file is written with a blank "category" line, so the
entry can be picked up again later.

If the LLM call itself fails after all retries (network error, malformed
output, or response failing schema validation), the script crashes; no output
file is written for that file.

Dependencies: Python stdlib only (plus the local
schema_validation module). Ollama must be running on OLLAMA_HOST
(default http://localhost:11434).
"""

import json
import os
import re
import sys
import time
import urllib.request

import schema_validation

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("RECEIPT_MODEL", "qwen3.8:27b")
TIMEOUT = int(os.environ.get("RECEIPT_TIMEOUT", "300"))

OUTPUT_EXT = ".classified.json"
PARSED_RE = re.compile(r"(?:\.parsed)?\.json$")


def derive_output_path(input_path):
    """Map a receipt input path to its <same>.classified.json output path.
    A '.parsed.json' suffix (from receipt_parse.py) or a bare '.json' suffix
    on the input is stripped; '.classified.json' inputs are treated as output
    files by main() and never reach here.
    """
    base = PARSED_RE.sub("", input_path)
    return base + OUTPUT_EXT


MAX_RETRIES = int(os.environ.get("RECEIPT_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("RECEIPT_RETRY_DELAY", "2"))

RECEIPT_CATEGORY_SCHEMA = {
    "type": "object",
    "required": ["category"],
    "properties": {
        "category": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
}


def is_retryable(exc):
    """True for transient failures worth retrying (HTTP 5xx/timeouts),
    False for client errors (4xx) where retrying will not help.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    return True  # URLError, timeout, connection reset, etc.


def ollama_chat(messages):
    """POST to /api/chat with exponential backoff on transient failures.
    Raises the last error if all retries are exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(
            OLLAMA_HOST + "/api/chat",
            data=json.dumps({
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.load(resp)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, OSError, json.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES or not is_retryable(exc):
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[categorize] attempt {attempt}/{MAX_RETRIES} failed: "
                  f"{exc}; retrying in {delay:.0f}s ...", file=sys.stderr)
            time.sleep(delay)


def parse_schema(schema_path):
    """Read the schema file: one category dot-path per line. Returns the
    list of categories, e.g. ['masina.motorina', 'supermarket'].
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        categories = []
        for lineno, line in enumerate(f, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            if "." not in text and not text.isidentifier():
                print(f"[categorize] WARNING: {schema_path}:{lineno}: "
                      f"skipping malformed category '{text}'", file=sys.stderr)
                continue
            if text in categories:
                continue
            categories.append(text)
    return categories


def build_prompt(receipt, categories):
    return f"""You are classifying a single purchase (parsed from a receipt)
into exactly one of a fixed list of categories.

Available categories (use these dot-paths VERBATIM, pick the MOST SPECIFIC
one that fits):
{"\n".join(categories)}

The purchase:
{json.dumps(receipt, ensure_ascii=False, indent=2)}

Choose the single best-fitting category from the list. Consider the store,
every item name, and the summary. Do NOT invent a category that is not in
the list. If nothing in the list plausibly describes this purchase, return
the empty list as the answer (i.e. classify it as uncategorizable, do NOT
pick the closest-sounding one).

Return ONLY a JSON object in exactly this shape (no markdown, no commentary):
{{"category": "<dot.path.from.the.list> | null", "reason": "<one short sentence>"}}
"""


def parse_llm_output(content):
    """Tolerant parse of the LLM's output: strip markdown fences if present,
    then fall back to the outermost {...} if the body is not pure JSON.
    Raises json.JSONDecodeError when no JSON object can be recovered.
    """
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(ln for ln in lines if ln.strip().startswith("{"))
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start : end + 1])
        raise


def classify(receipt, categories):
    """Classify a receipt into one of ``categories`` and return
    ``(category, reason)``; ``category`` is None when the model answers with
    no valid category (empty/None or an invented path not in ``categories``).

    Retries up to MAX_RETRIES times on transient network errors and on
    malformed model output (unparseable JSON or a response that fails
    schema validation against RECEIPT_CATEGORY_SCHEMA). Retries with
    exponential backoff (RETRY_BASE_DELAY). If all retries fail the last
    error is raised and the script exits (see main()).
    """
    messages = [
        {"role": "system", "content": "You classify purchases. Output strict JSON."},
        {"role": "user", "content": build_prompt(receipt, categories)},
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = ollama_chat(messages)
            content = result["message"]["content"].strip()
            data = parse_llm_output(content)
            schema_validation.validate(data, RECEIPT_CATEGORY_SCHEMA)
            break
        except (json.JSONDecodeError, ValueError, KeyError,
                AttributeError) as exc:
            if attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[categorize] attempt {attempt}/{MAX_RETRIES} failed: "
                  f"{exc}; retrying in {delay:.0f}s ...", file=sys.stderr)
            time.sleep(delay)
    category = data.get("category")
    reason = str(data.get("reason", "")).strip()

    valid = set(categories) | {None, ""}
    if category not in valid:
        # The model invented a path -> treat as uncategorizable
        print(f"[categorize] WARNING: model returned unknown category "
              f"'{category}' -> treating as blank", file=sys.stderr)
        return None, reason
    if category is None or category == "":
        return None, reason
    return category, reason


def build_output(category, reason, input_path, receipt):
    return {
        "category": category or "",
        "reason": reason,
        "source": os.path.basename(input_path),
        "total": receipt.get("total"),
        "currency": receipt.get("currency") or "",
        "store": receipt.get("store") or "",
        "summary": receipt.get("summary") or "",
    }


def read_existing_category(output_path):
    """Return the current 'category' value of an existing output JSON file,
    or None if the file cannot be read / has no category field.
    """
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data.get("category")


def main(argv):
    args = [a for a in argv[1:] if a != "--force"]
    force = "--force" in argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)

    schema_path, receipts = args[0], args[1:]
    if not os.path.isfile(schema_path):
        sys.exit(f"schema not found: {schema_path}")
    categories = parse_schema(schema_path)
    if not categories:
        sys.exit("schema contains no categories")

    handled, skipped, warned = 0, 0, 0
    for path in receipts:
        if not os.path.isfile(path):
            sys.exit(f"file not found: {path}")
        if path.endswith(OUTPUT_EXT):
            print(f"[categorize] skipping {path} "
                  f"(this is already an output file, not a receipt JSON)",
                  file=sys.stderr)
            skipped += 1
            continue
        out_path = derive_output_path(path)

        if os.path.isfile(out_path) and not force:
            existing = read_existing_category(out_path)
            if not existing:
                # Blank (or missing) category on disk -> retry this one
                print(f"[categorize] retrying {path} "
                      f"(existing category is blank)", file=sys.stderr)
            else:
                print(f"[categorize] skipping {path} "
                      f"({out_path} has category "
                      f"'{existing}', use --force to overwrite)",
                      file=sys.stderr)
                skipped += 1
                continue

        with open(path, "r", encoding="utf-8") as f:
            receipt = json.load(f)
        print(f"[categorize] processing {path} ...", file=sys.stderr)
        category, reason = classify(receipt, categories)
        if category is None:
            print(f"[categorize] WARNING: could not assign {path} to any "
                  f"category (reason: {reason or 'none given'}). Writing blank.",
                  file=sys.stderr)
            warned += 1
        doc = build_output(category, reason, path, receipt)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[categorize] wrote {out_path} "
              f"(category={category or '(blank)'})", file=sys.stderr)
        handled += 1

    print(f"[categorize] done: {handled} processed, {skipped} skipped, "
          f"{warned} uncategorizable", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
