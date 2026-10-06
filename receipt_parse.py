#!/usr/bin/env python3
"""Parse store, items and total from a receipt image via a local Ollama
vision model, and save the result as a JSON file with the .parsed.json
suffix (i.e. the same basename with .parsed.json appended).

Usage:
    python3 receipt_parse.py [--force] <receipt_image> [...]

By default, receipts whose <basename>.parsed.json already exists are skipped.
Pass --force to re-parse and overwrite existing JSON files.

Dependencies: Python stdlib + Pillow. Ollama must be running on
OLLAMA_HOST (default http://localhost:11434).

If the LLM call fails after all retries (network error, malformed output, or
response failing schema validation), the script crashes; no output file is
written for that file.
Default: 5 retries (RECEIPT_MAX_RETRIES) with exponential backoff
(RECEIPT_RETRY_DELAY, base seconds).
"""

import base64
import json
import os
import sys
import time
import urllib.request

import schema_validation

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Pillow is required: python3 -m pip install --user Pillow")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("RECEIPT_MODEL", "qwen3.8:27b")
TIMEOUT = int(os.environ.get("RECEIPT_TIMEOUT", "300"))

MAX_RETRIES = int(os.environ.get("RECEIPT_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("RECEIPT_RETRY_DELAY", "2"))

PROMPT = """\
You are reading a photograph of a printed receipt. Carefully transcribe it.

Return ONLY a valid JSON object, no markdown fences, with exactly this shape:
{
  "store": "store/merchant name if you can read it clearly, else \"\"",
  "items": [
    {"name": "...", "quantity": 2, "unit_price": 3.69, "line_total": 7.38,
     "confidence": "high"}
  ],
  "total": 6.90,
  "currency": "RON",
  "payment_method": "card",
  "summary": "1-2 sentences describing the whole purchase."
}

Rules:
- items: one entry per purchased line item, in receipt order. Treat deposits,
  biobags, fees and discounts as items too (discounts line_total negative).
- quantity may be null if it is a weight (kg) or not printed.
- unit_price and line_total are numbers; use null when a value is not
  legible or not printed (e.g. some restaurant receipts only print line totals).
- "confidence": "high" only if you read the text clearly; "low" otherwise.
- NEVER guess or invent text. If a name, quantity or price is not clearly
  legible, leave that field empty ("" for text, null for numbers) and set
  confidence to "low". A price or number on a line does not identify the
  product - do not name a product from surrounding context, category or
  price alone.
- total: the grand TOTAL printed on the receipt (numbers only, no currency).
  null if not legible.
- currency: 3-letter code. Use it directly if explicitly printed on the
  receipt. Otherwise infer the country from the store name, address or
  language of the receipt, and use that country's currency (e.g. Romania ->
  RON). Use "" only if the country/currency cannot be determined.
- summary: 1-2 sentences summarizing the whole purchase (store, main items).
  Write it in English. Do NOT mention any amount or total - the total is
  already captured in the "total" field. Do NOT mention how the purchase was
  paid - no card/cash, no card brand (e.g. Mastercard, Visa), no contactless/
  POS/terminal details - that is already captured in the "payment_method"
  field. Do NOT mention discounts or bottle/packaging deposit guarantees or
  recycling fees - these are recorded as items and do not describe the
  purchase; focus the summary on the main product(s) bought.
- payment_method: "card" or "cash" ONLY if the receipt explicitly prints the
  payment method; otherwise "".
- store: prefer the store/merchant name from the HEADER (top) of the receipt.
  Only if the header is a logo, blurred or otherwise illegible, fall back to
  the store/merchant name printed near the bottom (legal/registration name is
  fine for this). Leave store as "" only if no store name is legible at all.
- Ignore card transaction details (RRN, terminal, auth codes).
"""


def preprocess(img_path):
    """Upscale and auto-contrast the image to improve OCR/vision accuracy."""
    with Image.open(img_path) as im:
        gray = im.convert("L")
    w, h = gray.size
    if h < 2600 and (w * h) <= 664 * 3200:
        scale = 2
    else:
        scale = 1
    if scale > 1:
        gray = gray.resize((w * scale, h * scale), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray)
    img_dir = os.path.dirname(os.path.abspath(img_path))
    buf = os.path.join(img_dir, "tmp_ollama_" + os.path.basename(img_path) + ".jpg")
    gray.save(buf, "JPEG", quality=95)
    with open(buf, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(buf)
    return b64


def is_retryable(exc):
    """True for transient failures worth retrying (HTTP 5xx/timeouts),
    False for client errors (4xx) where retrying will not help.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    return True  # URLError, timeout, connection reset, etc.


def ollama_chat(messages):
    """Single POST to /api/chat. Raises on any transport-level failure;
    the caller (parse) is responsible for retrying.
    """
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
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


RECEIPT_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["store", "items", "total", "currency", "payment_method",
                 "summary"],
    "properties": {
        "store": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "quantity", "unit_price", "line_total",
                             "confidence"],
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": ["number", "null"]},
                    "unit_price": {"type": ["number", "null"]},
                    "line_total": {"type": ["number", "null"]},
                    "confidence": {"type": "string", "enum": ["high", "low"]},
                },
            },
        },
        "total": {"type": ["number", "null"]},
        "currency": {"type": "string"},
        "payment_method": {"type": "string"},
        "summary": {"type": "string"},
    },
}


def parse(img_path):
    """Parse receipt data, retrying up to MAX_RETRIES times on transient
    network errors, unparseable model output, or a response that fails
    validation against RECEIPT_RESPONSE_SCHEMA (validation is provided by
    the shared schema_validation module, which raises a SchemaError/ValueError).
    Raises the last error if all retries fail (main() then crashes).
    """
    image_b64 = preprocess(img_path)
    messages = [
        {"role": "system", "content": "You transcribe receipts into strict JSON."},
        {
            "role": "user",
            "content": PROMPT,
            "images": [image_b64],
        },
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            content = ollama_chat(messages)["message"]["content"]
            data = normalize(parse_json_loose(content))
            schema_validation.validate(data, RECEIPT_RESPONSE_SCHEMA)
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, OSError, json.JSONDecodeError,
                KeyError, AttributeError, ValueError) as exc:
            if attempt == MAX_RETRIES or not is_retryable(exc):
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[parse] attempt {attempt}/{MAX_RETRIES} failed: "
                  f"{exc}; retrying in {delay:.0f}s ...", file=sys.stderr)
            time.sleep(delay)


def parse_json_loose(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if ln.strip().startswith("{")]
        text = "\n".join(lines)
        close = text.rfind("}")
        if close != -1:
            text = text[: close + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def clean_str(value):
    if value is None:
        return ""
    s = str(value).strip()
    if s == "null" or s == "N/A" or s == "n/a":
        return ""
    return s


def clean_num(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip().replace(",", "").replace(" ", "")
    if s in ("", "null", "N/A", "n/a"):
        return None
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def normalize(data):
    items = []
    for raw in data.get("items", []) or []:
        if not isinstance(raw, dict):
            continue
        items.append({
            "name": clean_str(raw.get("name")),
            "quantity": clean_num(raw.get("quantity")),
            "unit_price": clean_num(raw.get("unit_price")),
            "line_total": clean_num(raw.get("line_total")),
            "confidence": clean_str(raw.get("confidence")) or "",
        })
    return {
        "store": clean_str(data.get("store")),
        "items": items,
        "total": clean_num(data.get("total")),
        "currency": clean_str(data.get("currency")),
        "payment_method": clean_str(data.get("payment_method")),
        "summary": clean_str(data.get("summary")),
    }


def main(argv):
    args = [a for a in argv[1:] if a != "--force"]
    force = "--force" in argv[1:]
    if not args:
        sys.exit(__doc__)
    for img_path in args:
        if not os.path.isfile(img_path):
            sys.exit(f"file not found: {img_path}")
        out_path = os.path.splitext(img_path)[0] + ".parsed.json"
        if os.path.isfile(out_path) and not force:
            print(f"[receipt_parse] skipping {img_path} "
                  f"({out_path} exists, use --force to overwrite)", file=sys.stderr)
            continue
        print(f"[receipt_parse] processing {img_path} ...", file=sys.stderr)
        result = parse(img_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"[receipt_parse] wrote {out_path}", file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv)
