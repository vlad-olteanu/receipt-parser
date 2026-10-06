# receipt-parser

Takes receipt photos, parses them with a vision model, tags them with a category, then creates a YAML
summary of the spending. Three steps, one script each.

1. `receipt_parse.py` - reads a receipt image, writes `<name>.parsed.json`
2. `receipt_categorize.py` - assigns each parsed receipt a category, writes `<name>.classified.json`
3. `receipt_report.py` - groups the classified receipts into a YAML report

Steps 1 and 2 call a local LLM via [Ollama](https://ollama.com). Step 3 is pure
Python.

## End to end

Run all three steps at once, images first and the report path last:

```
python3 run.py [--force] receipt1.jpg ... receiptn.jpg report.yaml
```

Any failing step stops the whole run (non-zero exit) and each script's log
lines appear as they run. `--force` re-parses and re-categorizes even if the
JSON files already exist.


## Requirements

- Python 3
- Pillow (only for step 1): `pip install -r requirements.txt`
- Ollama running locally, with a vision model pulled

## 1. Parse a receipt

Reads the image and writes the store, items, total, currency and a short summary.
Skips files that already have a `.parsed.json` unless you pass `--force`.

```
python3 receipt_parse.py [--force] receipt.jpg [more.jpg ...]
```

`receipt.jpg` -> `receipt.parsed.json`:

```json
{
  "store": "profi",
  "items": [
    {"name": "AQUA CARPAT.APA PLAT", "quantity": 2.0, "unit_price": 3.69,
     "line_total": 7.38, "confidence": "high"}
  ],
  "total": 6.9,
  "currency": "RON",
  "payment_method": "card",
  "summary": "Purchase at profi of two bottles of Aqua Carpat water."
}
```

## 2. Categorize

Reads a `.parsed.json`, asks the model to pick one category from a schema file
(one dot-path per line, `#` starts a comment), and writes `<name>.classified.json`.
A file is skipped unless its `.classified.json` has an empty category or you
pass `--force`.

```
python3 receipt_categorize.py [--force] schema.txt receipt.parsed.json [more ...]
```

`schema.txt`:

```
supermarket
leasure.restaurant
car.gasoline
```

`receipt.parsed.json` -> `receipt.classified.json`:

```json
{
  "category": "supermarket",
  "reason": "Bottled water at a supermarket.",
  "source": "receipt.parsed.json",
  "total": 6.9,
  "currency": "RON",
  "store": "profi",
  "summary": "Purchase at profi of two bottles of Aqua Carpat water."
}
```

If no category fits, the file is still written with `"category": ""` so it can
be picked up again later/filled in manually.

## 3. Report

Groups classified receipts under `currency` -> `category`, nesting each level
of a dot-path (e.g. `leasure.restaurant` becomes `leasure:` then `restaurant:`).
Prints to stdout by default; use `--output FILE` to write to a file instead.

```
python3 receipt_report.py [--output report.yaml] receipt.classified.json [more ...]
```

Output:

```yaml
RON:
  leasure:
    restaurant:
      - desc: "A delivery order of fish and sushi plates."
        amount: 215.0
  supermarket:
    - desc: "A purchase at profi of two bottles of Aqua Carpat water."
      amount: 6.9
    - desc: "A Kaufland grocery purchase including chicken, vegetables and fruit."
      amount: 425.85
```

## Configuration

All settings are environment variables with sensible defaults. Override any of
them for a run, e.g. `RECEIPT_MODEL=llava python3 receipt_parse.py receipt.jpg`.

- `OLLAMA_HOST` (default `http://localhost:11434`): Ollama server URL. Parse and classify.
- `RECEIPT_MODEL` (default `qwen3.8:27b`): model name to call. Parse and classify.
- `RECEIPT_TIMEOUT` (default `300`): per-request timeout, seconds. Parse and classify.
- `RECEIPT_MAX_RETRIES` (default `5`): retries on network or bad-output failures. Parse and classify.
- `RECEIPT_RETRY_DELAY` (default `2`): base backoff in seconds, doubles each attempt. Parse and classify.

## Validation

The JSON returned by the model is checked against a JSON Schema before it is
used (`schema_validation.py`). A response that fails the schema (wrong shape,
bad types, a category not in the list) is retried like any other failure and
never written to an output file as-is.
