"""Document -> JSON: fill a JSON schema from a receipt, email, form or OCR text (standard library only).

Uses any OpenAI-compatible endpoint; the default is llama-server on port 8080:

    llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --port 8080
    python doc_to_json.py                         # built-in sample receipt
    python doc_to_json.py --file my_receipt.txt
    python doc_to_json.py --base-url http://localhost:11434/v1 --model udgam   # Ollama

Pattern: put ONE schema in the system message with "Reply with JSON only", put the document in the user message, then
validate what comes back before you store it. Fields the document does not state should come back as null.
"""
import argparse
import json
import urllib.request

SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "date": {"type": "string", "description": "YYYY-MM-DD"},
        "currency": {"type": "string", "description": "ISO 4217 code"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "quantity": {"type": "integer"}, "amount": {"type": "number"}},
            "required": ["name", "quantity", "amount"]}},
        "total": {"type": "number"},
        "payment_method": {"type": ["string", "null"]},
    },
    "required": ["vendor", "date", "currency", "items", "total", "payment_method"],
}

SAMPLE = """RECEIPT
Corner Cafe, 12 High St
14/08/2027  10:42
2x Latte          6.40
Blueberry muffin  2.50
TOTAL GBP         8.90
Paid by card"""

TYPES = {"string": str, "number": (int, float), "integer": int, "object": dict, "array": list, "boolean": bool,
         "null": type(None)}


def check(value, schema, path="$"):
    """A tiny structural validator (types + required keys). Use the `jsonschema` package in production."""
    problems = []
    allowed = schema.get("type")
    if allowed:
        allowed = allowed if isinstance(allowed, list) else [allowed]
        if not any(isinstance(value, TYPES[t]) and not (t in ("number", "integer") and isinstance(value, bool))
                   for t in allowed):
            return [f"{path}: expected {'/'.join(allowed)}, got {type(value).__name__}"]
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{path}.{key}: missing")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                problems += check(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            problems += check(item, schema["items"], f"{path}[{i}]")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="udgam")
    ap.add_argument("--file", help="text file to extract from (default: a sample receipt)")
    a = ap.parse_args()
    document = open(a.file, encoding="utf-8").read() if a.file else SAMPLE

    system = ("Current date-time: Saturday, 2027-08-14 18:05 (Europe/London, UTC+01:00)\nDevice: iPhone\n\n"
              "Expense tracker. Turn the receipt into a JSON object that matches the schema below. Reply with JSON only.\n"
              + json.dumps(SCHEMA, separators=(",", ":")))
    body = {"model": a.model, "temperature": 0.1, "top_p": 0.9, "top_k": 40, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "save this receipt please\n\n" + document}]}
    req = urllib.request.Request(f"{a.base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = (json.load(r)["choices"][0]["message"].get("content") or "").strip()

    if text.startswith("```"):  # tolerate a fenced reply
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print("not valid JSON:\n" + text)
        raise SystemExit(1)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    problems = check(data, SCHEMA)
    print("schema check:", "OK" if not problems else "; ".join(problems))


if __name__ == "__main__":
    main()
