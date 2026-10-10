"""Parse Gemma 4 tool calls from raw model text.

Ollama and llama-server (with --jinja) already return parsed `tool_calls`. When you run the model yourself
(transformers, MLX) you get the raw text instead, for example:

    <|tool_call>call:set_alarm{label:<|"|>Gym<|"|>,time:<|"|>06:45<|"|>}<tool_call|>

`parse_tool_calls(text)` turns that into [{"name": "set_alarm", "arguments": {"label": "Gym", "time": "06:45"}}].
Strings are wrapped in <|"|> ... <|"|>; numbers, booleans and null are bare; objects use {} and arrays use [].
Standard library only.
"""
import json
import re

QUOTE = '<|"|>'
CALL_RE = re.compile(r"<\|tool_call>call:([A-Za-z0-9_.\-]+)(\{.*?\})<tool_call\|>", re.S)


def _to_json(body: str) -> str:
    """Convert a Gemma 4 argument object ({key:<|"|>v<|"|>,n:3}) into JSON text."""
    out, i = [], 0
    while i < len(body):
        if body.startswith(QUOTE, i):
            end = body.find(QUOTE, i + len(QUOTE))
            if end < 0:
                raise ValueError("unterminated string in tool call")
            out.append(json.dumps(body[i + len(QUOTE):end]))
            i = end + len(QUOTE)
            continue
        m = re.match(r"[A-Za-z_][A-Za-z0-9_.\-]*", body[i:])
        if m:
            word = m.group(0)
            j = i + len(word)
            if j < len(body) and body[j] == ":":
                out.append(json.dumps(word))  # an object key
            else:
                out.append(word)  # true / false / null
            i = j
            continue
        out.append(body[i])
        i += 1
    return "".join(out)


def parse_tool_calls(text: str):
    """Return a list of {"name", "arguments"} dicts found in `text` (empty if the model replied in words)."""
    calls = []
    for name, body in CALL_RE.findall(text):
        try:
            args = json.loads(_to_json(body))
        except (ValueError, json.JSONDecodeError):
            args = {"_unparsed": body}
        calls.append({"name": name, "arguments": args})
    return calls


def strip_special(text: str) -> str:
    """Remove Gemma 4 control tokens from a plain-text reply."""
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.S)  # optional thinking block
    for tok in ("<turn|>", "<|tool_response>", "<eos>", "<bos>"):
        text = text.replace(tok, "")
    return text.strip()


if __name__ == "__main__":
    demo = '<|tool_call>call:set_alarm{label:<|"|>Gym<|"|>,time:<|"|>06:45<|"|>}<tool_call|><|tool_response>'
    print(parse_tool_calls(demo))
    nested = '<|tool_call>call:f{a:{b:[1,2.5,true,null]},s:<|"|>x, y: z<|"|>}<tool_call|>'
    print(parse_tool_calls(nested))
