"""Tool calling with Hugging Face transformers (tested with transformers 5.18, PyTorch, CPU or Apple MPS).

    pip install transformers torch
    python transformers_tool_call.py
    python transformers_tool_call.py --model /path/to/local/Udgam-384M-Chat

The model writes raw Gemma 4 tool-call text; gemma4_tools.py (next to this file) parses it. The tool result goes back
as a `tool` message and the chat template renders it for the final reply. Greedy decoding in fp32 gives reproducible
calls.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_tools import parse_tool_calls, strip_special

SYSTEM = "Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)\nDevice: Android phone"
TOOLS = [{"type": "function", "function": {
    "name": "set_alarm", "description": "Set an alarm at a local time.",
    "parameters": {"type": "object", "properties": {
        "time": {"type": "string", "description": "HH:MM 24h"},
        "label": {"type": "string", "description": "Optional label"}}, "required": ["time"]}}}]


def generate(tok, model, messages, max_new_tokens=256):
    inputs = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt")  # thinking off unless enable_thinking=True
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="udgamlabs/Udgam-384M-Chat")
    ap.add_argument("--prompt", default="Wake me up at 6:45 tomorrow, label it gym.")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval()

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": a.prompt}]
    raw = generate(tok, model, messages)
    calls = parse_tool_calls(raw)
    if not calls:
        print("assistant:", strip_special(raw))
        return

    call = calls[0]
    print(f"tool call: {call['name']}({json.dumps(call['arguments'])})")
    result = {"ok": True, **call["arguments"]}  # your app runs the real tool here
    print(f"tool result: {json.dumps(result)}")
    messages.append({"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]})
    messages.append({"role": "tool", "name": call["name"], "content": json.dumps(result)})
    print("assistant:", strip_special(generate(tok, model, messages)))


if __name__ == "__main__":
    main()
