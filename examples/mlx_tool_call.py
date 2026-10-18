"""Tool calling with MLX on Apple silicon (tested with mlx-lm 0.32).

    pip install mlx-lm
    python mlx_tool_call.py                                    # 4-bit MLX weights (216 MB)
    python mlx_tool_call.py --model udgamlabs/Udgam-384M-Chat-MLX
    python mlx_tool_call.py --prompt "Text Priya that I'll be 10 minutes late."

Like transformers, MLX returns raw Gemma 4 text; gemma4_tools.py (next to this file) parses the tool call.
"""
import argparse
import json

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from gemma4_tools import parse_tool_calls, strip_special

SYSTEM = "Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)\nDevice: Android phone"


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


TOOLS = [
    tool("set_alarm", "Set an alarm at a local time.",
         {"time": {"type": "string", "description": "HH:MM 24h"},
          "label": {"type": "string", "description": "Optional label"}}, ["time"]),
    tool("send_message", "Send a text message to a contact.",
         {"contact": {"type": "string", "description": "Contact name"},
          "text": {"type": "string", "description": "Message body"}}, ["contact", "text"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="udgamlabs/Udgam-384M-Chat-MLX-4bit")
    ap.add_argument("--prompt", default="Wake me up at 6:45 tomorrow, label it gym.")
    a = ap.parse_args()

    model, tokenizer = load(a.model)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": a.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                   sampler=make_sampler(temp=0.0))  # greedy; or make_sampler(temp=0.1, top_p=0.9, top_k=40)

    calls = parse_tool_calls(raw)
    if not calls:
        print("assistant:", strip_special(raw))
        return
    for call in calls:
        print(f"tool call: {call['name']}({json.dumps(call['arguments'])})")
    # From here, validate the call, run your tool and send the result back as a `tool` message
    # (see transformers_tool_call.py for the second turn).


if __name__ == "__main__":
    main()
