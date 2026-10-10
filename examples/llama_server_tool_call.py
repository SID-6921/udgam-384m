"""Tool calling through an OpenAI-compatible /v1/chat/completions endpoint (standard library only).

Works with llama.cpp's llama-server (start it with --jinja so tool calls are parsed):

    llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --port 8080
    python llama_server_tool_call.py
    python llama_server_tool_call.py --prompt "Send a message to Rahul."   # missing detail -> it should ask

The same request shape works with LM Studio's local server (--base-url http://localhost:1234/v1 --model <id>) and
Ollama's OpenAI-compatible endpoint (--base-url http://localhost:11434/v1 --model udgam).
"""
import argparse
import json
import urllib.request

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
    tool("get_weather", "Current weather for a city.",
         {"city": {"type": "string", "description": "City name"}}, ["city"]),
]
ALLOWED = {t["function"]["name"] for t in TOOLS}


def run_tool(name, args):
    """Stub implementations; replace with your app's real functions (and confirm risky ones)."""
    return {"set_alarm": {"ok": True, "time": args.get("time"), "label": args.get("label")},
            "send_message": {"ok": True, "delivered_to": args.get("contact")},
            "get_weather": {"city": args.get("city"), "condition": "sunny", "temp_c": 21}}.get(
        name, {"ok": False, "error": "unknown tool"})


def complete(base_url, model, messages):
    body = {"model": model, "messages": messages, "tools": TOOLS,
            "temperature": 0.1, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False}}  # thinking off (llama-server)
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="udgam")
    ap.add_argument("--prompt", default="What's the weather like in Leeds right now?")
    a = ap.parse_args()

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": a.prompt}]
    msg = complete(a.base_url, a.model, messages)
    calls = msg.get("tool_calls") or []
    if not calls:
        print("assistant:", (msg.get("content") or "").strip())
        return

    messages.append(msg)
    for call in calls:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"] or "{}")  # arguments arrive as a JSON string here
        print(f"tool call: {name}({json.dumps(args)})")
        result = run_tool(name, args) if name in ALLOWED else {"ok": False, "error": "tool not allowed"}
        print(f"tool result: {json.dumps(result)}")
        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result)})

    print("assistant:", (complete(a.base_url, a.model, messages).get("content") or "").strip())


if __name__ == "__main__":
    main()
