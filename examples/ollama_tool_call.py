"""Tool calling through Ollama's native /api/chat endpoint (standard library only).

Setup (once): download Udgam-384M-Chat-Q8_0.gguf and the Modelfile from
https://huggingface.co/UdgamLabs/Udgam-384M-Chat-GGUF into one folder, then run `ollama create udgam -f Modelfile`.

    python ollama_tool_call.py
    python ollama_tool_call.py --prompt "Text Priya that I'll be 10 minutes late."
    python ollama_tool_call.py --prompt "Book me a table for two at 8pm."   # no tool fits -> it should decline

The loop: send the request with the app's tools -> the model returns a tool call (or a plain reply when it needs to
ask or decline) -> your app validates and runs the tool -> send the result back -> the model writes the final reply.
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
NEEDS_CONFIRMATION = {"send_message"}  # anything that sends, pays, deletes or shares


def run_tool(name, args):
    """Your app's real implementation goes here. These stubs just echo a plausible result."""
    if name == "set_alarm":
        return {"ok": True, "time": args.get("time"), "label": args.get("label")}
    if name == "send_message":
        return {"ok": True, "delivered_to": args.get("contact")}
    if name == "get_weather":
        return {"city": args.get("city"), "condition": "light rain", "temp_c": 14}
    return {"ok": False, "error": "unknown tool"}


def chat(host, model, messages):
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": False,
            "think": False,  # Ollama turns thinking ON by default for Gemma 4 models
            "options": {"temperature": 0.1, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1}}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["message"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="udgam")
    ap.add_argument("--prompt", default="Wake me up at 6:45 tomorrow, label it gym.")
    ap.add_argument("--yes", action="store_true", help="auto-confirm actions that need confirmation (demo only)")
    a = ap.parse_args()

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": a.prompt}]
    msg = chat(a.host, a.model, messages)
    calls = msg.get("tool_calls") or []
    if not calls:  # the model asked a question or declined
        print("assistant:", msg.get("content", "").strip())
        return

    messages.append(msg)
    for call in calls:
        name, args = call["function"]["name"], call["function"]["arguments"]
        print(f"tool call: {name}({json.dumps(args)})")
        if name not in ALLOWED:  # never trust the model's choice blindly
            result = {"ok": False, "error": "tool not allowed"}
        elif name in NEEDS_CONFIRMATION and not a.yes and input("  run it? [y/N] ").strip().lower() != "y":
            result = {"ok": False, "error": "user cancelled"}
        else:
            result = run_tool(name, args)
        print(f"tool result: {json.dumps(result)}")
        messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})

    print("assistant:", chat(a.host, a.model, messages).get("content", "").strip())


if __name__ == "__main__":
    main()
