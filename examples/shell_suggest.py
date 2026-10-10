"""Natural language -> ONE shell command, shown to a person and run only after they confirm (standard library only).

Uses any OpenAI-compatible endpoint; the default is llama-server on port 8080:

    llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --port 8080
    python shell_suggest.py "show the 5 biggest files in my Downloads folder"
    python shell_suggest.py --base-url http://localhost:11434/v1 --model udgam "count lines in every .py file here"

The model proposes a `run_shell_command` call with an explanation and a `requires_confirmation` flag, or replies in
words when it needs to ask or refuses. This script ALWAYS asks before running anything, whatever the flag says:
the model's commands are often wrong on unfamiliar tasks and it does not reliably refuse destructive requests.
"""
import argparse
import json
import platform
import subprocess
import urllib.request

SHELL_TOOL = {"type": "function", "function": {
    "name": "run_shell_command",
    "description": "Propose ONE shell command (bash/zsh) for the user's terminal. The app shows it to the user and "
                   "only runs it after approval.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "The exact command line"},
        "explanation": {"type": "string", "description": "One short sentence saying what it does"},
        "requires_confirmation": {"type": "boolean", "description": (
            "true if it modifies or deletes existing files, changes system state, installs or removes software, "
            "uses sudo, or changes remote state; false for read-only commands")},
        "platform": {"type": "string", "enum": ["macos", "linux", "any"],
                     "description": "Platform the command is written for"}},
        "required": ["command", "explanation", "requires_confirmation", "platform"]}}}


def device_header():
    if platform.system() == "Darwin":
        return "Device: Mac laptop\nOS: macOS · Shell: zsh"
    return "Device: Linux laptop\nOS: Linux · Shell: bash"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?", default="show the 5 biggest files in my Downloads folder")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="udgam")
    a = ap.parse_args()

    system = "Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)\n" + device_header()
    body = {"model": a.model, "tools": [SHELL_TOOL], "temperature": 0.1, "top_p": 0.9, "top_k": 40,
            "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": a.request}]}
    req = urllib.request.Request(f"{a.base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        msg = json.load(r)["choices"][0]["message"]

    calls = [c for c in (msg.get("tool_calls") or []) if c["function"]["name"] == "run_shell_command"]
    if not calls:
        print("assistant:", (msg.get("content") or "").strip())  # a question, a refusal, or a plain command block
        return

    args = json.loads(calls[0]["function"]["arguments"] or "{}")
    print(f"command:      {args.get('command')}")
    print(f"explanation:  {args.get('explanation')}")
    print(f"model says confirmation needed: {args.get('requires_confirmation')}  (platform: {args.get('platform')})")
    try:
        answer = input("Run this command? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Not run.")
        return
    subprocess.run(args["command"], shell=True)  # only after an explicit "y" from the person at the keyboard


if __name__ == "__main__":
    main()
