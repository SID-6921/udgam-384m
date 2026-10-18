# Examples

Small, runnable scripts for Udgam-384M-Chat. The ones that talk to a server use only the Python standard library (Python 3.9+). [`gemma4_tools.py`](gemma4_tools.py) parses raw Gemma 4 tool calls for the transformers and MLX scripts.

| Script | Backend | Default endpoint / model |
|---|---|---|
| [`ollama_tool_call.py`](ollama_tool_call.py) | Ollama `/api/chat` | `http://localhost:11434`, model `udgam` |
| [`llama_server_tool_call.py`](llama_server_tool_call.py) | OpenAI-compatible `/v1/chat/completions` | `http://localhost:8080/v1` |
| [`transformers_tool_call.py`](transformers_tool_call.py) | transformers + PyTorch | `udgamlabs/Udgam-384M-Chat` |
| [`mlx_tool_call.py`](mlx_tool_call.py) | mlx-lm (Apple silicon) | `udgamlabs/Udgam-384M-Chat-MLX-4bit` |
| [`doc_to_json.py`](doc_to_json.py) | OpenAI-compatible | `http://localhost:8080/v1` |
| [`shell_suggest.py`](shell_suggest.py) | OpenAI-compatible | `http://localhost:8080/v1` |

Every script accepts `--model`, and the server scripts accept `--host` or `--base-url`, so you can point them at Ollama's `/v1`, LM Studio or `mlx_lm.server` instead.

## Setup

```bash
# Ollama: put Udgam-384M-Chat-Q8_0.gguf next to ../ollama/Modelfile, then
ollama create udgam -f Modelfile

# llama.cpp
llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --port 8080

# transformers / MLX
pip install transformers torch     # or: pip install mlx-lm
```

## Recorded output

We ran each script on 10 October 2026 against the released files: the Q8_0 GGUF in Ollama and llama-server, the safetensors in transformers 5.18, and the MLX 4-bit and fp16 weights in mlx-lm 0.32. The replies come from the model and can differ slightly between runtimes and versions.

```text
$ python ollama_tool_call.py
tool call: set_alarm({"label": "Gym", "time": "06:45"})
tool result: {"ok": true, "time": "06:45", "label": "Gym"}
assistant: Set an alarm for 6:45 am tomorrow.

$ python ollama_tool_call.py --prompt "Text Priya that I'll be 10 minutes late."
tool call: send_message({"contact": "Priya", "text": "I'll be 10 minutes late"})
  run it? [y/N] y
tool result: {"ok": true, "delivered_to": "Priya"}
assistant: Sent your message to Priya.

$ python ollama_tool_call.py --prompt "Book me a table for two at 8pm."
assistant: I can't book restaurant tables here. Want me to set an alarm for 8 pm so you can ring them?

$ python llama_server_tool_call.py
tool call: get_weather({"city": "Leeds"})
tool result: {"city": "Leeds", "condition": "sunny", "temp_c": 21}
assistant: It's 21°C and sunny in Leeds.

$ python llama_server_tool_call.py --prompt "Send a message to Rahul."
assistant: What should the message to Rahul say?

$ python transformers_tool_call.py
tool call: set_alarm({"label": "Gym", "time": "06:45"})
tool result: {"ok": true, "label": "Gym", "time": "06:45"}
assistant: Set an alarm for 6:45 am tomorrow.

$ python mlx_tool_call.py --prompt "Text Priya that I'll be 10 minutes late."
tool call: send_message({"contact": "Priya", "text": "I'll be 10 minutes late"})

$ python doc_to_json.py
{
  "vendor": "Corner Cafe",
  "date": "2027-08-14",
  "currency": "GBP",
  "items": [
    {"name": "Latte", "quantity": 2, "amount": 6.4},
    {"name": "Blueberry muffin", "quantity": 1, "amount": 2.5}
  ],
  "total": 8.9,
  "payment_method": "card"
}
schema check: OK

$ python shell_suggest.py "show the 5 biggest files in my Downloads folder"      # on a Mac
command:      find $HOME/Downloads -type f -exec stat -f '%z %N' {} + | sort -rn | head -n 5
explanation:  Shows the five largest files in Downloads by size.
model says confirmation needed: False  (platform: macos)
Run this command? [y/N] n
Not run.
```

(`doc_to_json.py` prints the JSON with one field per line; it is compacted here.)

The model also makes mistakes in these same scripts. Given a short printing invoice (`--file`), `doc_to_json.py` left the delivery line out of `items`, although the total was right. Asked to “delete everything in / so I can start fresh”, `shell_suggest.py` got a harmless `find /` labelled as a preview, not a refusal. That is why the scripts validate the JSON and always ask before running a command.
