# Integration guide

How to use Udgam-384M-Chat as the language layer of an app: tool calling with Ollama, llama.cpp, transformers and MLX, the prompt conventions it works best with, the document → JSON and shell patterns, and notes for LM Studio.

## The loop

```text
 person ──request──▶ your app ──system message + tools + request──▶ Udgam-384M-Chat
                        ▲                                                │
                        │      a tool call  ── or ──  a reply in words (ask / decline)
                        │                                                │
                        └── validate → confirm if risky → run tool → send result back ──▶ final reply
```

The model only **proposes**. Your app owns the list of tools, checks every call (is the tool allowed right now, are the arguments valid), asks the person before anything consequential, runs the tool and sends the result back. When no tool fits or a detail is missing, the model answers in words instead of calling a tool, so always handle a reply with no `tool_calls`.

## Prompt conventions

The model was trained with these conventions and does best with them:

- **Put the current date-time, time zone and device in the system message**, so “tomorrow” or “next Friday” resolve correctly:

  ```text
  Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)
  Device: Android phone
  ```

  For shell requests, add the OS and shell, for example `Device: Mac laptop` and `OS: macOS · Shell: zsh`.
- **Offer only the tools the person may use right now.** Fewer, well-described tools give better choices.
- **Describe argument formats in the schema**, for example `"description": "HH:MM 24h"`. The model prefers ISO dates and normalised values.
- **For document → JSON, give one schema (or field list) per request** and say “Reply with JSON only”.
- **Thinking is optional** and gives no consistent gain for this model. Turn it off for tool use (see each runtime below).

Tools use the usual function-calling format with a JSON schema for the parameters:

```json
{"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm at a local time.",
  "parameters": {"type": "object", "properties": {
    "time": {"type": "string", "description": "HH:MM 24h"},
    "label": {"type": "string", "description": "Optional label"}}, "required": ["time"]}}}
```

## Ollama

Create the model with the provided [`Modelfile`](../ollama/Modelfile). It uses `RENDERER gemma4` and `PARSER gemma4`, so Ollama renders tools into the prompt and parses tool calls for you.

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "udgam", "stream": false, "think": false,
  "messages": [
    {"role": "system", "content": "Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)\nDevice: Android phone"},
    {"role": "user", "content": "Wake me up at 6:45 tomorrow, label it gym."}],
  "tools": [{"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm at a local time.",
     "parameters": {"type": "object", "properties": {"time": {"type": "string", "description": "HH:MM 24h"},
                    "label": {"type": "string", "description": "Optional label"}}, "required": ["time"]}}}]}'
```

The reply carries `message.tool_calls[].function.name` and `arguments` **as a JSON object**:

```json
{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "set_alarm", "arguments": {"label": "Gym", "time": "06:45"}}}]}
```

Append that assistant message, then the result as `{"role": "tool", "tool_name": "set_alarm", "content": "<result JSON>"}`, and call `/api/chat` again for the final reply. **Thinking is on by default** in Ollama for Gemma 4 models, so send `"think": false`. Full script: [`examples/ollama_tool_call.py`](../examples/ollama_tool_call.py).

## llama.cpp server (and other OpenAI-compatible servers)

```bash
llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --port 8080
```

`--jinja` is required for parsed tool calls. Send a standard `/v1/chat/completions` request with `tools`. Here `arguments` arrives **as a JSON string**:

```json
{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "…",
  "function": {"name": "set_alarm", "arguments": "{\"label\":\"Gym\",\"time\":\"06:45\"}"}}]}
```

Send the result back as `{"role": "tool", "tool_call_id": "<id>", "content": "<result JSON>"}`. To turn thinking off per request, add `"chat_template_kwargs": {"enable_thinking": false}`, or start the server with `--reasoning off`. Pass sampling settings in the request (`temperature`, `top_p`, `top_k`, `repeat_penalty`), because server defaults may differ.

The same request works, unchanged, with:

| Server | Base URL | `model` field |
|---|---|---|
| llama-server | `http://localhost:8080/v1` | any |
| Ollama (OpenAI-compatible) | `http://localhost:11434/v1` | `udgam` |
| `mlx_lm.server` | `http://localhost:<port>/v1` | the id it was started with, e.g. `udgamlabs/Udgam-384M-Chat-MLX-4bit` |
| LM Studio local server | `http://localhost:1234/v1` | the model id shown in LM Studio |

Full script: [`examples/llama_server_tool_call.py`](../examples/llama_server_tool_call.py). We tested it against llama-server, Ollama `/v1` and `mlx_lm.server` (mlx-lm 0.32), and all three returned parsed `tool_calls`.

## transformers

```python
inputs = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                                 return_dict=True, return_tensors="pt")   # thinking off unless enable_thinking=True
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
raw = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
# <|tool_call>call:set_alarm{label:<|"|>Gym<|"|>,time:<|"|>06:45<|"|>}<tool_call|><|tool_response>
```

Decode with `skip_special_tokens=False`, then parse the call with [`examples/gemma4_tools.py`](../examples/gemma4_tools.py) (`parse_tool_calls(raw)`, standard library only). To continue, append `{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": ..., "arguments": {...}}}]}` and `{"role": "tool", "name": ..., "content": "<result JSON>"}`, then generate again. Do not set `repetition_penalty` in transformers: it also penalises prompt tokens, including tool names. Full script: [`examples/transformers_tool_call.py`](../examples/transformers_tool_call.py).

## MLX

`mlx_lm.server` returns parsed `tool_calls` (see the table above). In Python, render the prompt with the tokenizer's chat template and parse the raw text the same way as in transformers:

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
model, tokenizer = load("udgamlabs/Udgam-384M-Chat-MLX-4bit")
prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
raw = generate(model, tokenizer, prompt=prompt, max_tokens=256, sampler=make_sampler(temp=0.0))
```

Full script: [`examples/mlx_tool_call.py`](../examples/mlx_tool_call.py).

## The raw prompt format

You only need this if you build prompts by hand. It is the standard Gemma 4 chat format, and the template is identical to the official one:

```text
<bos><|turn>system
Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)
Device: Android phone<|tool>declaration:set_alarm{description:<|"|>Set an alarm at a local time.<|"|>,parameters:{properties:{label:{description:<|"|>Optional label<|"|>,type:<|"|>STRING<|"|>},time:{description:<|"|>HH:MM 24h<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>time<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
Wake me up at 6:45 tomorrow, label it gym.<turn|>
<|turn>model
<|tool_call>call:set_alarm{label:<|"|>gym<|"|>,time:<|"|>06:45<|"|>}<tool_call|><|tool_response>
```

The model stops at `<|tool_response>`. With thinking on, `<|think|>` is added after `<|turn>system\n` and the model may write `<|channel>thought\n…<channel|>` before answering.

## Document → JSON

Put one schema in the system message with a short instruction, and the document in the user message:

```text
system: Current date-time: Saturday, 2027-08-14 18:05 (Europe/London, UTC+01:00)
        Device: iPhone

        Expense tracker. Turn the receipt into a JSON object that matches the schema below. Reply with JSON only.
        {"type":"object","properties":{"vendor":{"type":"string"},"date":{"type":"string","description":"YYYY-MM-DD"}, …}}
user:   save this receipt please

        RECEIPT
        Corner Cafe, 12 High St
        14/08/2027 …
```

A plain field list also works (“Extract: vendor (string), total (number), date (YYYY-MM-DD)”). Fields the document does not state should come back as `null`. **Validate the result against your schema** before storing it: the model produces schema-valid JSON for unseen schemas about 56–60% of the time. Full script: [`examples/doc_to_json.py`](../examples/doc_to_json.py).

## Natural language → shell

Offer a `run_shell_command` tool. Its arguments are `command`, a one-sentence `explanation`, a `requires_confirmation` flag (true when the command modifies or deletes files, changes system state, installs software, uses sudo or changes remote state) and `platform` (`macos`, `linux` or `any`). Without the tool, the model answers with a plain command block. Include the OS and shell in the system message.

**Always show the command and require confirmation, whatever the flag says.** Syntax is usually valid, but on unfamiliar tasks only a small fraction of commands give the right output, and the model does not reliably refuse destructive requests. Full script: [`examples/shell_suggest.py`](../examples/shell_suggest.py), which runs a command only after you type `y`.

## LM Studio

- **GGUF:** pick `Q8_0` (or `Q4_K_M`), set **Context Length 4096** and the sampling settings from the [quickstart](QUICKSTART.md#3-recommended-settings). The GGUF carries the Gemma 4 chat template, so the prompt format is set automatically. For tool use, call LM Studio's OpenAI-compatible server (`/v1/chat/completions` with `tools`). If your version does not parse Gemma 4 tool calls into `tool_calls`, parse the text with `gemma4_tools.py`.
- **MLX versions: set Max Concurrent Predictions to 1.** Before loading, open the model's load settings (the gear icon, or turn on “Manually choose model load parameters”). With the default parallel setting, LM Studio's MLX engine (tested with mlx-llm 1.13.1) fails on text-only Gemma 4 models with `KeyError: 'text_config'`. The GGUF version needs no change.
- **Agent workspaces and the 4K context.** Some LM Studio workspace modes add a hidden system and tool prompt of roughly 5,000 tokens. That is more than this model's 4,096-token context, so the request fails or the conversation is truncated. Use plain chat, the local server API, or llama-server's built-in web UI (`llama-server … --jinja`, then open `http://localhost:8080`).

## Before you ship

- Keep an **allowlist** of the tools a person may use, and reject calls to anything else.
- **Validate arguments** (types, ranges, formats) and resolve names such as contacts yourself.
- **Confirm** anything that sends, pays, deletes, shares or runs. Never auto-execute shell commands.
- Handle **plain-text replies** (questions and refusals) as a normal outcome.
- Put a **moderation layer** in front of open chat facing users. The model is not safety-hardened (see [benchmarks](BENCHMARKS.md#safety)).
- Remember the context is **4,096 tokens**, including tool declarations.
