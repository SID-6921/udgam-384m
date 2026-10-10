> A copy of the model card published at [huggingface.co/UdgamLabs/Udgam-384M-Chat](https://huggingface.co/UdgamLabs/Udgam-384M-Chat) on 10 October 2026. The Hugging Face page is the current version. The GGUF and MLX cards add a short file list and, for MLX, the LM Studio note in the [integration guide](INTEGRATION.md#lm-studio).

# Udgam-384M-Chat

**A 384M-parameter English model for on-device tool calling, trained from scratch by UdgamLabs.** It is a
decoder-only transformer compatible with the Gemma 4 architecture (`Gemma4ForCausalLM`), so it runs unmodified in
Ollama, LM Studio, llama.cpp, MLX and 🤗 transformers. Context: 4,096 tokens. The 4-bit GGUF is 238 MB. Built for four
jobs inside an app:

1. **App actions**: turn a request into a function call. "wake me at 6:45, label it gym" →
   `set_alarm{time:"06:45", label:"gym"}`.
2. **Document → JSON**: fill a given JSON schema from a receipt, email, form or OCR text.
3. **Natural language → shell**: propose one command (as a `run_shell_command` call with `requires_confirmation`, or
   as a plain command block) for a person to review.
4. **Relevance**: notice when no offered tool fits (decline), when a required detail is missing (ask), or when a
   request should be refused.

**What it is not:** a knowledgeable general assistant (at 384M parameters it makes factual mistakes), or a
safety-hardened open chatbot. Use it where *your app* decides which tools exist and confirms risky actions.

**Model family:** [Udgam-384M-Base](https://huggingface.co/UdgamLabs/Udgam-384M-Base) (2,048 context) · [Udgam-384M-Base-4K](https://huggingface.co/UdgamLabs/Udgam-384M-Base-4K) · [Udgam-384M-Chat](https://huggingface.co/UdgamLabs/Udgam-384M-Chat) · [GGUF](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-GGUF) · [MLX](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-MLX) · [MLX 4-bit](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-MLX-4bit) · **[Technical report](https://huggingface.co/UdgamLabs/Udgam-384M-Chat/blob/main/TECHNICAL_REPORT.md)**

| | |
|---|---|
| Developer | UdgamLabs |
| Parameters | 384M (20 layers, hidden size 1,024) |
| Architecture | decoder-only transformer, compatible with Gemma 4 (`Gemma4ForCausalLM`); sliding-window + global attention |
| Context | 4,096 tokens |
| Language | English |
| Training | pretrained from scratch on roughly 8B tokens of licensed public and synthetic text, then post-trained with supervised fine-tuning for chat and tool use |
| Licence | Apache-2.0 |

## Quick start

### Ollama (recommended)

Download a GGUF from the [GGUF repo](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-GGUF) and create a model with its `Modelfile` (it uses
Ollama's native Gemma 4 renderer and parser, so tool calls and the optional thinking mode work as trained):

```
FROM ./Udgam-384M-Chat-Q8_0.gguf
TEMPLATE {{ .Prompt }}
RENDERER gemma4
PARSER gemma4
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64
PARAMETER num_predict 1024
PARAMETER num_ctx 4096
```

```bash
ollama create udgam -f Modelfile
ollama run udgam "Explain what a hash map is in two sentences."
curl http://localhost:11434/api/chat -d '{
  "model": "udgam", "stream": false, "think": false,
  "messages": [
    {"role": "system", "content": "Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)\nDevice: Android phone"},
    {"role": "user", "content": "Wake me up at 6:45 tomorrow, label it gym."}],
  "tools": [{"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm at a local time.",
     "parameters": {"type": "object", "properties": {"time": {"type": "string", "description": "HH:MM 24h"},
                    "label": {"type": "string", "description": "Optional label"}}, "required": ["time"]}}}]}'
# -> "tool_calls": [{"function": {"name": "set_alarm", "arguments": {"label": "Gym", "time": "06:45"}}}]
```

Thinking is **on by default** in Ollama for Gemma 4 models; send `"think": false` (or `/set nothink`) to turn it off.
For this model thinking is optional and usually not better (see the tables).

### LM Studio

1. Search for `UdgamLabs/Udgam-384M-Chat-GGUF` (or drop the `.gguf` into LM Studio's models folder) and pick `Q8_0`
   (default) or `Q4_K_M` (phones / low memory).
2. The GGUF carries the Gemma 4 chat template, so the prompt format is set automatically. Set **Context Length 4096**,
   temperature 0.1, top-p 0.9, top-k 40, repeat penalty 1.1.
3. For tool use, call LM Studio's OpenAI-compatible server (`/v1/chat/completions` with `tools`). If your version does
   not parse Gemma 4 tool calls into `tool_calls`, read them from the text:
   `<|tool_call>call:NAME{key:<|"|>value<|"|>,...}<tool_call|>`.

### llama.cpp

```bash
llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja                  # OpenAI-compatible API, parsed tool_calls
llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --reasoning off  # same, thinking off
llama-cli    -m Udgam-384M-Chat-Q8_0.gguf -c 4096 -cnv                     # interactive chat
```

With the `set_alarm` tool above, `/v1/chat/completions` returns
`tool_calls: [{"name": "set_alarm", "arguments": "{\"label\":\"Gym\",\"time\":\"06:45\"}"}]` (tested with llama.cpp
build d7a695e).

### MLX (Apple silicon)

```bash
pip install mlx-lm
mlx_lm.generate --model UdgamLabs/Udgam-384M-Chat-MLX-4bit --prompt "Give me three tips for sleeping better." --max-tokens 200
```

### transformers

```python
# tested with transformers 5.18
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("UdgamLabs/Udgam-384M-Chat")
model = AutoModelForCausalLM.from_pretrained("UdgamLabs/Udgam-384M-Chat")
tools = [{"type": "function", "function": {"name": "set_alarm", "description": "Set an alarm at a local time.",
          "parameters": {"type": "object", "properties": {"time": {"type": "string", "description": "HH:MM 24h"},
                         "label": {"type": "string"}}, "required": ["time"]}}}]
msgs = [{"role": "user", "content": "Wake me up at 6:45, label it gym."}]
inputs = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)   # thinking off unless enable_thinking=True
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False))
# <|tool_call>call:set_alarm{label:<|"|>Gym<|"|>,time:<|"|>06:45<|"|>}<tool_call|><|tool_response>
```

Use fp32 (or llama.cpp) if you need bit-exact greedy results; bf16 generation in transformers depends on batch shape.

## Prompt format

The standard Gemma 4 chat format (the template is identical to the official one). A tool turn:

```
<bos><|turn>system
Current date-time: Tuesday, 2027-08-17 09:12 (Europe/London, UTC+01:00)
Device: Android phone<|tool>declaration:set_alarm{description:<|"|>Set an alarm at a local time.<|"|>,parameters:{properties:{label:{description:<|"|>Optional label<|"|>,type:<|"|>STRING<|"|>},time:{description:<|"|>HH:MM 24h<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>time<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
Wake me up at 6:45 tomorrow, label it gym.<turn|>
<|turn>model
<|tool_call>call:set_alarm{label:<|"|>gym<|"|>,time:<|"|>06:45<|"|>}<tool_call|><|tool_response>
```

The model stops at `<|tool_response>`; your app runs the tool, sends the result back as a `tool` message, and the
model writes the final reply. With thinking on, `<|think|>` is added after `<|turn>system\n` and the model may write
`<|channel>thought\n…<channel|>` first.

**Tips:** put the current date-time, time zone and device in the system message (as above) so relative dates resolve;
give one JSON schema (or field list) for document → JSON; offer only the tools the user may use right now.

Example outputs (Ollama, Q8_0, thinking off): "Text Priya that I'll be 10 minutes late." →
`send_message{contact:"Priya", text:"I'll be 10 minutes late"}`; "Send a message to Rahul." → "What should the message
to Rahul say?"; "Book me a table for two at 8pm." (only alarm / message / weather tools offered) → "I can't make
restaurant bookings here. Want me to set an alarm for 8 pm so you can ring them?"; a receipt →
`{"vendor": "Corner Cafe", "total": 8.9, "date": "2027-08-14"}`. A typical mistake: "What's the boiling point of water
in Fahrenheit?" → "about 37 degrees Celsius, roughly 100 degrees Fahrenheit" (wrong).

## Recommended quantization and sampling

**Quantization:** use **Q8_0** by default (lossless within noise) and **Q4_K_M** on phones / low-memory devices (small
loss, see below). Avoid Q4_0 unless your runtime lacks K-quants.

| setting | tool / app use (default `Modelfile`) | open chat (`Modelfile.chat`) |
|---|---|---|
| temperature | 0.1 | 0.5 |
| top_p / top_k | 0.9 / 40 | 0.9 / 40 |
| repeat_penalty (last 64 tokens) | 1.1 | 1.1 |
| num_predict / num_ctx | 1024 / 4096 | 1024 / 4096 |

Why: temperature is what costs tool accuracy (BFCL simple in Ollama: 60% greedy, 59% at T 0.1, 52% at 0.5, 46% at 0.7),
while a mild repeat penalty removes repetition loops without hurting tool calls. In transformers,
`generation_config.json` sets T 0.1 / top-p 0.9 / top-k 40 (no `repetition_penalty`: transformers applies it to the
prompt too, which would penalise tool names). Use greedy decoding for reproducible tool calls.

## Evaluation

Automatic checkers, greedy decoding with llama.cpp on a Mac. The **held-out task evals** are our own test sets (274
items) built on tools, document types and shell tasks that were excluded from all training data, plus a held-out date
range. They follow this model's prompt conventions, so they favour it; **BFCL v3, IFEval and XSTest are neutral public
benchmarks.**

### Compared with other tiny open models (same harness, same prompts)

Each model used its own chat template and native tool-call format; outputs were scored by the same code.

| metric (greedy, thinking off) | Udgam-384M-Chat Q8_0 | Udgam-384M-Chat Q4_K_M | Granite 4.0 350M | Granite 4.0 H 350M | SmolLM2 360M Instruct | Qwen3 0.6B (reference, ~2× params) |
|---|---|---|---|---|---|---|
| Held-out app actions (unseen tools): exact call | 31.4 | 28.6 | 45.7 | 45.7 | 1.4 | 44.3 |
| Held-out app actions: right decision (call / no call) | 74 | 74 | 70 | 72 | 66 | 70 |
| Held-out app actions: asks when a detail is missing | 20 | 25 | 0 | 5 | 0 | 20 |
| Held-out relevance: right decision | 75 | 68.3 | 18.3 | 20 | 46.7 | 66.7 |
| Held-out relevance: declines when no tool fits | 74 | 66 | 2 | 4 | 44 | 66 |
| Held-out document → JSON (unseen schemas): schema-valid | 56 | 60 | 32 | 38 | 0 | 24 |
| Held-out document → JSON: field F1 | 68.7 | 69.7 | 52.7 | 45 | 9.3 | 63.7 |
| Held-out shell (unseen tasks): right behaviour (command / ask / refuse) | 85.9 | 85.9 | 59.4 | 67.2 | 39.1 | 67.2 |
| Held-out shell: bash syntax OK | 96.6 | 96.6 | 97.7 | 98 | 100 | 95.1 |
| Held-out shell: output matches when executed (n=26) | 7.7 | 7.7 | 3.8 | 7.7 | 0 | 0 |
| BFCL v3 simple (n=150) | 60 | 62.7 | 90.7 | 88 | 4.7 | 80 |
| BFCL v3 multiple (n=100) | 67 | 64 | 88 | 90 | 5 | 76 |
| BFCL v3 parallel (n=50) | 46 | 50 | 82 | 76 | 0 | 64 |
| BFCL v3 irrelevance (n=150) | 68 | 65.3 | 55.3 | 66.7 | 75.3 | 88 |
| IFEval prompt-level strict (n=541) | 26.1 | 25.5 | 47 | 53.4 | 31.4 | 58 |
| IFEval instruction-level strict | 40.4 | 39.7 | 58.9 | 65.6 | 42.4 | 68.7 |
| XSTest unsafe: declined (higher = safer) | 54.0 | 54.0 | 80.0 | 92.5 | 27.5 | 17.0 |
| XSTest safe: over-refused (lower is better) | 14.8 | 14.8 | 37.2 | 44.8 | 10.0 | 3.2 |

Speed and memory (llama.cpp `llama-bench`, Q4_K_M, 512-token prompt, 128 generated tokens, Apple M6 Mac mini):

| model (Q4_K_M GGUF) | file | params | Mac GPU (Metal) prompt / generate, tok/s | 4 CPU threads prompt / generate, tok/s | RAM, CPU run |
|---|---|---|---|---|---|
| **Udgam-384M-Chat** | 237 MB | 384M | 14,191 / **444** | 983 / **365** | 323 MB |
| Udgam-384M-Chat (Q8_0) | 409 MB | 384M | 14,561 / 312 | 1,271 / 297 | 480 MB |
| Granite 4.0 350M | 233 MB | 352M | 17,626 / 411 | 1,187 / 348 | 349 MB |
| Granite 4.0 H 350M | 219 MB | 340M | 5,175 / 262 | 966 / 307 | 352 MB |
| SmolLM2 360M Instruct | 269 MB | 362M | 14,175 / 358 | 1,378 / 253 | 238 MB |
| Qwen3 0.6B | 478 MB | 752M | 9,774 / 282 | 657 / 213 | 579 MB |

RAM = macOS peak memory footprint of the CPU run. Phones are slower; the relative order should be similar on CPU.

**Where Udgam does well:** deciding *whether* to act: declining when no offered tool fits (held-out relevance 75% vs
18–20% for both Granite models, which call a tool almost every time), asking for a missing detail (20–25% vs 0–5%),
document → JSON against an unseen schema (schema-valid 56–60% vs 24–38%) and shell behaviour (86% vs 59–67%). It is
also the fastest generator in the group on the Apple-silicon GPU (444 tokens/s at Q4_K_M).

**Where it is behind:** exact tool calls and instruction following. Both Granite 4.0 350M models are far ahead on BFCL
(simple 88–91% vs 60%, parallel 76–82% vs 46%), on held-out unseen-tool exact match (46% vs 31%) and on IFEval (47–53%
vs 26%). Qwen3-0.6B (about twice the parameters) is also ahead on BFCL and IFEval. Shell commands are rarely *correct*
when executed for any model here (0–8%).

**Fairness notes:**
- The held-out task sets use this model's prompt conventions (date/device system headers, normalised argument values
  such as ISO dates); other models lose some points to convention rather than ability. BFCL, IFEval and XSTest are
  neutral.
- **On the neutral BFCL irrelevance test, Qwen3-0.6B (88%) and SmolLM2 (75%) beat Udgam (68%)**; Udgam's relevance
  advantage shows mainly on our own held-out relevance set.
- Granite and Qwen3 used their native templates with tools; SmolLM2-360M's template has no tool support, so it got the
  function-calling system prompt from its model card. It mostly copied the example call from that prompt, so its tool
  scores (≈0–5%) mean "not a tool model", not "bad model".
- Qwen3 ran with thinking off; Udgam with thinking off; Granite and SmolLM2 have no thinking mode. All greedy, Q8_0,
  same token limits, same scorer.
- XSTest uses one refusal-phrase classifier for every model (first 300 characters); on this model it agreed with an
  LLM judge within 3 points.
- Not compared: FunctionGemma-270M and Gemma 3 270M (gated licence terms on Hugging Face; not evaluated).

### Quantization

| metric (thinking off / on) | f16 (reference) | Q8_0 | Q6_K | Q4_K_M + imatrix | Q4_K_M (no imatrix) | Q4_0 + imatrix |
|---|---|---|---|---|---|---|
| Held-out app actions: exact call | 32.9 / 25.7 | 31.4 / 24.3 | 32.9 / 22.9 | 28.6 / 28.6 | 30 / 18.6 | 28.6 / 21.4 |
| Held-out app actions: right decision | 74 / 69 | 74 / 67 | 76 / 67 | 74 / 74 | 77 / 71 | 74 / 76 |
| Held-out app actions: asks when a detail is missing | 20 / 25 | 20 / 20 | 30 / 25 | 25 / 25 | 30 / 20 | 25 / 45 |
| Held-out relevance: right decision | 76.7 / 63.3 | 75 / 65 | 76.7 / 61.7 | 68.3 / 63.3 | 78.3 / 68.3 | 71.7 / 71.7 |
| Held-out relevance: declines when no tool fits | 76 / 62 | 74 / 64 | 76 / 62 | 66 / 60 | 78 / 68 | 70 / 72 |
| Held-out document → JSON: schema-valid | 58 / 58 | 56 / 56 | 58 / 58 | 60 / 64 | 60 / 52 | 44 / 46 |
| Held-out document → JSON: field F1 | 68.8 / 71.2 | 68.7 / 72.6 | 70.3 / 70.7 | 69.7 / 68.2 | 70 / 68.1 | 69.7 / 68.9 |
| Held-out shell: right behaviour | 85.9 / 82.8 | 85.9 / 79.7 | 87.5 / 76.6 | 85.9 / 85.9 | 85.9 / 79.7 | 84.4 / 82.8 |
| Held-out shell: bash syntax OK | 96.6 / 100 | 96.6 / 100 | 98.3 / 100 | 96.6 / 100 | 100 / 98.2 | 96.6 / 100 |
| Held-out shell: output matches when executed (n=26) | 7.7 / 7.7 | 7.7 / 7.7 | 7.7 / 7.7 | 7.7 / 7.7 | 7.7 / 7.7 | 7.7 / 7.7 |
| BFCL v3 simple (n=150) | 60 / 60.7 | 60 / 60.7 | 60 / 60.7 | 62.7 / 61.3 | 60 / 58.7 | 58 / 58 |
| BFCL v3 multiple (n=100) | 67 / 67 | 67 / 67 | 65 / 63 | 64 / 63 | 67 / 67 | 57 / 56 |
| BFCL v3 parallel (n=50) | 50 / 46 | 46 / 44 | 48 / 48 | 50 / 50 | 48 / 48 | 44 / 44 |
| BFCL v3 irrelevance (n=150) | 70 / 72 | 68 / 70 | 70.7 / 71.3 | 65.3 / 66 | 66 / 70 | 68.7 / 69.3 |
| IFEval prompt-level strict (n=541) | 26.2 / 26.2 | 26.1 / 26.1 | 27.2 / 27.5 | 25.5 / 26.2 | 24.6 / 24.4 | 27.4 / 28.8 |
| IFEval instruction-level strict | 40.5 / 40.2 | 40.4 / 40.5 | 40.9 / 41.5 | 39.7 / 40.5 | 38.4 / 38.6 | 41 / 42.2 |

Perplexity on held-out chat and tool-use text: f16 3.193, Q8_0 +0.07%, Q6_K +0.5%, Q4_K_M (imatrix) +4.2%, Q4_0
(imatrix) +8.3%.

## Safety

XSTest (200 clearly unsafe + 250 safe-but-alarming prompts), greedy, thinking off: **the model declines about 54–57%
of clearly harmful prompts and over-refuses about 15% of safe ones** (Q8_0 and Q4_K_M identical). It rarely declines
XSTest's discriminatory-claim, privacy and historical-atrocity prompts, and it over-refuses idioms and homonyms
("kill time", "shoot a photo"). This is **below what we would want for an open chatbot**; treat the model as not
safety-hardened.

- **For app actions:** rely on your app's allowlist of tools, argument validation and user confirmation for anything
  with consequences (send, pay, delete, share, run); do not rely on the model's judgement.
- **For shell commands:** always show the command and require confirmation; never auto-execute.
- **For user-facing open chat:** put a moderation model or policy layer in front.

## Intended use and limitations

**Good fits:** on-device assistants that map requests to a fixed set of app functions; form filling and receipt /
email → JSON with a given schema; suggesting shell commands for a person to review; routing (call / ask / decline).

**Use with care / not intended for:**
- **Unsupervised actions.** It picks a wrong tool or argument value often enough (exact match on unseen tools is about
  one in three) that any consequential action needs app-side allowlists, validation and confirmation.
- **Shell commands:** syntax is usually valid, but on unfamiliar tasks only a small fraction produce the right output,
  and it does not reliably refuse destructive requests.
- **Unsupervised open-ended chat or advice:** fluent but shallow and often factually wrong (capitals, arithmetic, unit
  conversions). Not for medical, legal or financial use.
- **Safety:** see above (≈54–57% of harmful prompts declined, ≈15% over-refusal).
- **Context:** 4,096 tokens. English only.
- It prefers its own argument conventions (ISO dates, normalised values) and sometimes guesses a value instead of
  asking.

## Training

Pretrained from scratch on roughly 8B tokens of licensed public and synthetic text (base:
[Udgam-384M-Base-4K](https://huggingface.co/UdgamLabs/Udgam-384M-Base-4K)), then post-trained with supervised fine-tuning for chat and tool use.
See the [technical report](https://huggingface.co/UdgamLabs/Udgam-384M-Chat/blob/main/TECHNICAL_REPORT.md) for an overview.

## Licence

Apache-2.0 (weights and tokenizer); see `LICENSE`. The chat template is the official Gemma 4 template (Apache-2.0).

## Citation

```bibtex
@misc{udgamlabs2026udgam384m,
  title        = {Udgam-384M: a small on-device tool-calling model trained from scratch},
  author       = {{UdgamLabs}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/UdgamLabs/Udgam-384M-Chat}},
  note         = {Technical report: TECHNICAL_REPORT.md in the Udgam-384M-Chat repository}
}
```

## Contact

UdgamLabs. Please use the Community (discussions) tab of the model page for questions, issues and feedback.
