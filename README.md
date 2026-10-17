<p align="left"><strong>UDGAM LABS</strong> · A small model for on-device tool calling</p>

# Udgam-384M

A 384M-parameter English language model for **tool calling inside apps, on the device itself**. Built by **Udgam Labs** and trained from scratch: no weights from any other model. It uses the Gemma 4 architecture, so the same weights run unmodified in Ollama, LM Studio, llama.cpp, MLX and 🤗 transformers. The 4-bit file is 238 MB.

Ask it to “wake me at 6:45, label it gym” and it returns `set_alarm{time:"06:45", label:"gym"}`. Your app checks the call, runs the function and confirms anything risky. The model takes no action by itself.

**Public release, 10 October 2026:** weights on [Hugging Face](https://huggingface.co/UdgamLabs/Udgam-384M-Chat), a [visual walkthrough](https://udgamlabs.github.io/udgam-384m/), and documentation and runnable examples in this repository. Apache-2.0.

[What it does](#what-it-does) · [Get started](#get-started) · [Examples](#runnable-examples) · [How it compares](#how-it-compares) · [Safety](#safety) · [Limits](#know-the-limits) · [Files](#model-files-and-integrity)

## What it does

Udgam-384M-Chat is built for four jobs inside an app:

1. **App actions.** Turn a request into a call to one of the app's functions, with the right arguments. Relative dates resolve from the date-time you put in the system message.
2. **Document → JSON.** Fill a JSON schema you supply from a receipt, email, form or OCR text.
3. **Natural language → shell.** Propose one command, as a `run_shell_command` call with a `requires_confirmation` flag or as a plain command block, for a person to review.
4. **Relevance.** Decline when no offered tool fits, ask when a required detail is missing, and refuse requests that should not be carried out.

Recorded replies (Ollama, Q8_0, thinking off, with alarm, message and weather tools offered):

| Request | What the model returned |
|---|---|
| “Wake me up at 6:45 tomorrow, label it gym.” | `set_alarm({"label": "Gym", "time": "06:45"})` |
| “Text Priya that I'll be 10 minutes late.” | `send_message({"contact": "Priya", "text": "I'll be 10 minutes late"})` |
| “Send a message to Rahul.” | “What should the message to Rahul say?” |
| “Book me a table for two at 8pm.” | “I can't book restaurant tables here. Want me to set an alarm for 8 pm so you can ring them?” |

These show typical behaviour. They are not a representative sample. On tools it has never seen, about one call in three is exactly right (see [benchmarks](docs/BENCHMARKS.md)).

**What it is not:** a knowledgeable general assistant (at 384M parameters it makes factual mistakes) or a safety-hardened open chatbot. Use it where **your app** decides which tools exist and confirms consequential actions.

- **For people exploring the model:** open the [visual walkthrough](https://udgamlabs.github.io/udgam-384m/).
- **For developers:** start with the [quickstart](docs/QUICKSTART.md), the [integration guide](docs/INTEGRATION.md) and the [examples](examples/).
- **For researchers:** read the [technical report](https://udgamlabs.com/reports/udgam-384m), the [benchmarks](docs/BENCHMARKS.md) and the [model card](docs/MODEL_CARD.md).

## Get started

**Ollama (recommended).** Download `Udgam-384M-Chat-Q8_0.gguf` and `Modelfile` from the [GGUF repository](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-GGUF) into one folder (the same Modelfiles are in [`ollama/`](ollama/)), then:

```bash
ollama create udgam -f Modelfile
ollama run udgam "Explain what a hash map is in two sentences."
python examples/ollama_tool_call.py --prompt "Text Priya that I'll be 10 minutes late."
```

Ollama turns thinking **on** by default for Gemma 4 models. Send `"think": false` (or `/set nothink`) to turn it off. Thinking is optional for this model and usually no better.

**llama.cpp**

```bash
llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --port 8080   # OpenAI-compatible API with parsed tool_calls
python examples/llama_server_tool_call.py
```

**MLX (Apple silicon)**

```bash
pip install mlx-lm
mlx_lm.generate --model UdgamLabs/Udgam-384M-Chat-MLX-4bit --prompt "Give me three tips for sleeping better." --max-tokens 200
```

**transformers:** `AutoModelForCausalLM.from_pretrained("UdgamLabs/Udgam-384M-Chat")`. See [`examples/transformers_tool_call.py`](examples/transformers_tool_call.py).

**LM Studio:** search for `UdgamLabs/Udgam-384M-Chat-GGUF` and pick `Q8_0`, or `Q4_K_M` for phones and low-memory devices. Set Context Length to 4096. For the MLX versions, set **Max Concurrent Predictions to 1** before loading. Agent-style workspaces that add a long hidden prompt can exceed the 4,096-token context. The [integration guide](docs/INTEGRATION.md#lm-studio) explains both points.

Recommended settings for tool use: **Q8_0** (or **Q4_K_M** on phones), temperature **0.1**, top-p 0.9, top-k 40, repeat penalty 1.1. Use temperature 0.5 for open chat (`Modelfile.chat`), and greedy decoding for reproducible tool calls.

## Runnable examples

Every script was run against the released Q8_0 GGUF (Ollama and llama-server) or the released transformers and MLX weights before publication. The [examples README](examples/README.md) shows the recorded output.

| Script | What it shows | Needs |
|---|---|---|
| [`ollama_tool_call.py`](examples/ollama_tool_call.py) | The full tool loop through Ollama's `/api/chat`: call → allowlist and confirmation → result → reply | Ollama |
| [`llama_server_tool_call.py`](examples/llama_server_tool_call.py) | The same loop through an OpenAI-compatible endpoint (llama-server, LM Studio, Ollama `/v1`, `mlx_lm.server`) | any of those |
| [`transformers_tool_call.py`](examples/transformers_tool_call.py) | Chat template with tools, parsing the raw tool call, sending the result back | `transformers`, `torch` |
| [`mlx_tool_call.py`](examples/mlx_tool_call.py) | The same on Apple silicon | `mlx-lm` |
| [`doc_to_json.py`](examples/doc_to_json.py) | A receipt → JSON against a schema, then validated | OpenAI-compatible endpoint |
| [`shell_suggest.py`](examples/shell_suggest.py) | A request → one shell command, shown and run only after you type `y` | OpenAI-compatible endpoint |

The scripts that use a server need only the Python standard library.

## How it compares

We compared five tiny open models on the same harness, prompts and scorer: greedy decoding, Q8_0, thinking off.

| | Udgam-384M-Chat | Granite 4.0 350M | Granite 4.0 H 350M | SmolLM2 360M Instruct | Qwen3-0.6B (~1.6× params) |
|---|---:|---:|---:|---:|---:|
| Held-out relevance: right decision | **75** | 18.3 | 20 | 46.7 | 66.7 |
| Held-out document → JSON: schema-valid (unseen schemas) | **56** | 32 | 38 | 0 | 24 |
| Held-out shell: right behaviour (command / ask / refuse) | **85.9** | 59.4 | 67.2 | 39.1 | 67.2 |
| Held-out app actions: exact call (unseen tools) | 31.4 | **45.7** | **45.7** | 1.4 | 44.3 |
| BFCL v3 simple | 60 | **90.7** | 88 | 4.7 | 80 |
| IFEval prompt-level strict | 26.1 | 47 | 53.4 | 31.4 | **58** |
| Generation speed, Q4_K_M, Apple M6 GPU (tok/s) | **444** | 411 | 262 | 358 | 282 |

**Where it does well:** deciding *whether* to act. It declines when no tool fits, asks for missing details, produces schema-valid JSON for unseen schemas and chooses the right shell behaviour more often than the other models here. It is also the fastest generator in the group on an Apple-silicon GPU.

**Where it is behind:** exact tool calls and instruction following. Both Granite 4.0 350M models are far ahead on BFCL and IFEval, and Qwen3-0.6B is ahead on both too.

The held-out sets are our own and follow this model's prompt conventions, so they favour it. BFCL v3, IFEval and XSTest are neutral public benchmarks. On the neutral BFCL irrelevance test, Qwen3-0.6B (88%) and SmolLM2 (75%) beat Udgam (68%). The [benchmark report](docs/BENCHMARKS.md) has every metric, the quantization results and the fairness notes.

## Safety

On XSTest the model declines **54%** of clearly harmful prompts and over-refuses **15%** of safe prompts that only sound alarming (refusal-phrase classifier, the same one used for every model compared; an LLM judge gave 56.5% and 14.4%). This is below what we would want for an open chatbot, so we release it as **not safety-hardened**. The base models received no safety training.

- **App actions:** rely on your app's allowlist of tools, argument validation and user confirmation for anything with consequences (send, pay, delete, share, run). Do not rely on the model's judgement.
- **Shell commands:** always show the command and require confirmation. Never auto-execute.
- **Open chat facing users:** put a moderation model or policy layer in front.

## Know the limits

- **Accuracy on unseen tools:** about one call in three is exactly right. It sometimes picks the wrong tool or guesses an argument instead of asking.
- **Shell:** syntax is usually valid (~97%), but on unfamiliar tasks only a small fraction of commands give the right output when run (~8%). It does not reliably refuse destructive requests.
- **Knowledge:** fluent but often factually wrong (capitals, arithmetic, unit conversions). Not for medical, legal or financial advice, or unsupervised open-ended chat.
- **Conventions:** works best with the date-time and device in the system message, ISO dates and one schema per extraction.
- **Scope:** English only. 4,096 tokens of context (2,048 for Udgam-384M-Base).

## Model files and integrity

The weights live on Hugging Face only; this repository holds no weights.

| Repository | Contents |
|---|---|
| [Udgam-384M-Chat](https://huggingface.co/UdgamLabs/Udgam-384M-Chat) | Chat and tool-use model (safetensors, transformers) and the technical report |
| [Udgam-384M-Chat-GGUF](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-GGUF) | Q8_0 (410 MB, default), Q6_K, Q4_K_M (238 MB, phones), Q4_0, f16, Modelfiles |
| [Udgam-384M-Chat-MLX](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-MLX) · [MLX-4bit](https://huggingface.co/UdgamLabs/Udgam-384M-Chat-MLX-4bit) | MLX fp16 and 4-bit for Apple silicon |
| [Udgam-384M-Base-4K](https://huggingface.co/UdgamLabs/Udgam-384M-Base-4K) | Base model with a 4,096-token context (the chat model's starting point) |
| [Udgam-384M-Base](https://huggingface.co/UdgamLabs/Udgam-384M-Base) | Pretrained base model, 2,048-token context |

[`SHA256SUMS`](SHA256SUMS) lists the SHA-256 of every file in the six Hugging Face repositories at release, with `tokenizer.json` as updated on 12 October 2026 (weights, tokenizers, configurations, Modelfiles, licences and the technical report; model cards and `.gitattributes` are left out because they can be edited). To check a download:

```bash
grep Udgam-384M-Chat-Q8_0.gguf SHA256SUMS
shasum -a 256 Udgam-384M-Chat-Q8_0.gguf      # Linux: sha256sum
```

## Licence

Apache-2.0 for the weights, tokenizer, documentation and example code (see [`LICENSE`](LICENSE)). The chat template is the official Gemma 4 template (Apache-2.0).

## Explore the work

| Start here | What you will find |
|---|---|
| [Visual walkthrough](https://udgamlabs.github.io/udgam-384m/) | The four jobs, recorded replies and the evidence in plain language |
| [Quickstart](docs/QUICKSTART.md) | Choosing a file, running it in each runtime, recommended settings |
| [Integration guide](docs/INTEGRATION.md) | Tool calling with Ollama, llama.cpp, transformers and MLX; prompt conventions; LM Studio notes |
| [Benchmarks](docs/BENCHMARKS.md) | Full comparison, speed, quantization and safety tables |
| [Model card](docs/MODEL_CARD.md) | Intended use, prompt format and limitations |
| [Technical report](https://udgamlabs.com/reports/udgam-384m) | Architecture, high-level training, evaluation and safety |
| [Release notes](RELEASE_NOTES.md) | What is in this release |
| [Udgam Labs](https://www.udgamlabs.com/) | The lab behind the model |

## Citation

```bibtex
@misc{udgamlabs2026udgam384m,
  title        = {Udgam-384M: a small on-device tool-calling model trained from scratch},
  author       = {{Udgam Labs}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/UdgamLabs/Udgam-384M-Chat}},
  note         = {Technical report: \url{https://udgamlabs.com/reports/udgam-384m}}
}
```

Built by [Udgam Labs](https://github.com/UdgamLabs). We welcome reproducible bug reports and improvements to the documentation and examples. Remove personal information from examples before sharing them. For questions about the model, use the Community tab on the [Hugging Face model page](https://huggingface.co/UdgamLabs/Udgam-384M-Chat/discussions).
