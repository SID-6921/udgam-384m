# Udgam-384M 1.0.0

First public release, 10 October 2026. The weights are on Hugging Face. This repository adds documentation, runnable examples, the Ollama Modelfiles and checksums.

**Update, 12 October 2026:** `tokenizer.json` in the Base, Base-4K, Chat, Chat-MLX and Chat-MLX-4bit repositories now adds `<bos>` to plain-text input (`tokenizer(text)`), as the models were trained. The weights, the GGUF files and chat-template use are unchanged. If you add `<bos>` yourself, pass `add_special_tokens=False`. `SHA256SUMS` lists the updated files.

- **Three models, trained from scratch** (no weights from any other model), each with 384.4M parameters and the Gemma 4 architecture (`Gemma4ForCausalLM`):
  - [Udgam-384M-Base](https://huggingface.co/udgamlabs/Udgam-384M-Base): pretrained base model, 2,048-token context.
  - [Udgam-384M-Base-4K](https://huggingface.co/udgamlabs/Udgam-384M-Base-4K): the base model after a short extra pretraining stage at 4,096 tokens.
  - [Udgam-384M-Chat](https://huggingface.co/udgamlabs/Udgam-384M-Chat): post-trained from Base-4K for chat and tool use, 4,096-token context.
- **Formats:** safetensors for transformers; GGUF f16, Q8_0 (default), Q6_K, Q4_K_M (phones, 238 MB) and Q4_0 ([Chat-GGUF](https://huggingface.co/udgamlabs/Udgam-384M-Chat-GGUF)); MLX fp16 and 4-bit ([Chat-MLX](https://huggingface.co/udgamlabs/Udgam-384M-Chat-MLX), [Chat-MLX-4bit](https://huggingface.co/udgamlabs/Udgam-384M-Chat-MLX-4bit)). Each base repository includes f16 and Q8_0 GGUF files.
- **Four jobs:** app actions (function calls), document → JSON, natural language → shell (for review), and relevance (decline, ask or refuse).
- **Runs in** Ollama, LM Studio, llama.cpp, MLX and transformers with the standard Gemma 4 chat template and tool-call format.
- **Measured:** leads the tiny open models we compared on our held-out relevance, document → JSON and shell-behaviour tests, and on generation speed on an Apple-silicon GPU. It is behind Granite 4.0 350M and Qwen3-0.6B on BFCL v3 and IFEval. See [benchmarks](docs/BENCHMARKS.md).
- **Safety:** declines 54% of clearly harmful XSTest prompts and over-refuses 15% of safe ones (refusal-phrase classifier; an LLM judge gave 56.5% and 14.4%). Released as **not safety-hardened**.
- **Licence:** Apache-2.0.

## In this repository

| Path | Contents |
|---|---|
| `README.md` | Overview, quick start, comparison highlights, safety and limits |
| `docs/` | [Quickstart](docs/QUICKSTART.md), [integration guide](docs/INTEGRATION.md), [benchmarks](docs/BENCHMARKS.md), [model card](docs/MODEL_CARD.md), [technical report](https://udgamlabs.com/reports/udgam-384m), and the [visual walkthrough](https://udgamlabs.github.io/udgam-384m/) (`docs/index.html`) |
| `examples/` | Six runnable scripts plus a small Gemma 4 tool-call parser |
| `ollama/` | `Modelfile` (tool use, temperature 0.1) and `Modelfile.chat` (open chat, temperature 0.5), identical to the Hugging Face copies |
| `SHA256SUMS` | SHA-256 of every published file in the six Hugging Face repositories, except model cards and `.gitattributes` |

## Known limits

About one call in three is exactly right on tools the model has never seen, so consequential actions need app-side allowlists, validation and confirmation. Shell commands are usually valid syntax but rarely correct on unfamiliar tasks; always confirm before running them. English only; 4,096-token context. The model is fluent but often factually wrong, and not suited to medical, legal or financial advice.
