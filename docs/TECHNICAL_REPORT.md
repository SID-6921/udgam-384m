# Udgam-384M: a small on-device tool-calling model trained from scratch

*UdgamLabs, October 2026. Technical report for Udgam-384M-Base, Udgam-384M-Base-4K and Udgam-384M-Chat.*

## 1. Overview

Udgam-384M is a family of 384M-parameter English language models built by UdgamLabs for one job: acting as the
language layer of an app on the device itself. Given the tools an app offers, the chat model turns a user's request
into a structured function call, extracts the fields of a document into a JSON schema, proposes a shell command for a
person to review, or recognises that it should ask a question or decline. All three models were trained from scratch
(no weights from any other model) and are released under Apache-2.0.

| Model | What it is | Context |
|---|---|---|
| **Udgam-384M-Base** | Pretrained base model: continues text | 2,048 |
| **Udgam-384M-Base-4K** | Base model with a short extra pretraining stage at 4,096 tokens | 4,096 |
| **Udgam-384M-Chat** | Post-trained for chat and tool use (fine-tuned from Base-4K) | 4,096 |

The chat model is also published as GGUF (f16, Q8_0, Q6_K, Q4_K_M, Q4_0) and MLX (fp16 and 4-bit). It runs in Ollama,
LM Studio, llama.cpp, MLX and transformers; the Q4_K_M file is 238 MB and generates about 440 tokens per second on an
Apple M6 GPU.

## 2. Architecture

Udgam-384M is a decoder-only transformer compatible with the Gemma 4 architecture (`Gemma4ForCausalLM`). We chose
this architecture so that the same weights load without conversion code in every major runtime, and we verified that
transformers, llama.cpp and MLX give the same greedy outputs before releasing.

- 20 layers, hidden size 1,024, 16 query heads and 4 key-value heads (grouped-query attention).
- Attention alternates three sliding-window layers (window 512) with one global layer, which keeps memory and
  latency low on long prompts.
- GeGLU feed-forward blocks (4,096 wide) and a final logit soft-cap.
- A 32,768-token vocabulary that includes the Gemma 4 chat-format special tokens, so the official Gemma 4 chat
  template and tool-call format work as-is.
- 384.4M parameters in total (317.3M in the transformer layers, plus untied input and output embeddings).

## 3. Training (high level)

The base model was pretrained from scratch on roughly 8B tokens of licensed public and synthetic text at a sequence
length of 2,048. A short continued-pretraining stage at 4,096 tokens produced Udgam-384M-Base-4K. The chat model was
then post-trained from Base-4K with supervised fine-tuning for chat and tool use, covering the four target jobs below,
general conversation, and examples of declining harmful requests and helping with benign requests that merely sound
alarming. Our held-out evaluation tools, document types and shell tasks were excluded from all training data.

## 4. Capabilities and intended use

Udgam-384M-Chat is built for four jobs inside an app:

1. **App actions:** map a request to one of the app's functions with the right arguments ("wake me at 6:45, label it
   gym" → `set_alarm{time:"06:45", label:"gym"}`), resolving relative dates from a date-time given in the system
   message.
2. **Document → JSON:** fill a supplied JSON schema from receipts, emails, forms or OCR text.
3. **Natural language → shell:** propose one command, either as a `run_shell_command` call with a
   `requires_confirmation` flag or as a plain command block, for a person to review.
4. **Relevance:** decline when no offered tool fits, ask when a required detail is missing, and refuse requests that
   should not be carried out.

It uses the standard Gemma 4 chat format, including native tool declarations, tool calls and an optional thinking
mode (thinking gives no consistent gain for this model). Recommended defaults: Q8_0 (or Q4_K_M on phones), temperature
0.1, top-p 0.9, top-k 40 and a mild repeat penalty (1.1) for tool use; temperature 0.5 for open chat. Temperature is
what costs tool accuracy: BFCL simple drops from 60% (greedy) to 46% at temperature 0.7.

## 5. Evaluation

All results use automatic checkers and greedy decoding with llama.cpp. We report two kinds of tests:

- **Held-out task evaluations** (274 items): our own tests of the four jobs, built on tools, document types and shell
  tasks that were never used in training, plus a held-out date range. They follow this model's prompt conventions
  (date and device headers, normalised argument values), so they favour it.
- **Neutral public benchmarks:** BFCL v3 (simple, multiple, parallel and irrelevance subsets), IFEval and XSTest.

### 5.1 Comparison with other tiny open models

Every model was run on the same harness with its own chat template and native tool-call format, at Q8_0, greedy,
thinking off, and scored by the same code.

| metric | Udgam-384M-Chat | Granite 4.0 350M | Granite 4.0 H 350M | SmolLM2 360M Instruct | Qwen3 0.6B (~2× params) |
|---|---|---|---|---|---|
| Held-out app actions: exact call (unseen tools) | 31.4 | 45.7 | 45.7 | 1.4 | 44.3 |
| Held-out app actions: asks when a detail is missing | 20 | 0 | 5 | 0 | 20 |
| Held-out relevance: right decision | **75** | 18.3 | 20 | 46.7 | 66.7 |
| Held-out document → JSON: schema-valid (unseen schemas) | **56** | 32 | 38 | 0 | 24 |
| Held-out shell: right behaviour (command / ask / refuse) | **85.9** | 59.4 | 67.2 | 39.1 | 67.2 |
| Held-out shell: output correct when executed (n=26) | 7.7 | 3.8 | 7.7 | 0 | 0 |
| BFCL v3 simple | 60 | **90.7** | 88 | 4.7 | 80 |
| BFCL v3 multiple | 67 | 88 | **90** | 5 | 76 |
| BFCL v3 parallel | 46 | **82** | 76 | 0 | 64 |
| BFCL v3 irrelevance | 68 | 55.3 | 66.7 | 75.3 | **88** |
| IFEval prompt-level strict | 26.1 | 47 | 53.4 | 31.4 | **58** |
| XSTest unsafe prompts declined (higher = safer) | 54.0 | 80.0 | **92.5** | 27.5 | 17.0 |
| XSTest safe prompts over-refused (lower is better) | 14.8 | 37.2 | 44.8 | 10.0 | **3.2** |
| Generation speed, Q4_K_M, Apple M6 GPU (tok/s) | **444** | 411 | 262 | 358 | 282 |
| Peak RAM, Q4_K_M CPU run | 323 MB | 349 MB | 352 MB | 238 MB | 579 MB |

**Strengths.** Udgam is strongest at deciding *whether* to act: it declines when no tool fits far more often than
either Granite model (which call a tool almost every time), asks for missing details, produces schema-valid JSON for
unseen schemas more often than any model in the group, and chooses the right shell behaviour most often. It is the
fastest generator in the group on an Apple-silicon GPU.

**Weaknesses.** It is clearly behind both Granite 4.0 350M models on exact tool calls (BFCL simple 60% vs 88–91%,
parallel 46% vs 76–82%) and on instruction following (IFEval 26% vs 47–53%). Qwen3-0.6B, with about twice the
parameters, is also ahead on BFCL and IFEval.

**Fairness notes.** The held-out sets use Udgam's prompt conventions, so other models lose some points to convention
rather than ability. On the neutral BFCL irrelevance test, Qwen3-0.6B (88%) and SmolLM2 (75%) beat Udgam (68%); its
relevance advantage shows mainly on our own held-out set. SmolLM2-360M's template has no tool support, so it was given
the function-calling prompt from its model card; its near-zero tool scores mean "not a tool model". XSTest was scored
with one refusal-phrase classifier for all models, which agreed with an LLM judge on Udgam within 3 points.
FunctionGemma-270M and Gemma 3 270M were not evaluated (gated licence terms).

### 5.2 Quantization

Q8_0 and Q6_K are within noise of f16 across the full suite (perplexity +0.07% and +0.5%). Q4_K_M with an importance
matrix costs about half a point on average (perplexity +4.2%, worst single subset −10 points) and is our recommendation
for phones. Q4_0 loses the most (perplexity +8.3%, worst subset −14 points).

### 5.3 Base models

Udgam-384M-Base reaches 0.668 bits per byte on mixed held-out text at 2,048 tokens. The 4K stage leaves short-context
quality unchanged (+0.15%) while making longer positions usable: retrieval of a code hidden in filler text at 3,500
tokens rises from 1% to 83%, and copying a span at 3,500 tokens from 18% to 86% next-token accuracy.

## 6. Safety

On XSTest, Udgam-384M-Chat declines about **54–57%** of clearly harmful prompts and over-refuses about **15%** of safe
prompts that merely sound alarming. It rarely declines discriminatory-claim, privacy and historical-atrocity prompts,
and over-refuses idioms such as "kill time". Post-training included refusal examples, but the result is below what we would want for an open chatbot, and we release it as **not
safety-hardened**. The base models received no safety training at all and will continue any text.

We recommend:

- app-side allowlists of the tools a user may use, argument validation, and user confirmation for consequential
  actions (send, pay, delete, share, run);
- never auto-executing shell commands; show them and require confirmation;
- a moderation model or policy layer in front of any user-facing open chat.

## 7. Limitations

- **Accuracy on unseen tools:** about one in three calls is exactly right on held-out tools; the model sometimes picks
  the wrong tool or guesses an argument instead of asking.
- **Shell:** syntax is usually valid (~97%), but on unfamiliar tasks only a small fraction of commands produce the
  right output when executed (~8%), and destructive requests are not reliably refused.
- **Knowledge:** at 384M parameters the model is fluent but often factually wrong (capitals, arithmetic, unit
  conversions). It is not suitable for medical, legal or financial advice, or for unsupervised open-ended chat.
- **Conventions:** it works best with the prompt conventions shown in the model card (date-time and device in the
  system message, ISO dates, one schema per extraction).
- **Scope:** English only; 4,096 tokens of context (2,048 for Udgam-384M-Base). Machine-generated training text can
  carry over stylistic habits and errors.

## 8. Licence and citation

The weights and tokenizer of all Udgam-384M models are released under Apache-2.0. The chat template is the official
Gemma 4 template (Apache-2.0).

```bibtex
@misc{udgamlabs2026udgam384m,
  title        = {Udgam-384M: a small on-device tool-calling model trained from scratch},
  author       = {{UdgamLabs}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/UdgamLabs/Udgam-384M-Chat}},
  note         = {Technical report: TECHNICAL_REPORT.md in the Udgam-384M-Chat repository}
}
```

Contact: UdgamLabs, via the Community tab of the Hugging Face model pages.
