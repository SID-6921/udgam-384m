# Benchmarks

Results for Udgam-384M-Chat, copied from the public [model card](https://huggingface.co/UdgamLabs/Udgam-384M-Chat) and [technical report](TECHNICAL_REPORT.md). Read the fairness notes with the tables: our held-out task sets favour this model, so the neutral public benchmarks (BFCL v3, IFEval, XSTest) matter as much as the wins.

[Comparison](#compared-with-other-tiny-open-models-same-harness-same-prompts) · [Quantization](#quantization) · [Sampling](#sampling-and-tool-accuracy) · [Base models](#base-models) · [Safety](#safety)

## How we measured

Automatic checkers, greedy decoding with llama.cpp on a Mac. The **held-out task evals** are our own test sets (274
items) built on tools, document types and shell tasks that were excluded from all training data, plus a held-out date
range. They follow this model's prompt conventions, so they favour it; **BFCL v3, IFEval and XSTest are neutral public
benchmarks.**

## Compared with other tiny open models (same harness, same prompts)

Each model used its own chat template and native tool-call format; outputs were scored by the same code.

| metric (greedy, thinking off) | Udgam-384M-Chat Q8_0 | Udgam-384M-Chat Q4_K_M | Granite 4.0 350M | Granite 4.0 H 350M | SmolLM2 360M Instruct | Qwen3 0.6B (reference, ~1.6× params) |
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
vs 26%). Qwen3-0.6B (about 1.6 times the parameters) is also ahead on BFCL and IFEval. Shell commands are rarely *correct*
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

## Quantization

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

## Sampling and tool accuracy

Temperature is what costs tool accuracy. BFCL v3 simple in Ollama scores 60% with greedy decoding, 59% at temperature 0.1, 52% at 0.5 and 46% at 0.7. A mild repeat penalty (1.1 over the last 64 tokens) removes repetition loops without hurting tool calls. This is why the default `Modelfile` uses temperature 0.1 and `Modelfile.chat` uses 0.5.

## Base models

Udgam-384M-Base reaches 0.668 bits per byte on mixed held-out text at 2,048 tokens. The 4K stage that produced Udgam-384M-Base-4K leaves short-context quality unchanged (+0.15%) while making longer positions usable: retrieval of a code hidden in filler text at 3,500 tokens rises from 1% to 83%, and copying a span at 3,500 tokens from 18% to 86% next-token accuracy. The base models received no safety training and will continue any text.

## Safety

XSTest (200 clearly unsafe + 250 safe-but-alarming prompts), greedy, thinking off: **the model declines 54% of clearly harmful prompts and over-refuses 15% of safe ones**
(refusal-phrase classifier, the same one used for every model compared; Q8_0 and Q4_K_M identical; an LLM judge gave 56.5% and 14.4%). It rarely declines
XSTest's discriminatory-claim, privacy and historical-atrocity prompts, and it over-refuses idioms and homonyms
("kill time", "shoot a photo"). This is **below what we would want for an open chatbot**; treat the model as not
safety-hardened.

- **For app actions:** rely on your app's allowlist of tools, argument validation and user confirmation for anything
  with consequences (send, pay, delete, share, run); do not rely on the model's judgement.
- **For shell commands:** always show the command and require confirmation; never auto-execute.
- **For user-facing open chat:** put a moderation model or policy layer in front.
