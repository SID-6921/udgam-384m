# Quickstart

Udgam-384M-Chat is a 384M-parameter English model for tool calling inside apps. It uses the standard Gemma 4 chat format, so it runs in Ollama, LM Studio, llama.cpp, MLX and transformers without custom code. Context: 4,096 tokens.

## 1. Pick a file

| You are using | Download | Notes |
|---|---|---|
| Ollama, llama.cpp, LM Studio | [`Udgam-384M-Chat-Q8_0.gguf`](https://huggingface.co/udgamlabs/Udgam-384M-Chat-GGUF) (410 MB) | **Default.** Lossless within noise. |
| The same, on a phone or low-memory device | `Udgam-384M-Chat-Q4_K_M.gguf` (238 MB) | Small quality loss. |
| A runtime without K-quants | `Udgam-384M-Chat-Q4_0.gguf` (227 MB) | Loses the most. Avoid otherwise. |
| MLX on Apple silicon | [`Udgam-384M-Chat-MLX-4bit`](https://huggingface.co/udgamlabs/Udgam-384M-Chat-MLX-4bit) (216 MB) or [`-MLX`](https://huggingface.co/udgamlabs/Udgam-384M-Chat-MLX) (fp16) | |
| transformers / PyTorch | [`Udgam-384M-Chat`](https://huggingface.co/udgamlabs/Udgam-384M-Chat) | safetensors |
| Text continuation or your own fine-tune | [`Udgam-384M-Base-4K`](https://huggingface.co/udgamlabs/Udgam-384M-Base-4K) or [`Udgam-384M-Base`](https://huggingface.co/udgamlabs/Udgam-384M-Base) | No chat or safety training |

Check a download against [`SHA256SUMS`](../SHA256SUMS) with `shasum -a 256 <file>` (Linux: `sha256sum`).

## 2. Run it

### Ollama

Put the GGUF and [`ollama/Modelfile`](../ollama/Modelfile) in one folder. The Modelfile uses Ollama's native Gemma 4 renderer and parser, so tool calls come back parsed:

```bash
ollama create udgam -f Modelfile           # tool-use defaults (temperature 0.1)
ollama create udgam-chat -f Modelfile.chat # open-chat preset (temperature 0.5)
ollama run udgam "Explain what a hash map is in two sentences."
```

For a phone-sized model, change the `FROM` line to `./Udgam-384M-Chat-Q4_K_M.gguf`. Thinking is **on** by default in Ollama for Gemma 4 models. Send `"think": false` in API calls, or type `/set nothink` in `ollama run`.

### llama.cpp

```bash
llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja                  # OpenAI-compatible API on :8080
llama-server -m Udgam-384M-Chat-Q8_0.gguf -c 4096 --jinja --reasoning off  # the same, thinking off
llama-cli    -m Udgam-384M-Chat-Q8_0.gguf -c 4096 -cnv                     # interactive chat
```

`--jinja` makes the server use the chat template embedded in the GGUF and return parsed `tool_calls`. llama-server also serves a chat web UI at `http://localhost:8080`.

### LM Studio

1. Search for `udgamlabs/Udgam-384M-Chat-GGUF` (or drop the `.gguf` into LM Studio's models folder) and pick `Q8_0`, or `Q4_K_M` for low memory.
2. Set **Context Length 4096**, temperature 0.1, top-p 0.9, top-k 40, repeat penalty 1.1.
3. For the **MLX** versions, set **Max Concurrent Predictions to 1** before loading (see [LM Studio notes](INTEGRATION.md#lm-studio)).

### MLX

```bash
pip install mlx-lm
mlx_lm.generate --model udgamlabs/Udgam-384M-Chat-MLX-4bit --prompt "Give me three tips for sleeping better." --max-tokens 200
mlx_lm.server   --model udgamlabs/Udgam-384M-Chat-MLX-4bit --port 8091   # OpenAI-compatible API with parsed tool_calls
```

When you call `mlx_lm.server`, send the same model id in the request's `model` field.

### transformers

```python
# tested with transformers 5.18
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("udgamlabs/Udgam-384M-Chat")
model = AutoModelForCausalLM.from_pretrained("udgamlabs/Udgam-384M-Chat")
msgs = [{"role": "user", "content": "Give me three tips for sleeping better."}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_dict=True, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use fp32 (or llama.cpp) if you need bit-exact greedy results. bf16 generation in transformers depends on batch shape.

## 3. Recommended settings

| setting | tool / app use (`Modelfile`) | open chat (`Modelfile.chat`) |
|---|---|---|
| temperature | 0.1 | 0.5 |
| top_p / top_k | 0.9 / 40 | 0.9 / 40 |
| repeat_penalty (last 64 tokens) | 1.1 | 1.1 |
| max new tokens / context | 1024 / 4096 | 1024 / 4096 |

Temperature is what costs tool accuracy: BFCL simple in Ollama scores 60% greedy, 59% at 0.1, 52% at 0.5 and 46% at 0.7. A mild repeat penalty stops repetition loops without hurting tool calls. In transformers, `generation_config.json` sets temperature 0.1, top-p 0.9 and top-k 40 but **no** repetition penalty, because transformers applies it to the prompt too and would penalise tool names. Use greedy decoding when you need reproducible tool calls.

## 4. Next steps

- Wire it into an app: [integration guide](INTEGRATION.md).
- Run the [examples](../examples/): tool calls, document → JSON, shell suggestions with confirmation.
- Know what to expect: [benchmarks](BENCHMARKS.md) and [model card](MODEL_CARD.md).
