# four-axis-ai-security-eval

Four-axis LLM safety evaluation demo (15 benchmarks × 5 prompts).

**Axes:** Safety · Cyber misuse · Excessive refusal · General

## Requirements

### Required (UI + Full / Judge-only evaluation)

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) **installed and running** (`ollama serve` or the Ollama app)
- **Fixed judge LLM:** `OLLAMA::llama3:8b-instruct-q4_K_M` (~4.9 GB)

The demo UI does **not** let you change the judge. `Full (generate + judge)` and `Judge only` always call this Ollama model. Download it before those modes:

```bash
ollama pull llama3:8b-instruct-q4_K_M
```

`Generate only` and ad-hoc user prompts do not use the judge.

### Optional (models under test)

Add at least one **target** model in the UI before **RUN EVALUATION**. Targets are not the judge.

| Target | Download | Notes |
|--------|----------|--------|
| `OLLAMA::llama3.2:1b` | `ollama pull llama3.2:1b` | Lightest preset; CPU is fine |
| Hugging Face presets | Hub download into the local cache (see [Hugging Face targets](#hugging-face-targets-optional)) | CUDA GPU recommended (8B-class). Ollama-only runs work on CPU |

## Quick start

```bash
cd four-axis-ai-security-eval
uv sync --python 3.12
source .venv/bin/activate

# Ollama must already be running
ollama pull llama3:8b-instruct-q4_K_M   # required judge (OLLAMA::llama3:8b-instruct-q4_K_M)
ollama pull llama3.2:1b                 # optional small target

streamlit run CybersecurityBenchmarks/scripts/demo/evaluate_ui.py --server.port 8501
```

Open `http://localhost:8501`.

In the UI, add a target (preset `Ollama · llama3.2:1b` is enough for a smoke test), select benchmarks, then **RUN EVALUATION**.

`.streamlit/config.toml` disables Streamlit's file watcher (avoids noisy `transformers` import traces) and usage stats.

If `uv sync --python 3.12` fails on Apple Silicon with `Bad CPU type in executable`, uv is likely using a managed **x86_64** CPython. Install or select an **arm64** 3.12:

```bash
uv python list
uv python install 3.12    # if no macos-aarch64 3.12 is listed
uv sync --python 3.12
```

If it still picks x86_64, pass the arm64 interpreter explicitly (path from `uv python list`):

```bash
uv sync --python /path/to/cpython-3.12-macos-aarch64-none/bin/python3.12
```

## Hugging Face targets (optional)

HF models are **evaluation targets only**. They do not replace the Ollama judge. Skip this section for an Ollama-only demo.

The UI **Hugging Face offline mode** checkbox is **on by default** (local cache only; no Hub access during a run). Pre-download weights, or uncheck the box / start with `CREATE_EVAL_DEMO_HF_OFFLINE=0`.

Public presets and matching Hub ids:

```bash
# gated: accept the license on Hugging Face and run `hf auth login`
hf download meta-llama/Llama-3.2-1B-Instruct

hf download fdtn-ai/Foundation-Sec-8B-Instruct
hf download Qwen/Qwen3-8B
```

| UI preset | Spec |
|-----------|------|
| Hugging Face · Llama-3.2-1B-Instruct | `HF16::meta-llama/Llama-3.2-1B-Instruct` |
| Hugging Face · Foundation-Sec-8B-Instruct | `HF16::fdtn-ai/Foundation-Sec-8B-Instruct` |
| Hugging Face · Qwen3-8B | `HF16::Qwen/Qwen3-8B` |
| Hugging Face · PLaMo 3 NICT 2B Base | `HF16::pfnet/plamo-3-nict-2b-base` (separate venv; see below) |

`huggingface-cli download …` works as an alias of `hf download`. You can also add a local folder as `HF16::/absolute/path`.

## What the UI does

1. **Models and test set** — add Ollama or Hugging Face **targets** (manual or preset); select any of the 15 tasks; optional ad-hoc prompts (generate only).
2. **Run evaluation** — generate responses, judge with the fixed `OLLAMA::llama3:8b-instruct-q4_K_M`, then verify/fix labels.
3. **Dashboard** — standardized scores vs bundled `base_v2` baselines, plus prompt/response/judge details.

Each benchmark runs a **fixed 5-prompt subset** under `CybersecurityBenchmarks/datasets/demo/`.

## PLaMo 3 (optional)

`pfnet/plamo-3-nict-2b-base` needs transformers 4.57 in a separate venv, plus a Hub download of the weights:

```bash
bash scripts/setup_plamo_venv.sh
hf download pfnet/plamo-3-nict-2b-base
```

See [scripts/PLAMO_SETUP.md](scripts/PLAMO_SETUP.md). The Ollama judge above is still required for Full / Judge-only runs.

## Layout

```text
four-axis-ai-security-eval/
  CybersecurityBenchmarks/
    scripts/demo/          # Streamlit UI
    benchmark/             # generate + judge engine
    datasets/demo/         # *_demo5.json
    outputs/demo/base_v2/  # comparison baselines
    outputs/demo/test_v2/  # live run output (not committed)
```

## Intended use

Research and defensive evaluation only. Demo prompts include jailbreak, HarmBench, and cyber-misuse items. See [NOTICE](NOTICE) for dataset attribution and license caveats.
