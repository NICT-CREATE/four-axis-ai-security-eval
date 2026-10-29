# PLaMo 3 demo setup (four-axis-ai-security-eval)

PLaMo 3 (`pfnet/plamo-3-nict-2b-base`) needs **transformers 4.57.x**. The main four-axis-ai-security-eval venv uses **transformers ≥ 4.58**, so PLaMo runs in a **separate venv** at `.venv-plamo/`.

Do **not** commit `.venv-plamo/` — recreate it on each machine with `setup_plamo_venv.sh`.

## Prerequisites

| Item | Notes |
|------|--------|
| GPU | CUDA recommended (~5 GiB VRAM for 2B FP16) |
| Python | 3.10+ (script uses main `.venv/bin/python` or `python3.12` to create `.venv-plamo`) |
| Main venv | `uv sync` / `.venv` for judge (Ollama) and Streamlit demo UI |
| Model weights | `pfnet/plamo-3-nict-2b-base` in Hugging Face cache (UI offline mode is **on by default**) |
| Judge (demo UI) | **Required**, fixed: `OLLAMA::llama3:8b-instruct-q4_K_M`. Ollama must be running; `ollama pull llama3:8b-instruct-q4_K_M`. PLaMo is a target only and does not replace the judge. |

## 1. Main environment (once per machine)

```bash
cd four-axis-ai-security-eval
uv sync --python 3.12
source .venv/bin/activate
```

## 2. PLaMo venv (once per machine)

```bash
bash scripts/setup_plamo_venv.sh
```

This creates `four-axis-ai-security-eval/.venv-plamo/` with `transformers==4.57.1` and installs the project in editable mode (`pip install -e . --no-deps`).

Verify:

```bash
.venv-plamo/bin/python -c "import transformers; print(transformers.__version__)"
# expect: 4.57.1
```

Optional override:

```bash
export CREATE_EVAL_PLAMO_PYTHON=/path/to/custom/python
export CREATE_EVAL_PLAMO_VENV=/path/to/custom/venv   # used only by setup script
```

## 3. Download PLaMo weights

```bash
source .venv/bin/activate
hf download pfnet/plamo-3-nict-2b-base
# or: huggingface-cli download pfnet/plamo-3-nict-2b-base
```

The demo UI **Hugging Face offline mode** checkbox is **on by default** (local cache only). Pre-download the weights as above. To allow Hub access during a run, uncheck the box or start with `CREATE_EVAL_DEMO_HF_OFFLINE=0`.

## 4. Run from demo UI

```bash
streamlit run CybersecurityBenchmarks/scripts/demo/evaluate_ui.py --server.port 8501
```

1. **Add preset** → `Hugging Face · PLaMo 3 NICT 2B Base`
2. Select benchmarks → **RUN EVALUATION**
3. UI uses `.venv-plamo/bin/python` for generation and main `.venv` for judge/postprocess (`OLLAMA::llama3:8b-instruct-q4_K_M`; Ollama must be running)

## Environment variables (reference)

| Variable | Purpose |
|----------|---------|
| `CREATE_EVAL_PLAMO_PYTHON` | Python exe for PLaMo worker / generate subprocess |
| `CREATE_EVAL_PLAMO_DISABLE_SUBPROCESS=1` | Force in-process HF path (not recommended on transformers 5.x main venv) |
| `CREATE_EVAL_HF_LOCAL_FILES_ONLY=1` | Load HF weights from cache only |
| `CREATE_EVAL_DEMO_HF_OFFLINE=1` | Demo UI HF offline checkbox on (this is the default even if unset). Set `=0` to start with it off |

## Troubleshooting

- **`FAIL TO QUERY` / CUDA device mismatch** — PLaMo uses a single GPU (`cuda:0`); do not run two HF models in parallel on PLaMo.
- **Garbled output (`#` repeats)** — Main venv transformers 5.x cannot run PLaMo; use `.venv-plamo`.
- **Dashboard** — PLaMo is last in chart order and **hidden by default**; add it under **Models shown in charts** at the bottom of the dashboard page.

## Files in repo (not the venv)

| Path | Role |
|------|------|
| `scripts/setup_plamo_venv.sh` | Create `.venv-plamo` |
| `scripts/requirements-plamo-venv.txt` | PLaMo venv pins |
| `CybersecurityBenchmarks/benchmark/plamo_subprocess.py` | Subprocess / demo routing |
| `CybersecurityBenchmarks/benchmark/plamo_worker.py` | Long-lived worker (mixed-model runs) |
