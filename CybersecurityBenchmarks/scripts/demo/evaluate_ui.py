import importlib.util
import inspect
import json
import os
import shlex
import subprocess
import threading
import time
import sys
from typing import List, Optional

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import runpy

import shutil

# Demo UI policy:
# - Prefer demo-only behavior in this folder; shared benchmark code may change when needed for correctness.
# - All user-visible strings in this app must be English.
# - Multipage hub: streamlit run demo_app.py (sidebar: Current vs Legacy).
#
# Load demo_benchmark_config from the same directory (not installed as a package).
_SCRIPT_DIR_FOR_CFG = os.path.dirname(os.path.abspath(__file__))
_cfg_spec = importlib.util.spec_from_file_location(
    "demo_benchmark_config",
    os.path.join(_SCRIPT_DIR_FOR_CFG, "demo_benchmark_config.py"),
)
_demo_cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec.loader is not None
_cfg_spec.loader.exec_module(_demo_cfg)
_batch_spec = importlib.util.spec_from_file_location(
    "demo_batch_pipeline",
    os.path.join(_SCRIPT_DIR_FOR_CFG, "demo_batch_pipeline.py"),
)
_demo_batch = importlib.util.module_from_spec(_batch_spec)
assert _batch_spec.loader is not None
_batch_spec.loader.exec_module(_demo_batch)
_up_spec = importlib.util.spec_from_file_location(
    "demo_user_prompts",
    os.path.join(_SCRIPT_DIR_FOR_CFG, "demo_user_prompts.py"),
)
_demo_user = importlib.util.module_from_spec(_up_spec)
assert _up_spec.loader is not None
_up_spec.loader.exec_module(_demo_user)
DEMO_BENCHMARKS = _demo_cfg.DEMO_BENCHMARKS
BENCH_BY_ID = _demo_cfg.BENCH_BY_ID
bench_output_paths = _demo_cfg.bench_output_paths
DEMO_MODEL_PRESETS = _demo_cfg.DEMO_MODEL_PRESETS
DEMO_AXES = _demo_cfg.DEMO_AXES
AXIS_BENCHMARK_IDS = _demo_cfg.AXIS_BENCHMARK_IDS
DEMO_AXIS_UI_DESCRIPTIONS = _demo_cfg.DEMO_AXIS_UI_DESCRIPTIONS
ordered_demo_benchmarks = _demo_cfg.ordered_demo_benchmarks
benchmark_source_label = _demo_cfg.benchmark_source_label

# st.container(height=...) enables scrolling (Streamlit ≥ 1.30)
_CONTAINER_SUPPORTS_HEIGHT = "height" in inspect.signature(st.container).parameters

# safe rerun helper: streamlit のバージョン差に対応した安全な rerun ヘルパー
def _safe_rerun():
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.query_params = {"_rerun": int(time.time())}
        except Exception:
            st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)

# スクリプトの場所からCybersecurityBenchmarksディレクトリのパスを動的に取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # CybersecurityBenchmarksディレクトリ（scripts/demo/から2階層上）
DEMO_DATASETS_DIR = os.path.join(BASE, "datasets", "demo")
OUTPUT_BASE = os.path.join(BASE, "outputs/demo/test_v2/with_syspro")
JUDGE_SPEC = "OLLAMA::llama3:8b-instruct-q4_K_M"
LOG_DIR = os.path.join(BASE, "log/demo")
os.makedirs(LOG_DIR, exist_ok=True)

_CREATE_EVAL_ROOT = os.path.dirname(BASE)
if _CREATE_EVAL_ROOT not in sys.path:
    sys.path.insert(0, _CREATE_EVAL_ROOT)

st.set_page_config(page_title="four-axis-ai-security-eval", layout="wide")

# UI button style: make buttons larger and clearer
st.markdown(
    """
    <style>
    /* General button styling (larger, rounded) */
    div.stButton>button {
        font-size: 16px;
        padding: 12px 20px;
        border-radius: 10px;
        border: none;
        box-shadow: none;
        color: #fff;
        background-color: #2b3036; /* fallback neutral */
    }
    /* Make action buttons more visible (RUN EVALUATION will use this style) */
    /* Use a stronger blue so it stands out on dark theme */
    div.stButton>button[title="run_eval_primary"], div.stButton>button.primary {
        background-color: #0b84ff !important;
        color: white !important;
        font-weight: 700;
        font-size: 18px;
        padding: 14px 26px;
    }
    /* Hover */
    div.stButton>button:hover {
        filter: brightness(1.05);
    }
    .bench-table-header {
        font-weight: 600;
        font-size: 0.8rem;
        opacity: 0.92;
        padding-bottom: 0.12rem;
        margin-bottom: 0;
        border-bottom: 1px solid rgba(250,250,250,0.15);
    }
    /* Dense rows: scrollable benchmark table only (:has .bench-table-header) */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.bench-table-header) div[data-testid="stVerticalBlock"] label[data-testid="stWidgetLabel"] p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.12 !important;
        font-size: 0.86rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.bench-table-header) div[data-testid="stVerticalBlock"] .stCheckbox {
        gap: 0.2rem !important;
        margin-top: -0.15rem !important;
        margin-bottom: -0.2rem !important;
        min-height: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.bench-table-header) div[data-testid="stVerticalBlock"] div[data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.12 !important;
        font-size: 0.86rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.bench-table-header) div[data-testid="stHorizontalBlock"] {
        gap: 0.4rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.bench-table-header) div[data-testid="column"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("four-axis-ai-security-eval")
st.caption("AI Security Evaluation Platform — four-axis LLM safety demo")

st.subheader("Step 1. Input the model and test set")
st.caption(
    "Add models to evaluate and select test sets. Multiple models and test sets can be chosen. "
    "Pick **Ollama** or **Hugging Face (HF)** and enter a model id (**Add manually**), or use **Add preset** below."
)
st.caption(
    "On the right: **15 benchmarks** in four axes—use **Axis** to toggle a whole group or each **Benchmark** row. "
    "**# Prompts** is the full benchmark size (this demo still runs a fixed small subset of 5 prompts per task)."
)
# session state for models list: list[{"id": int, "spec": str}] (stable ids for Remove buttons)
if "models" not in st.session_state:
    st.session_state["models"] = []
if "model_uid_next" not in st.session_state:
    st.session_state["model_uid_next"] = 0
if st.session_state["models"] and isinstance(st.session_state["models"][0], str):
    st.session_state["models"] = [
        {"id": i, "spec": s} for i, s in enumerate(st.session_state["models"])
    ]
    st.session_state["model_uid_next"] = len(st.session_state["models"])


def _model_specs() -> List[str]:
    return [e["spec"] for e in st.session_state["models"]]


def _append_model_entry(spec: str) -> None:
    uid = st.session_state["model_uid_next"]
    st.session_state["model_uid_next"] = uid + 1
    st.session_state["models"].append({"id": uid, "spec": spec})


def _normalize_llm_spec(spec: str) -> str:
    """Demo-only: UI shows Hugging Face (HF) but subprocess uses HF16:: (HF:: in llm.create defaults to 4-bit)."""
    s = spec.strip()
    if len(s) < 6 or "::" not in s:
        return s
    prov, rest = s.split("::", 1)
    u = prov.upper()
    if u in ("HF", "HF_FP32", "HF_NO4BIT"):
        return f"HF16::{rest}"
    return s


def _spec_for_runner(spec: str) -> str:
    """Normalize HF→HF16 and apply env remaps (``demo_benchmark_config``)."""
    s = _normalize_llm_spec(spec)
    return _demo_cfg.demo_remap_hf_local_path_in_spec(s)


# Same offline flags as CREATE-eval RQ2 transfer / Phase 3 (llm.py reads CREATE_EVAL_*).
_DEMO_HF_OFFLINE_ENV_FLAGS = {
    "CREATE_EVAL_HF_LOCAL_FILES_ONLY": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CREATE_EVAL_HF_STABLE_ATTENTION": "1",
}


def _demo_hf_offline_default_enabled() -> bool:
    raw = os.environ.get("CREATE_EVAL_DEMO_HF_OFFLINE", "").strip().lower()
    if not raw:
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return raw in ("1", "true", "yes", "on")


def _model_specs_use_hf(models: List[str]) -> bool:
    hf_providers = frozenset({"HF", "HF16", "HF_FP32", "HF_NO4BIT"})
    for spec in models:
        normalized = _spec_for_runner(spec).strip()
        if "::" not in normalized:
            continue
        if normalized.split("::", 1)[0].upper() in hf_providers:
            return True
    return False


def _validate_local_hf_model_paths(models: List[str]) -> Optional[str]:
    """Return an English error message if any HF spec uses a missing absolute local path."""
    hf_providers = frozenset({"HF", "HF16", "HF_FP32", "HF_NO4BIT"})
    for spec in models:
        normalized = _spec_for_runner(spec).strip()
        if "::" not in normalized:
            continue
        prov, rest = normalized.split("::", 1)
        if prov.upper() not in hf_providers:
            continue
        path = os.path.expanduser(rest.strip())
        if os.path.isabs(path) and not os.path.isdir(path):
            return (
                "A listed Hugging Face model points to a local folder that was not found on this machine. "
                "Update the model list and try again."
            )
    return None


def _demo_generation_python(models: List[str]) -> str:
    from CybersecurityBenchmarks.benchmark.plamo_subprocess import generation_python_exec

    return generation_python_exec(models, sys.executable, spec_for_runner=_spec_for_runner)


def _render_plamo_venv_notice(models: List[str]) -> None:
    from CybersecurityBenchmarks.benchmark.plamo_subprocess import (
        any_model_is_plamo,
        resolve_plamo_python,
    )

    if not any_model_is_plamo(models, _spec_for_runner):
        return
    if st.session_state.get("demo_run_in_progress"):
        return
    py = resolve_plamo_python()
    if py is not None:
        st.info(
            f"PLaMo uses a separate Python environment (transformers 4.57): `{py}`"
        )
    else:
        st.warning(
            "PLaMo requires `.venv-plamo` (transformers 4.57). "
            "Run `bash scripts/setup_plamo_venv.sh` from the four-axis-ai-security-eval root."
        )


# dashboard mode flag: when True, auto-refresh should be disabled so dashboard view is stable
if "show_dashboard" not in st.session_state:
    st.session_state["show_dashboard"] = False
# whether to show the live log area in this session (default: hidden)
if "show_log" not in st.session_state:
    st.session_state["show_log"] = False
# flags to request changing auto-refresh on the next rerun (avoid modifying widget-backed keys
# after the widget has been instantiated in the same run)
if "disable_auto_refresh_next_run" not in st.session_state:
    st.session_state["disable_auto_refresh_next_run"] = False
if "enable_auto_refresh_next_run" not in st.session_state:
    st.session_state["enable_auto_refresh_next_run"] = False

# If a previous action requested auto_refresh to be disabled/enabled on next run,
# apply it now BEFORE any checkbox with key="auto_refresh" is created.
if st.session_state.get("disable_auto_refresh_next_run", False):
    st.session_state["auto_refresh"] = False
    st.session_state["disable_auto_refresh_next_run"] = False
if st.session_state.get("enable_auto_refresh_next_run", False):
    st.session_state["auto_refresh"] = True
    st.session_state["enable_auto_refresh_next_run"] = False


def _init_bench_axis_session_state() -> None:
    for b in DEMO_BENCHMARKS:
        if f"bench_run_{b['id']}" not in st.session_state:
            st.session_state[f"bench_run_{b['id']}"] = False
    for ax in DEMO_AXES:
        aid = ax["id"]
        bids = AXIS_BENCHMARK_IDS[aid]
        if f"axis_run_{aid}" not in st.session_state:
            st.session_state[f"axis_run_{aid}"] = all(
                st.session_state.get(f"bench_run_{bid}", False) for bid in bids
            )


_init_bench_axis_session_state()


def _axis_toggle_all(axis_id: str, bench_ids: List[str]) -> None:
    v = st.session_state[f"axis_run_{axis_id}"]
    for bid in bench_ids:
        st.session_state[f"bench_run_{bid}"] = v


def _bench_sync_axis(axis_id: str, bench_ids: List[str]) -> None:
    st.session_state[f"axis_run_{axis_id}"] = all(
        st.session_state.get(f"bench_run_{bid}", True) for bid in bench_ids
    )


col_left, col_right = st.columns([1.2, 2])
_PROVIDER_LABEL_TO_CODE = {
    "Ollama": "OLLAMA",
    # Label is HF; subprocess gets HF16:: (unchanged llm.create in benchmark core).
    "Hugging Face (HF)": "HF16",
}

with col_left:
    st.subheader("Models")
    _prov_label = st.selectbox(
        "Provider (for manual add)",
        list(_PROVIDER_LABEL_TO_CODE.keys()),
        help="Prefix for manual add. If you type OLLAMA:: / HF:: / HF16:: etc., it is normalized before passing to the runner.",
    )
    provider = _PROVIDER_LABEL_TO_CODE[_prov_label]
    with st.form("add_model_form", clear_on_submit=True):
        model_input = st.text_input("Model ID (with or without prefix)", "")
        submitted = st.form_submit_button("Add manually")
        if submitted:
            v = model_input.strip()
            if v:
                uv = v.upper()
                if (
                    uv.startswith("HF::")
                    or uv.startswith("OLLAMA::")
                    or uv.startswith("HF16::")
                    or uv.startswith("HF_FP32::")
                    or uv.startswith("HF_NO4BIT::")
                ):
                    m = _spec_for_runner(v)
                else:
                    m = _spec_for_runner(f"{provider}::{v}")
                _append_model_entry(m)
                _safe_rerun()
    preset_labels = ["(Select a preset)"]
    preset_map: dict[str, str | None] = {"(Select a preset)": None}
    for title, spec in DEMO_MODEL_PRESETS:
        preset_labels.append(title)
        preset_map[title] = spec
    _preset_help = (
        "Examples: Ollama — llama3.2:1b, llama3-instruct-backdoor; "
        "Hugging Face — meta-llama/Llama-3.2-1B-Instruct, fdtn-ai/Foundation-Sec-8B-Instruct. "
        "This demo passes HF-like specs as HF16:: (no 4-bit quantization)."
    )
    chosen_preset = st.selectbox(
        "Add from preset",
        preset_labels,
        help=_preset_help,
    )
    if st.button("Add preset to model list", key="add_preset_btn"):
        spec = preset_map.get(chosen_preset)
        if spec:
            spec = _spec_for_runner(spec)
        if spec and spec not in _model_specs():
            _append_model_entry(spec)
            _safe_rerun()
    for row_idx, entry in enumerate(list(st.session_state["models"])):
        c1, c2 = st.columns([6, 3])
        c1.write(f"{row_idx + 1}. {_demo_cfg.demo_short_label_llm_spec(entry['spec'])}")
        if c2.button("Remove", key=f"rm_{entry['id']}"):
            st.session_state["models"] = [
                e for e in st.session_state["models"] if e["id"] != entry["id"]
            ]
            _safe_rerun()

with col_right:
    st.subheader("Benchmarks (15 tasks)")
    st.caption(
        "Standardized metrics are oriented so **higher is better** within each axis. "
        "Hover the **?** next to each axis name for a short definition."
    )
    if _CONTAINER_SUPPORTS_HEIGHT:
        _bench_scroll = st.container(height=550, border=True)
    else:
        _bench_scroll = st.container()
    with _bench_scroll:
        _w_axis, _w_bench, _w_n, _w_met = 1.05, 3.15, 0.72, 0.88
        h = st.columns([_w_axis, _w_bench, _w_n, _w_met])
        h[0].markdown('<div class="bench-table-header">Axis</div>', unsafe_allow_html=True)
        h[1].markdown('<div class="bench-table-header">Benchmark (source)</div>', unsafe_allow_html=True)
        h[2].markdown(
            '<div class="bench-table-header"># Prompts<br/><span style="font-weight:400;opacity:0.85">(full set)</span></div>',
            unsafe_allow_html=True,
        )
        h[3].markdown('<div class="bench-table-header">Metric (↑)</div>', unsafe_allow_html=True)

        _axis_sep = '<hr style="margin:0.12rem 0;border:none;border-top:1px solid rgba(255,255,255,0.12)"/>'
        for ax_idx, axis in enumerate(DEMO_AXES):
            aid = axis["id"]
            bids = AXIS_BENCHMARK_IDS[aid]
            if ax_idx > 0:
                st.markdown(_axis_sep, unsafe_allow_html=True)
            axis_desc = DEMO_AXIS_UI_DESCRIPTIONS.get(aid)
            for j, bid in enumerate(bids):
                b = BENCH_BY_ID[bid]
                c0, c1, c2, c3 = st.columns([_w_axis, _w_bench, _w_n, _w_met])
                with c0:
                    if j == 0:
                        st.checkbox(
                            axis["label"],
                            key=f"axis_run_{aid}",
                            help=axis_desc,
                            on_change=_axis_toggle_all,
                            args=(aid, bids),
                        )
                    else:
                        st.write("")
                with c1:
                    st.checkbox(
                        benchmark_source_label(b),
                        key=f"bench_run_{bid}",
                        on_change=_bench_sync_axis,
                        args=(aid, bids),
                    )
                with c2:
                    st.markdown(f"{b['full_n']:,}")
                with c3:
                    if j == 0:
                        st.markdown(axis["standardized_metric"])
                    else:
                        st.write("")
    if not _CONTAINER_SUPPORTS_HEIGHT:
        st.caption("Upgrade Streamlit (≥1.30) for a scrollable benchmark list.")


def _init_user_prompt_slots() -> None:
    if "user_prompt_slot_ids" not in st.session_state:
        st.session_state["user_prompt_slot_ids"] = [0]
        st.session_state["user_prompt_uid_next"] = 1
    for uid in st.session_state["user_prompt_slot_ids"]:
        key = f"user_prompt_text_{uid}"
        if key not in st.session_state:
            st.session_state[key] = ""


def _user_prompt_texts_from_session() -> List[str]:
    return [
        st.session_state.get(f"user_prompt_text_{uid}", "")
        for uid in st.session_state.get("user_prompt_slot_ids", [])
    ]


def _user_prompts_active() -> bool:
    return _demo_user.prompts_configured(_user_prompt_texts_from_session())


_init_user_prompt_slots()

st.markdown("---")
st.subheader("User-input prompts")
st.caption(
    "Ad-hoc prompts for quick checks (generate only, no judge). "
    f"Up to {_demo_user.USER_PROMPT_MAX_SLOTS} prompts. "
    "Do not use together with benchmarks in one run. "
    "Results appear under Data details → User-input prompts (not on the Dashboard)."
)
_slot_ids = list(st.session_state["user_prompt_slot_ids"])
for row_i, uid in enumerate(_slot_ids):
    c1, c2 = st.columns([10, 1])
    with c1:
        st.text_area(
            f"Prompt {row_i + 1}",
            key=f"user_prompt_text_{uid}",
            height=68,
            label_visibility="collapsed",
            placeholder=f"Enter prompt {row_i + 1}…",
        )
    with c2:
        if len(_slot_ids) > 1 and st.button("Remove", key=f"user_prompt_rm_{uid}"):
            st.session_state["user_prompt_slot_ids"] = [
                x for x in _slot_ids if x != uid
            ]
            _safe_rerun()
add_col, _ = st.columns([2, 8])
with add_col:
    if len(_slot_ids) < _demo_user.USER_PROMPT_MAX_SLOTS and st.button(
        "Add prompt", key="user_prompt_add"
    ):
        nuid = st.session_state["user_prompt_uid_next"]
        st.session_state["user_prompt_uid_next"] = nuid + 1
        st.session_state["user_prompt_slot_ids"] = _slot_ids + [nuid]
        _safe_rerun()

st.markdown("---")
st.subheader("Step 2. Run evaluation")
st.caption(
    f"Judge LLM (fixed): `{JUDGE_SPEC}` "
)
if "demo_hf_offline" not in st.session_state:
    st.session_state["demo_hf_offline"] = _demo_hf_offline_default_enabled()
st.checkbox(
    "Hugging Face offline mode (local cache only)",
    key="demo_hf_offline",
    help=(
        "For Hugging Face models under test: load weights from the local Hugging Face cache "
        "or an absolute folder path only (no Hub network access). Sets "
        "CREATE_EVAL_HF_LOCAL_FILES_ONLY, HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, and "
        "CREATE_EVAL_HF_STABLE_ATTENTION on evaluation subprocesses. Ollama models are unchanged. "
        "On by default; set CREATE_EVAL_DEMO_HF_OFFLINE=0 in the shell to start with it off."
    ),
)
if st.session_state.get("demo_hf_offline") and not _model_specs_use_hf(_model_specs()):
    st.caption(
        "Offline mode is enabled; it applies when you add Hugging Face models. Ollama-only runs are unchanged."
    )

_render_plamo_venv_notice(_model_specs())

if "demo_batch_benchmarks" not in st.session_state:
    st.session_state["demo_batch_benchmarks"] = True
st.checkbox(
    "Batch benchmarks (merge prompts per run type)",
    key="demo_batch_benchmarks",
    help=(
        "When on and two or more benchmarks are selected: merge prompts so each under-test model "
        "is loaded once per generate group (LLM_generate_only vs prompt-injection), and run one "
        "batched judge pass. When off: one subprocess per benchmark (legacy behavior). "
        "Shell override: CREATE_EVAL_DEMO_BATCH=0 forces batch off."
    ),
)
if not st.session_state.get("demo_batch_benchmarks", True):
    st.caption(
        "Batching is off: each selected benchmark runs in its own generate/judge subprocess."
    )

_DEMO_RUN_MODE_LABELS = {
    "full": "Full (generate + judge)",
    "generate": "Generate only",
    "judge": "Judge only",
}
if "demo_run_mode" not in st.session_state:
    st.session_state["demo_run_mode"] = "full"
_user_prompt_mode_ui = _user_prompts_active()
st.radio(
    "Run mode",
    options=list(_DEMO_RUN_MODE_LABELS.keys()),
    format_func=lambda k: _DEMO_RUN_MODE_LABELS[k],
    key="demo_run_mode",
    horizontal=True,
    disabled=_user_prompt_mode_ui,
    help=(
        "Full: LLM response generation then judge. "
        "Generate only: write response JSON only (no judge). "
        "Judge only: run judge on existing response files (responses are kept). "
        "Disabled while user-input prompts are in use (those runs are generate-only)."
    ),
)
if _user_prompt_mode_ui:
    st.caption(
        "Run mode is disabled: user-input prompt runs always use generate-only (no judge)."
    )


def _selected_bench_ids_session() -> List[str]:
    return [
        b["id"]
        for b in ordered_demo_benchmarks()
        if st.session_state.get(f"bench_run_{b['id']}", False)
    ]


if (
    st.session_state.get("demo_batch_benchmarks", True)
    and len(_selected_bench_ids_session()) >= 2
):
    st.caption(
        "Batching is on: e.g. 5 benchmarks × 5 prompts → 25 generations per model in one "
        "LLM_generate_only subprocess (prompt-injection stays separate)."
    )

# Place RUN EVALUATION at left and make it prominent.
# Disable when a run thread is alive; brief debounce after click to prevent double-starts.
if "run_disabled_until" not in st.session_state:
    st.session_state["run_disabled_until"] = 0.0


def _arm_run_cooldown(seconds: float = 3.0) -> None:
    """Brief debounce only; main lock is run_thread.is_alive()."""
    now = time.time()
    st.session_state["_last_run_clicked"] = now
    st.session_state["run_disabled_until"] = now + seconds


def _run_thread_alive() -> bool:
    t = st.session_state.get("run_thread")
    try:
        return bool(t and hasattr(t, "is_alive") and t.is_alive())
    except Exception:
        return False


def _run_status_path() -> str:
    return os.path.join(LOG_DIR, ".demo_run_status.json")


def _write_run_status(*, running: bool, failed: bool = False, log_path: str = "") -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_run_status_path(), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "running": running,
                    "failed": failed,
                    "log_path": log_path,
                    "ts": time.time(),
                },
                f,
            )
    except Exception:
        pass


def _read_run_status() -> dict:
    try:
        with open(_run_status_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _evaluation_running() -> bool:
    if _run_thread_alive():
        return True
    return bool(_read_run_status().get("running"))


def _release_run_controls(*, failed: bool = False) -> None:
    st.session_state["run_disabled_until"] = 0.0
    st.session_state["last_run_failed"] = failed
    st.session_state["last_run_finished_at"] = time.time()
    st.session_state["demo_run_in_progress"] = False


def _read_log_tail(path: str, *, max_lines: int = 120) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) <= max_lines:
            return "".join(lines)
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


def _prepare_run_log(prefix: str = "evaluate") -> str:
    """Create log file immediately so Live log can display before the worker thread starts."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{prefix}_{timestamp}.log")
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"===== START {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        lf.write("Launching evaluation pipeline...\n")
        lf.flush()
    st.session_state["last_log"] = log_path
    st.session_state["show_log"] = True
    st.session_state["auto_refresh"] = True
    st.session_state["demo_run_in_progress"] = True
    _write_run_status(running=True, log_path=log_path)
    return log_path


# detect running thread (if any)
now_ts = time.time()
cooldown_until = float(st.session_state.get("run_disabled_until", 0.0))
running = _evaluation_running()
disabled = running or (now_ts < cooldown_until)

btn_col, spacer = st.columns([2, 10])
with btn_col:
    # disable button when running or in cooldown
    run_button = st.button("RUN EVALUATION", key="run_eval_btn", disabled=disabled)

progress_bar = st.progress(0)
status_text = st.empty()

if running:
    status_text.info(
        "Evaluation in progress (PLaMo runs can take several minutes). "
        "The log below updates automatically when auto-refresh is on."
    )
    progress_bar.progress(50)
elif st.session_state.get("last_run_finished_at"):
    finished_at = float(st.session_state["last_run_finished_at"])
    if now_ts - finished_at < 30:
        if st.session_state.get("last_run_failed"):
            status_text.warning("Last evaluation finished with errors. See the log.")
        else:
            status_text.success(
                "Last evaluation finished. You can run again or open the dashboard."
            )
        progress_bar.progress(100)
    else:
        progress_bar.empty()
        status_text.empty()
else:
    progress_bar.empty()
    status_text.empty()

# Demo postprocess: ``judge_verify_replay/run_demo_judge_verify.py`` (bundled RQ2 verify).
_RUN_DEMO_JUDGE_VERIFY = os.path.join(
    _SCRIPT_DIR_FOR_CFG, "judge_verify_replay", "run_demo_judge_verify.py"
)


def _pipeline_env(*, hf_offline: bool = False, models: Optional[List[str]] = None) -> dict:
    """PYTHONPATH includes the four-axis-ai-security-eval project root (parent of CybersecurityBenchmarks)."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    project_root = os.path.dirname(BASE)
    prev = env.get("PYTHONPATH", "")
    if project_root not in prev.split(os.pathsep):
        env["PYTHONPATH"] = project_root + (os.pathsep + prev if prev else "")
    if hf_offline:
        for key, value in _DEMO_HF_OFFLINE_ENV_FLAGS.items():
            env.setdefault(key, value)
    if models:
        from CybersecurityBenchmarks.benchmark.plamo_subprocess import apply_plamo_env

        env = apply_plamo_env(env, models, spec_for_runner=_spec_for_runner)
    return env


def _skip_demo_judge_postprocess() -> bool:
    v = os.environ.get("CREATE_EVAL_DEMO_SKIP_JUDGE_POSTPROCESS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _validate_response_files_for_judge(selected_bench_ids: List[str]) -> Optional[str]:
    missing: list[str] = []
    for bid in selected_bench_ids:
        resp_fp = bench_output_paths(OUTPUT_BASE, bid)["responses"]
        if not os.path.isfile(resp_fp):
            missing.append(f"{bid} ({resp_fp})")
    if not missing:
        return None
    return (
        "Judge-only run needs existing response files from a prior generate step. Missing: "
        + "; ".join(missing)
    )


def _clear_outputs_before_run(selected_bench_ids: List[str], run_mode: str) -> None:
    """Full run wipes test output root; generate-only clears selected bench artifacts; judge-only keeps responses."""
    if run_mode == "judge":
        return
    if run_mode == "full":
        if os.path.exists(OUTPUT_BASE):
            shutil.rmtree(OUTPUT_BASE)
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        return
    for bid in selected_bench_ids:
        paths = bench_output_paths(OUTPUT_BASE, bid)
        os.makedirs(paths["dir"], exist_ok=True)
        for key in ("responses", "judge", "stat"):
            fp = paths[key]
            if os.path.isfile(fp):
                os.remove(fp)


def build_pipeline_plan(
    models: List[str],
    selected_bench_ids: List[str],
    run_mode: str,
    batch_benchmarks: bool = True,
) -> tuple[list, bool, dict]:
    """Return (steps, use_batch_plan, ctx). steps are subprocess cmds or inline step dicts."""
    python_exec = sys.executable
    gen_python = _demo_generation_python(models)
    if _demo_batch.should_batch(selected_bench_ids, ui_enabled=batch_benchmarks):
        steps, ctx = _demo_batch.build_batched_pipeline_plan(
            models=models,
            selected_bench_ids=selected_bench_ids,
            run_mode=run_mode,
            output_base=OUTPUT_BASE,
            demo_datasets_dir=DEMO_DATASETS_DIR,
            judge_spec=JUDGE_SPEC,
            python_exec=python_exec,
            bench_by_id=BENCH_BY_ID,
            bench_output_paths_fn=bench_output_paths,
            spec_for_runner=_spec_for_runner,
            generate_python_exec=gen_python,
        )
        return steps, True, ctx
    cmds = _demo_batch.build_per_bench_commands(
        models=models,
        selected_bench_ids=selected_bench_ids,
        run_mode=run_mode,
        output_base=OUTPUT_BASE,
        demo_datasets_dir=DEMO_DATASETS_DIR,
        judge_spec=JUDGE_SPEC,
        python_exec=python_exec,
        bench_by_id=BENCH_BY_ID,
        bench_output_paths_fn=bench_output_paths,
        spec_for_runner=_spec_for_runner,
        generate_python_exec=gen_python,
    )
    return [{"type": _demo_batch.STEP_SUBPROCESS, "label": "run", "cmd": c} for c in cmds], False, {}


def _run_pipeline_step(
    step: dict,
    lf,
    *,
    hf_offline: bool,
    selected_bench_ids: List[str],
    models: List[str],
) -> int:
    """Execute one pipeline step. Returns subprocess exit code (0 if non-subprocess)."""
    python_exec = sys.executable
    stype = step["type"]
    env = _pipeline_env(hf_offline=hf_offline, models=models)

    if stype == _demo_batch.STEP_SUBPROCESS:
        cmd = step["cmd"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            lf.write(line)
            lf.flush()
        return p.wait()

    if stype == _demo_batch.STEP_SPLIT_RESPONSES:
        counts, orphan = _demo_batch.split_combined_records(
            step["merged_path"],
            step["bench_ids"],
            bench_output_paths,
            OUTPUT_BASE,
            "responses",
        )
        lf.write(f"[OK] Split responses -> per-benchmark files ({step['label']})\n")
        for bid, n in counts.items():
            lf.write(f"    {bid}: {n} records\n")
        if orphan:
            lf.write(
                f"*** WARNING: {orphan} record(s) missing demo_bench_id; not assigned ***\n"
            )
        lf.flush()
        return 1 if orphan else 0

    if stype == _demo_batch.STEP_MERGE_RESPONSES_FOR_JUDGE:
        n = _demo_batch.merge_per_bench_response_files(
            step["bench_ids"],
            BENCH_BY_ID,
            bench_output_paths,
            OUTPUT_BASE,
            step["out_path"],
        )
        lf.write(f"[OK] Merged {n} responses for batched judge\n")
        lf.flush()
        return 0

    if stype == _demo_batch.STEP_SPLIT_JUDGE:
        counts, orphan = _demo_batch.split_combined_records(
            step["merged_path"],
            step["bench_ids"],
            bench_output_paths,
            OUTPUT_BASE,
            "judge",
        )
        lf.write("[OK] Split judge outputs -> per-benchmark files\n")
        for bid, n in counts.items():
            lf.write(f"    {bid}: {n} records\n")
        if orphan:
            lf.write(
                f"*** WARNING: {orphan} judge record(s) missing demo_bench_id ***\n"
            )
        lf.flush()
        return 1 if orphan else 0

    lf.write(f"*** Unknown pipeline step type: {stype} ***\n")
    lf.flush()
    return 1


def run_user_prompt_pipeline(
    models: List[str],
    prompt_texts: List[str],
    hf_offline: bool = False,
    log_path: Optional[str] = None,
):
    pipeline_failed = False
    if not log_path:
        log_path = _prepare_run_log(prefix="evaluate_user")
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("Target: user-input prompts (generate only, no judge)\n")
            if hf_offline:
                lf.write(
                    "HF offline mode: enabled "
                    f"({', '.join(_DEMO_HF_OFFLINE_ENV_FLAGS.keys())})\n"
                )
            lf.flush()
            try:
                _demo_user.clear_user_prompt_outputs(OUTPUT_BASE)
                paths = _demo_user.write_run_files(OUTPUT_BASE, prompt_texts)
                lf.write(f"Wrote {len(prompt_texts)} prompt(s) under {paths['dir']}\n")
                lf.flush()
            except Exception as e:
                lf.write(f"*** Failed to prepare prompt files: {e} ***\n")
                lf.flush()
                pipeline_failed = True
                return

            cmd = _demo_user.build_generate_command(
                python_exec=_demo_generation_python(models),
                output_base=OUTPUT_BASE,
                models=models,
                spec_for_runner=_spec_for_runner,
                run_llm_in_parallel=_demo_user.run_llm_parallel(),
            )
            lf.write(f"===== STEP 1/1: {shlex.join(cmd)} =====\n")
            lf.flush()
            env = _pipeline_env(hf_offline=hf_offline, models=models)
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )
            for line in p.stdout:
                lf.write(line)
                lf.flush()
            ret = p.wait()
            lf.write(f"===== STEP 1 exit {ret} =====\n")
            lf.flush()
            if ret != 0:
                lf.write(f"*** Generation failed (exit {ret}). See above. ***\n")
                pipeline_failed = True
            else:
                lf.write(
                    f"[OK] Responses written to {paths['responses']}. "
                    "View them in Data details → User-input prompts.\n"
                )
            lf.write(f"===== END {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            lf.flush()
    finally:
        st.session_state["last_log"] = log_path
        _write_run_status(running=False, failed=pipeline_failed, log_path=log_path)
        try:
            _release_run_controls(failed=pipeline_failed)
        except Exception:
            pass


def run_pipeline(
    models: List[str],
    selected_bench_ids: List[str],
    hf_offline: bool = False,
    run_mode: str = "full",
    batch_benchmarks: bool = True,
    log_path: Optional[str] = None,
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(OUTPUT_BASE)
    os.makedirs(out_base, exist_ok=True)
    steps, batched, batch_ctx = build_pipeline_plan(
        models, selected_bench_ids, run_mode, batch_benchmarks=batch_benchmarks
    )
    total = len(steps)
    if total == 0:
        log_path = log_path or os.path.join(LOG_DIR, f"evaluate_{timestamp}.log")
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"No commands to run (run_mode={run_mode}).\n")
        st.session_state["last_log"] = log_path
        _write_run_status(running=False, failed=False, log_path=log_path)
        try:
            _release_run_controls(failed=False)
        except Exception:
            pass
        return

    if not log_path:
        log_path = os.path.join(LOG_DIR, f"evaluate_{timestamp}.log")
    python_exec = sys.executable
    pipeline_failed = False
    _write_run_status(running=True, log_path=log_path)
    # 書き込みはすべてファイルへ。ここでは streamlit API を呼ばない。
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"Run mode: {_DEMO_RUN_MODE_LABELS.get(run_mode, run_mode)}\n")
        lf.write(
            f"Batch benchmarks: {'on' if batch_benchmarks else 'off'} "
            f"({'batched pipeline' if batched else 'per-benchmark subprocesses'})\n"
        )
        if batched:
            lf.write(
                f"Batched pipeline: {len(selected_bench_ids)} benchmarks "
                f"(work dir: {batch_ctx.get('batch_dir', '')})\n"
            )
            for rb, info in (batch_ctx.get("groups") or {}).items():
                lf.write(
                    f"  - {rb}: {len(info.get('bench_ids', []))} benchmarks, "
                    f"{info.get('n_prompts', 0)} prompts\n"
                )
        lf.flush()
        if hf_offline:
            lf.write(
                "HF offline mode: enabled "
                f"({', '.join(_DEMO_HF_OFFLINE_ENV_FLAGS.keys())})\n"
            )
        from CybersecurityBenchmarks.benchmark.plamo_subprocess import (
            any_model_is_plamo as _any_plamo,
        )

        if _any_plamo(models, _spec_for_runner):
            lf.write(f"PLaMo generate Python: {_demo_generation_python(models)}\n")
        lf.flush()
        for idx, step in enumerate(steps, start=1):
            label = step.get("label") or step.get("type", "step")
            if step.get("type") == _demo_batch.STEP_SUBPROCESS:
                lf.write(f"===== STEP {idx}/{total} ({label}): {shlex.join(step['cmd'])} =====\n")
            else:
                lf.write(f"===== STEP {idx}/{total} ({label}) =====\n")
            lf.flush()
            ret = _run_pipeline_step(
                step,
                lf,
                hf_offline=hf_offline,
                selected_bench_ids=selected_bench_ids,
                models=models,
            )
            lf.write(f"===== STEP {idx} exit {ret} =====\n")
            lf.flush()
            if ret != 0:
                lf.write(f"*** STEP {idx} failed (exit {ret}). See above. ***\n")
                lf.flush()
                pipeline_failed = True
                break

        run_postprocess = run_mode in ("full", "judge")
        if not pipeline_failed and run_postprocess and not _skip_demo_judge_postprocess():
            if not os.path.isfile(_RUN_DEMO_JUDGE_VERIFY):
                lf.write(
                    f"*** POSTPROCESS skipped: missing {_RUN_DEMO_JUDGE_VERIFY} ***\n"
                )
                lf.flush()
            else:
                lf.write(
                    "===== POSTPROCESS: label verify/fix (judge_verify_replay) + stat recompute =====\n"
                )
                lf.flush()
                post_step = 0
                for bid in selected_bench_ids:
                    b = BENCH_BY_ID[bid]
                    paths = bench_output_paths(OUTPUT_BASE, bid)
                    judge_fp = paths["judge"]
                    stat_fp = paths["stat"]
                    task = b["judge_task"]
                    if not os.path.isfile(judge_fp):
                        lf.write(
                            f"*** POSTPROCESS skip bench_id={bid}: judge file not found: {judge_fp} ***\n"
                        )
                        lf.flush()
                        continue

                    verify_cmd: List[str] = [
                        python_exec,
                        _RUN_DEMO_JUDGE_VERIFY,
                        "--judge-json",
                        judge_fp,
                        "--bench",
                        task,
                    ]
                    if task == "harmbench":
                        verify_cmd.append("--fix")
                    verify_cmd.extend(["--fix-unknown", "--write", "--sync-back"])

                    post_step += 1
                    lf.write(
                        f"===== POSTPROCESS STEP {post_step}a verify: {shlex.join(verify_cmd)} =====\n"
                    )
                    lf.flush()
                    env = _pipeline_env(hf_offline=hf_offline, models=models)
                    p = subprocess.Popen(
                        verify_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    for line in p.stdout:
                        lf.write(line)
                        lf.flush()
                    vret = p.wait()
                    lf.write(f"===== POSTPROCESS STEP {post_step}a exit {vret} =====\n")
                    lf.flush()

                    recompute_cmd = [
                        python_exec,
                        "-m",
                        "CybersecurityBenchmarks.benchmark.generic_judge_evaluator",
                        "--recompute-stat-only",
                        f"--judge-response-path={judge_fp}",
                        f"--stat-path={stat_fp}",
                        f"--task={task}",
                    ]
                    post_step += 1
                    lf.write(
                        f"===== POSTPROCESS STEP {post_step}b recompute-stat: {shlex.join(recompute_cmd)} =====\n"
                    )
                    lf.flush()
                    env = _pipeline_env(hf_offline=hf_offline, models=models)
                    p = subprocess.Popen(
                        recompute_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    for line in p.stdout:
                        lf.write(line)
                        lf.flush()
                    sret = p.wait()
                    lf.write(f"===== POSTPROCESS STEP {post_step}b exit {sret} =====\n")
                    lf.flush()
                    if sret != 0:
                        lf.write(
                            f"*** POSTPROCESS stat recompute failed for bench_id={bid} (exit {sret}). ***\n"
                        )
                        lf.flush()
        elif not pipeline_failed and run_postprocess and _skip_demo_judge_postprocess():
            lf.write(
                "===== POSTPROCESS skipped (CREATE_EVAL_DEMO_SKIP_JUDGE_POSTPROCESS is set) =====\n"
            )
            lf.flush()
        elif not pipeline_failed and not run_postprocess:
            lf.write("===== POSTPROCESS skipped (generate-only run) =====\n")
            lf.flush()

        lf.write(f"===== END {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        lf.flush()
    st.session_state["last_log"] = log_path
    _write_run_status(running=False, failed=pipeline_failed, log_path=log_path)
    try:
        _release_run_controls(failed=pipeline_failed)
    except Exception:
        pass

if run_button:
    # recompute running flag in case thread started between reruns
    current_thread = st.session_state.get("run_thread")
    is_running = False
    try:
        if current_thread and hasattr(current_thread, "is_alive"):
            is_running = current_thread.is_alive()
    except Exception:
        is_running = False

    if is_running:
        st.warning("An evaluation is already running. Please wait for it to finish.")
    else:
        prompt_texts = _user_prompt_texts_from_session()
        selected_bench_ids = _selected_bench_ids_session()
        mutual_err = _demo_user.validate_mutual_exclusive(
            prompt_texts, bool(selected_bench_ids)
        )
        if mutual_err:
            st.error(mutual_err)
        elif not st.session_state["models"]:
            st.error("Please add at least one model.")
        elif _demo_user.prompts_configured(prompt_texts):
            user_err = _demo_user.validate_user_prompt_fields_for_run(prompt_texts)
            if user_err:
                st.error(user_err)
            else:
                hf_path_err = _validate_local_hf_model_paths(_model_specs())
                if hf_path_err:
                    st.error(hf_path_err)
                else:
                    hf_offline = bool(st.session_state.get("demo_hf_offline", True))
                    log_path = _prepare_run_log(prefix="evaluate_user")
                    t = threading.Thread(
                        target=run_user_prompt_pipeline,
                        args=(_model_specs(), list(prompt_texts), hf_offline, log_path),
                        daemon=True,
                    )
                    t.start()
                    st.session_state["run_thread"] = t
                    _arm_run_cooldown()
                    st.success(
                        "User-input evaluation started (generate only). "
                        "Check the log or use Refresh log; view responses under Data details."
                    )
                    _safe_rerun()
        else:
            run_mode = st.session_state.get("demo_run_mode", "full")
            needs_models = run_mode in ("full", "generate")
            if needs_models and not st.session_state["models"]:
                st.error("Please add at least one model.")
            elif not selected_bench_ids:
                st.error("Select at least one benchmark to run.")
            else:
                block_err: Optional[str] = None
                if needs_models:
                    block_err = _validate_local_hf_model_paths(_model_specs())
                if not block_err and run_mode == "judge":
                    block_err = _validate_response_files_for_judge(selected_bench_ids)
                if block_err:
                    st.error(block_err)
                else:
                    try:
                        _clear_outputs_before_run(selected_bench_ids, run_mode)
                        if run_mode == "full":
                            st.info("Cleared previous test outputs for this run.")
                        elif run_mode == "generate":
                            st.info(
                                "Cleared response/judge/stat files for selected benchmarks."
                            )
                        else:
                            st.info("Kept existing response files (judge-only run).")
                    except Exception as e:
                        st.warning(f"Failed to clear outputs before run: {e}")

                    hf_offline = bool(st.session_state.get("demo_hf_offline", True))
                    batch_benchmarks = bool(
                        st.session_state.get("demo_batch_benchmarks", True)
                    )
                    models_for_run = _model_specs() if needs_models else []
                    log_path = _prepare_run_log(prefix="evaluate")
                    t = threading.Thread(
                        target=run_pipeline,
                        args=(
                            models_for_run,
                            selected_bench_ids,
                            hf_offline,
                            run_mode,
                            batch_benchmarks,
                            log_path,
                        ),
                        daemon=True,
                    )
                    t.start()
                    st.session_state["run_thread"] = t
                    _arm_run_cooldown()
                    mode_label = _DEMO_RUN_MODE_LABELS.get(run_mode, run_mode)
                    batch_label = (
                        "batched"
                        if _demo_batch.should_batch(
                            selected_bench_ids, ui_enabled=batch_benchmarks
                        )
                        else "per-benchmark"
                    )
                    st.success(
                        f"Evaluation started ({mode_label}, {batch_label}). "
                        "Check the log file or use Refresh log."
                    )
                    _safe_rerun()

# Live log header and controls
st.markdown("---")
# Live log header and inline controls: title on left, refresh/auto on the same line at right
col_title, col_controls = st.columns([8, 4])
with col_title:
    st.subheader("Live log")
with col_controls:
    c1, c2 = st.columns([1, 1])
    with c1:
        # When user explicitly presses Refresh, enable log display for this session
        refresh_clicked = st.button("Refresh log")
        if refresh_clicked:
            st.session_state["show_log"] = True
            # record a manual refresh request in session_state so it survives the rerun
            st.session_state["manual_refresh_request"] = True
            _safe_rerun()
    with c2:
        # give the checkbox a key so its value lives in session_state
        # Default OFF so a fresh page does not show past logs automatically
        auto_refresh = st.checkbox("Auto refresh", key="auto_refresh", value=False, help="Automatically refresh the log view every 2 seconds")
        # if auto refresh is enabled by the user, show logs automatically
        if auto_refresh:
            st.session_state["show_log"] = True

# log display area (controls are above the log)
log_area = st.empty()

# helper to get latest log path
def _get_latest_log_path():
    candidates = []
    lp = st.session_state.get("last_log")
    if lp and os.path.exists(lp):
        candidates.append(lp)
    try:
        files = [f for f in os.listdir(LOG_DIR) if f.startswith("evaluate_")]
        candidates.extend(os.path.join(LOG_DIR, f) for f in files)
    except Exception:
        pass
    existing = [p for p in candidates if p and os.path.exists(p)]
    if not existing:
        return None
    latest = max(existing, key=lambda p: os.path.getmtime(p))
    st.session_state["last_log"] = latest
    return latest

log_path = _get_latest_log_path()
run_status = _read_run_status()
if run_status.get("running") and run_status.get("log_path"):
    log_path = run_status["log_path"]
    st.session_state["last_log"] = log_path
if st.session_state.pop("manual_refresh_request", False):
    # reload by forcing a rerun (will pick up updated last_log)
    _safe_rerun()

# Auto refresh: only enable when not in dashboard mode
if auto_refresh and not st.session_state.get("show_dashboard", False):
    # st_autorefresh is milliseconds; fixed 2s
    count = st_autorefresh(interval=2000, key="auto_refresh_counter")
    # count is the refresh counter (0,1,2,...)

running_for_log = _evaluation_running()
# Show logs while running, when explicitly enabled, or when auto-refresh is on.
should_show_logs = (
    running_for_log
    or st.session_state.get("show_log", False)
    or (auto_refresh and not st.session_state.get("show_dashboard", False))
)

if should_show_logs:
    if log_path and os.path.exists(log_path):
        log_area.code(_read_log_tail(log_path))
        st.caption(f"Latest log: {os.path.basename(log_path)}")
    else:
        log_area.info("No logs yet. Click 'Run evaluation' or 'Refresh log' to start.")
else:
    # hide log area on fresh load when auto_refresh is disabled and user hasn't requested logs
    log_area.empty()
    st.info("Logs are hidden. Enable 'Auto refresh' or press 'Refresh log' to display.")

# Inline dashboard runner: dashboard.py をほぼそのまま読み込んで同ページ内に表示する
dash_path = os.path.join(os.path.dirname(__file__), "dashboard_eval_ui.py")

st.markdown("---")
st.subheader("Step 3. Evaluation results")
st.caption("Click 'Show dashboard' to load the dashboard view.")
# Dashboard loader: toggle session_state['show_dashboard'] to disable auto-refresh while dashboard is shown
if st.session_state.get("show_dashboard", False):
    # show hide button to go back to normal logs view
    if st.button("Hide dashboard"):
        st.session_state["show_dashboard"] = False
        # request enabling auto-refresh on next rerun (avoid modifying widget-backed key now)
        st.session_state["enable_auto_refresh_next_run"] = True
        _safe_rerun()
    with st.spinner("Loading dashboard..."):
        try:
            # OUTPUT_BASEをグローバル変数として渡す
            runpy.run_path(dash_path, run_name="__main__", init_globals={"OUTPUT_BASE": OUTPUT_BASE})
        except Exception as e:
            st.error(f"Failed to load dashboard: {e}")
else:
    if st.button("Show dashboard"):
        script_path = os.path.join(os.path.dirname(__file__), "merge_base_test.py")
        python_exec = sys.executable or "python3"
        cmd = [python_exec, script_path, OUTPUT_BASE]
        # run merge silently; show only errors to the user and persist details to a log
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
            if res.returncode != 0:
                try:
                    os.makedirs(LOG_DIR, exist_ok=True)
                    with open(os.path.join(LOG_DIR, "merge_error.log"), "a", encoding="utf-8") as lf:
                        lf.write(f"{time.time()} MERGE FAILED cmd={' '.join(shlex.quote(p) for p in cmd)}\n")
                        lf.write(res.stdout or "")
                        lf.write("\n--- STDERR ---\n")
                        lf.write(res.stderr or "")
                        lf.write("\n\n")
                except Exception:
                    pass
                st.error("Merge failed.")
        except Exception as e:
            st.error(f"Failed to run merge script: {e}")

        # set flags so next rerun will disable auto-refresh before widget recreation
        st.session_state["show_dashboard"] = True
        st.session_state["disable_auto_refresh_next_run"] = True
        # load dashboard (merged outputs are expected to have been written by the script above)
        with st.spinner("Loading dashboard..."):
            try:
                # OUTPUT_BASEをグローバル変数として渡す
                runpy.run_path(dash_path, run_name="__main__", init_globals={"OUTPUT_BASE": OUTPUT_BASE})
            except Exception as e:
                st.error(f"Failed to load dashboard: {e}")
