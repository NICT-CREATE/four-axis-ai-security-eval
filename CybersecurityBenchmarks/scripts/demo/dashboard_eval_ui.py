import importlib.util
import json
import os

import pandas as pd
import streamlit as st
import altair as alt

# Demo UI policy: do not edit CybersecurityBenchmarks/benchmark/* for demo-only behavior.
# All user-visible strings in this Streamlit app must be English.

# Resolve CybersecurityBenchmarks directory from this script location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_cfg_spec = importlib.util.spec_from_file_location(
    "demo_benchmark_config",
    os.path.join(SCRIPT_DIR, "demo_benchmark_config.py"),
)
_demo_cfg = importlib.util.module_from_spec(_cfg_spec)
assert _cfg_spec.loader is not None
_cfg_spec.loader.exec_module(_demo_cfg)
_up_spec = importlib.util.spec_from_file_location(
    "demo_user_prompts",
    os.path.join(SCRIPT_DIR, "demo_user_prompts.py"),
)
_demo_user = importlib.util.module_from_spec(_up_spec)
assert _up_spec.loader is not None
_up_spec.loader.exec_module(_demo_user)
_std_spec = importlib.util.spec_from_file_location(
    "demo_standardized_metrics",
    os.path.join(SCRIPT_DIR, "demo_standardized_metrics.py"),
)
_demo_std = importlib.util.module_from_spec(_std_spec)
assert _std_spec.loader is not None
_std_spec.loader.exec_module(_demo_std)
CYBERSECURITY_BENCHMARKS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # CybersecurityBenchmarksディレクトリ（scripts/demo/から2階層上）

# データファイルのパス: evaluate_ui.pyのOUTPUT_BASEと同じパスを使用
# evaluate_ui.pyからrunpy.run_pathで読み込まれる場合、OUTPUT_BASEがグローバル変数として渡される
# そうでない場合は、evaluate_ui.pyと同じ方法でパスを計算
try:
    # runpy.run_pathのinit_globalsで渡されたOUTPUT_BASEを使用
    DATA_ROOT = OUTPUT_BASE
except NameError:
    # 直接実行された場合、またはOUTPUT_BASEが渡されていない場合
    # evaluate_ui.pyと同じ方法でパスを計算
    DATA_ROOT = os.path.join(CYBERSECURITY_BENCHMARKS_DIR, "outputs", "demo", "test_v2", "with_syspro")

# --- merged outputs handling ---
# TEST_ROOT (the current DATA_ROOT), BASE_ROOT (sibling base_v2), MERGED_ROOT (merge_v2)
TEST_ROOT = os.path.normpath(DATA_ROOT)
BASE_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(DATA_ROOT)), "base_v2", os.path.basename(DATA_ROOT)))
MERGED_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(DATA_ROOT)), "merge_v2", os.path.basename(DATA_ROOT)))

def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _merge_and_write_all():
    """Merge base+test for each TASK and write merged files under MERGED_ROOT preserving relative paths."""
    try:
        for t in TASKS:
            # stat
            stat_base_path = os.path.join(BASE_ROOT, t["stat"])
            stat_test_path = os.path.join(TEST_ROOT, t["stat"])
            sb = {}
            stt = {}
            if os.path.exists(stat_base_path):
                try:
                    sb = load_json(stat_base_path) or {}
                except Exception:
                    sb = {}
            if os.path.exists(stat_test_path):
                try:
                    stt = load_json(stat_test_path) or {}
                except Exception:
                    stt = {}
            merged_stat = _merge_stat_dicts(sb, stt)
            out_stat_path = os.path.join(MERGED_ROOT, t["stat"])
            _ensure_parent_dir(out_stat_path)
            with open(out_stat_path, "w", encoding="utf-8") as f:
                json.dump(merged_stat, f, ensure_ascii=False, indent=2)

            # judge (concatenate lists)
            judge_base_path = os.path.join(BASE_ROOT, t["judge"])
            judge_test_path = os.path.join(TEST_ROOT, t["judge"])
            merged_j = _merge_judge_lists(judge_base_path, judge_test_path)
            out_judge_path = os.path.join(MERGED_ROOT, t["judge"])
            _ensure_parent_dir(out_judge_path)
            with open(out_judge_path, "w", encoding="utf-8") as f:
                json.dump(merged_j, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        # do not break UI on failure
        return False

# If a merge marker exists under TEST_ROOT, perform merge now and remove marker.
merge_marker = os.path.join(TEST_ROOT, ".merge_requested")
if os.path.exists(merge_marker):
    _merge_and_write_all()
    try:
        os.remove(merge_marker)
    except Exception:
        pass

# If merged outputs exist, use them as DATA_ROOT for the remainder of this script.
if os.path.exists(MERGED_ROOT):
    DATA_ROOT = MERGED_ROOT

_HF_QWEN_M = _demo_cfg.demo_remap_hf_local_path_in_spec(
    _demo_cfg.demo_local_qwen_hf16_join("notebooks", "qwen", "Qwen_3-8b_multidim")
)
_HF_QWEN_O = _demo_cfg.demo_remap_hf_local_path_in_spec(
    _demo_cfg.demo_local_qwen_hf16_join("notebooks", "Qwen", "Qwen3-8B_orthogonalized")
)

PLAMO_2B_MODEL_ID = "pfnet/plamo-3-nict-2b-base"
PLAMO_8B_MODEL_ID = "pfnet/plamo-3-nict-8b-base"
PLAMO_MODEL_ID = PLAMO_2B_MODEL_ID  # backward-compatible alias
PLAMO_MODEL_IDS = (PLAMO_8B_MODEL_ID, PLAMO_2B_MODEL_ID)

MODEL_ORDER = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3-8B",
    _HF_QWEN_O,
    _HF_QWEN_M,
    PLAMO_8B_MODEL_ID,
    PLAMO_2B_MODEL_ID,
]
MODEL_LABELS = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1 (8B)",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "Deepseek-R1 (8B)",
    "Qwen/Qwen3-8B": "Qwen3 (8B)",
    "google/gemma-3-4b-it": "Gemma3 (4B)",
    PLAMO_2B_MODEL_ID: "PLaMo 3 (2B base)",
    PLAMO_8B_MODEL_ID: "PLaMo 3 (8B base)",
    _HF_QWEN_M: "Qwen3 multidim",
    _HF_QWEN_O: "Qwen3 orthogonalized",
}


def _model_order_key(model_id: str) -> str:
    """Map a stored model id (HF hub id or absolute path) to a MODEL_ORDER token."""
    mid = str(model_id).strip()
    if mid in MODEL_ORDER:
        return mid

    path = mid
    if "::" in path:
        path = path.split("::", 1)[1].strip()
    norm = os.path.normpath(os.path.expanduser(path))
    base = os.path.basename(norm)

    if mid == PLAMO_8B_MODEL_ID or base == "plamo-3-nict-8b-base":
        return PLAMO_8B_MODEL_ID
    if mid == PLAMO_2B_MODEL_ID or base == "plamo-3-nict-2b-base":
        return PLAMO_2B_MODEL_ID
    if base == "Qwen3-8B_orthogonalized" or norm.endswith(
        os.path.join("Qwen", "Qwen3-8B_orthogonalized")
    ):
        return _HF_QWEN_O
    if base == "Qwen_3-8b_multidim" or norm.endswith(
        os.path.join("qwen", "Qwen_3-8b_multidim")
    ):
        return _HF_QWEN_M

    for entry in MODEL_ORDER:
        if mid == entry or base == entry.split("/")[-1]:
            return entry
    return mid


def _sort_model_ids(model_ids: list[str]) -> list[str]:
    """Sort ids: MODEL_ORDER first (Qwen group, then PLaMo), then unknown ids."""
    order_ix = {m: i for i, m in enumerate(MODEL_ORDER)}

    def key(mid: str):
        k = _model_order_key(mid)
        if k in order_ix:
            return (0, order_ix[k], mid)
        return (1, mid)

    return sorted(model_ids, key=key)


def _default_chart_model_labels(lbl_order: list[str], chart_label_by_id: dict[str, str]) -> list[str]:
    """Chart models shown on first load (PLaMo variants hidden until user selects them)."""
    plamo_lbls = {chart_label_by_id[mid] for mid in PLAMO_MODEL_IDS if mid in chart_label_by_id}
    if plamo_lbls:
        return [lbl for lbl in lbl_order if lbl not in plamo_lbls]
    return lbl_order[:]
# Descriptions for variants/tasks/methods (English)
PROMPT_INJECTION_VARIANT_DESC = {
    "ignore_previous_instructions": "Explicitly instruct to ignore previous instructions to break system constraints.",
    "indirect_reference": "Request sensitive information indirectly without direct reference.",
    "token_smuggling": "Use encoding/obfuscation to extract forbidden information.",
    "system_mode": "Forge privileged modes such as SYSTEM/GOD/MAINTENANCE MODE.",
    "different_user_input_language": "Use a different language to bypass multilingual defenses.",
    "repeated_token_attack": "Use repeated meaningless tokens to disrupt defenses.",
    "persuasion": "Use authority or threats to persuade and elicit responses.",
    "payload_splitting": "Split attack payload into pieces and recombine to bypass filters.",
    "output_formatting_manipulation": "Force output format to cause information leakage.",
    "hypothetical_scenario": "Justify forbidden info using hypothetical scenarios.",
    "virtualization": "Use virtual or future settings to evade constraints.",
    "few_shot_attack": "Provide repeated Q&A examples, then append the forbidden task.",
    "many_shot_attack": "Provide many examples to bury system constraints.",
    "overload_with_information": "Flood with irrelevant queries then sneak in a forbidden request.",
    "mixed_techniques": "Combine multiple attack techniques into a hybrid approach.",
}

JAILBREAK_VARIANT_DESC = {
    "Begin with fixed sentence": "Start generation with a specific opening sentence to bypass refusals.",
    "fixed sentence": "Start generation with a specific opening sentence to bypass refusals.",
    "Begin with bad words": "Start with insults/offensive words to confuse defenses.",
    "bad words": "Start with insults/offensive words to confuse defenses.",
    "Encoded string": "Hide harmful instructions via URL-encoding or similar encodings.",
    "encoded string": "Hide harmful instructions via URL-encoding or similar encodings.",
    "No punctuation": "Remove punctuation to break refusal templates.",
    "no punctuation": "Remove punctuation to break refusal templates.",
    "No long words": "Ban long words to disrupt refusal template matching.",
    "no long words": "Ban long words to disrupt refusal template matching.",
    "No \u201cthe\u201d": "Ban specific words (like 'the') to invalidate refusal templates.",
    "no the": "Ban specific words (like 'the') to invalidate refusal templates.",
    "JSON format output": "Force JSON output to bypass refusal handling.",
    "json format": "Force JSON output to bypass refusal handling.",
    "Refusal sentence prohibition": "Prohibit refusal words to force dangerous responses.",
    "refusal prohibition": "Prohibit refusal words to force dangerous responses.",
    "Leetspeak": "Use character substitutions (1337) to evade filters.",
    "leetspeak": "Use character substitutions (1337) to evade filters.",
    "Scenario setting": "Use role/situation settings to weaken defenses.",
    "scenario setting": "Use role/situation settings to weaken defenses.",
    "CoT (Chain-of-thought)": "Force step-by-step reasoning to leak details.",
    "cot": "Force step-by-step reasoning to leak details.",
    "Multi-task": "Mix harmless and harmful questions to bypass filters.",
    "multi task": "Mix harmless and harmful questions to bypass filters.",
    "Programming function": "Embed harmful tasks inside code/function formats.",
    "programming function": "Embed harmful tasks inside code/function formats.",
    "COU (Cooperative Output Undermining)": "Use multi-model cooperation to induce forbidden outputs and ban refusal expressions.",
    "cou": "Use multi-model cooperation to induce forbidden outputs and ban refusal expressions.",
}

ADVGLUE_TASK_DESC = {
    "sst2": "SST-2 (Sentiment classification)",
    "qqp": "QQP (Duplicate question detection)",
    "mnli": "MNLI (Natural Language Inference)",
    "qnli": "QNLI (Question-answering NLI)",
}

ADVGLUE_METHOD_DESC = {
    "SememePSO": "PSO-optimized synonym replacement based on sememe semantics.",
    "TextFooler": "Synonym substitution of important words to induce misclassification.",
    "TextBugger": "Character-level edits (insert/delete/replace) to inject spelling noise.",
    "SemAttack": "Meaning-preserving synonym replacement under semantic constraints.",
    "BERT-ATTACK": "Use BERT context predictions for natural word substitutions.",
    "T3": "Paraphrase generation model-based rephrasing.",
    "SCPN": "Syntactic-controlled paraphrase to change sentence structure.",
    "StressTest": "Simple stress operations (repetition, numeric additions, etc.).",
    "CheckList": "Linguistic transformations like negation, added questions, punctuation tweaks.",
}
TASKS = _demo_cfg.dashboard_tasks_from_demo()
USER_PROMPT_TASK = _demo_user.USER_PROMPT_TASK
DATA_DETAILS_TASKS = [USER_PROMPT_TASK] + TASKS

TASK_SOURCES = {
    t["bench_id"]: _demo_cfg.BENCH_BY_ID[t["bench_id"]]["source"] for t in TASKS
}

def _altair_std_score_y() -> alt.Y:
    return alt.Y(
        "Score:Q",
        title="Score (%)",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%"),
    )


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_model_label(model):
    """Return simplified label for a model id using MODEL_LABELS mapping.
    Fallback: show last path segment if not mapped (e.g. 'org/model' -> 'model')."""
    key = _model_order_key(model)
    if key in MODEL_LABELS:
        return MODEL_LABELS[key]
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    if isinstance(model, str) and "/" in model:
        return model.split("/")[-1]
    return str(model)


def _detail_json_records(
    g1: pd.DataFrame,
    order: list[str],
    *,
    drop_cols: tuple[str, ...] = (),
) -> list[dict]:
    """JSON rows for Data details expander: always include model_id and model_label."""
    if g1.empty or "model" not in g1.columns:
        return g1.to_dict(orient="records")
    by_model = g1.drop_duplicates(subset=["model"], keep="last").set_index("model")
    skip = set(drop_cols)
    records: list[dict] = []
    for mid in order:
        if mid not in by_model.index:
            continue
        row = by_model.loc[mid].to_dict()
        for k in skip:
            row.pop(k, None)
        rec = {"model_id": mid, "model_label": get_model_label(mid)}
        rec.update(row)
        records.append(rec)
    return records


def _model_label_suffix(model: str) -> str:
    """Short stable suffix used only when multiple model ids share a display label."""
    text = str(model)
    if "::" in text:
        text = text.split("::", 1)[1]
    text = os.path.normpath(os.path.expanduser(text.strip()))
    if os.path.isabs(text):
        base = os.path.basename(text)
        parent = os.path.basename(os.path.dirname(text))
        return f"{parent}/{base}" if parent else base
    return text.split("/")[-1] if "/" in text else text


def _unique_model_labels(model_ids: list[str]) -> dict[str, str]:
    """
    Return chart/table labels that stay unique even when display names collide.

    Altair groups bars/legend entries by the encoded text value. If two distinct
    model ids are both rendered as the same label (for example a saved baseline
    model and a renamed local checkpoint both shown as "Qwen3 orthogonalized"),
    the chart can collapse them into one visual series even though the raw stat
    rows are separate. Tables are less affected because they preserve rows, but
    charts need a unique encoded label.

    Keep the normal human-readable label when it is unique. Only duplicated
    labels receive a short suffix derived from the model id/path, so the common
    case stays clean while collision cases remain distinguishable.
    """
    base_labels = {mid: get_model_label(mid) for mid in model_ids}
    label_counts: dict[str, int] = {}
    for label in base_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    used: set[str] = set()
    out: dict[str, str] = {}
    for mid in model_ids:
        label = base_labels[mid]
        candidate = label
        if label_counts.get(label, 0) > 1:
            candidate = f"{label} [{_model_label_suffix(mid)}]"
        if candidate in used:
            i = 2
            base = candidate
            while f"{base} #{i}" in used:
                i += 1
            candidate = f"{base} #{i}"
        used.add(candidate)
        out[mid] = candidate
    return out


def _normalize(s: object) -> str:
    """Normalize a string for loose matching: lower, strip, replace '_' and '-' with spaces."""
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    return " ".join(t.split())


def _find_description(desc_map: dict, variant: object) -> str:
    """Find a best-effort description in desc_map for variant using loose normalization/membership.

    Returns empty string when no match found.
    """
    if not variant:
        return ""
    v_norm = _normalize(variant)
    # try direct lookup first
    if variant in desc_map:
        return desc_map[variant]
    # try normalized comparisons
    for k, v in desc_map.items():
        k_norm = _normalize(k)
        if k_norm == v_norm or k_norm in v_norm or v_norm in k_norm:
            return v
    return ""


# Sidebar: page and task (Data details includes user-input prompts task)
def _task_display_name(t: dict) -> str:
    if t.get("is_user_prompts"):
        return "User-input prompts (generate only)"
    src = TASK_SOURCES.get(t.get("bench_id", ""), "")
    if src:
        return f"{t['name']} ({src})"
    return t["name"]


data_details_display_names = [_task_display_name(t) for t in DATA_DETAILS_TASKS]

page = st.sidebar.selectbox("Select page", ["Dashboard", "Data details"])
if page == "Data details":
    task_idx = st.sidebar.selectbox(
        "Select task",
        range(len(data_details_display_names)),
        format_func=lambda i: data_details_display_names[i],
        key="dashboard_task_idx",
    )
    task = DATA_DETAILS_TASKS[task_idx]
else:
    task_idx = st.session_state.get("dashboard_task_idx", 0)
    if task_idx >= len(DATA_DETAILS_TASKS):
        task_idx = 0
    task = DATA_DETAILS_TASKS[task_idx]

is_user_prompt_task = bool(task.get("is_user_prompts"))
task_key = task["task_key"]

TEST_ROOT = os.path.normpath(DATA_ROOT)
BASE_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.dirname(DATA_ROOT)),
        "base_v2",
        os.path.basename(DATA_ROOT),
    )
)


def _merge_stat_dicts(base_stat: dict, test_stat: dict) -> dict:
    """Merge two stat dicts: prefer values from test_stat when keys conflict.
    Perform a shallow merge per-model, and deep-merge nested dicts where appropriate."""
    if not isinstance(base_stat, dict):
        base_stat = {}
    if not isinstance(test_stat, dict):
        test_stat = {}
    merged = {}
    # start with base
    for k, v in base_stat.items():
        merged[k] = v
    # merge/override with test
    for k, v in test_stat.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            # merge nested dicts shallowly (tasks/overall etc.)
            merged[k] = {**merged[k], **v}
            # for nested 'tasks' dictionaries, merge per-task as well
            if isinstance(merged[k].get("tasks"), dict) and isinstance(v.get("tasks"), dict):
                merged[k]["tasks"] = {**merged[k].get("tasks", {}), **v.get("tasks", {})}
        else:
            merged[k] = v
    return merged


def _merge_judge_lists(base_judge_path: str, test_judge_path: str):
    """Load judge JSON lists from base/test paths and concatenate (base first, test second)."""
    base_list = []
    test_list = []
    try:
        if base_judge_path and os.path.exists(base_judge_path):
            base_list = load_json(base_judge_path) or []
    except Exception:
        base_list = []
    try:
        if test_judge_path and os.path.exists(test_judge_path):
            test_list = load_json(test_judge_path) or []
    except Exception:
        test_list = []
    # ensure lists
    if not isinstance(base_list, list):
        base_list = []
    if not isinstance(test_list, list):
        test_list = []
    return base_list + test_list


stat: dict = {}
judge: list = []
order = MODEL_ORDER[:]

if page == "Data details" and not is_user_prompt_task:
    stat_test_path = os.path.join(TEST_ROOT, task["stat"])
    stat_base_path = os.path.join(BASE_ROOT, task["stat"])
    judge_test_path = os.path.join(TEST_ROOT, task["judge"])
    judge_base_path = os.path.join(BASE_ROOT, task["judge"])

    stat_base = {}
    stat_test = {}
    if os.path.exists(stat_base_path):
        try:
            stat_base = load_json(stat_base_path) or {}
        except Exception:
            stat_base = {}
    if os.path.exists(stat_test_path):
        try:
            stat_test = load_json(stat_test_path) or {}
        except Exception:
            stat_test = {}

    stat = _merge_stat_dicts(stat_base, stat_test)
    judge = _merge_judge_lists(judge_base_path, judge_test_path)

    models_present = []
    if isinstance(stat, dict):
        for k, v in stat.items():
            if not isinstance(v, dict):
                continue
            key_l = str(k).lower()
            if key_l.startswith("stat") or "per_model" in key_l or "stat_" in key_l:
                continue
            models_present.append(k)

    order = _sort_model_ids(models_present)
    if not order:
        order = MODEL_ORDER[:]

display_label = _task_display_name(task)

if page == "Dashboard":
    st.title(
        "Statistics Dashboard"
    )

    # stat は上で base+test マージ済み。ここで test のみ読み直さない。

    # --- Cross-bench standardized scores (RQ2 Phase 1: higher = better) ---
    try:
        _by_axis, _by_bench, _std_models = _demo_std.compute_standardized_snapshot(
            TASKS,
            TEST_ROOT,
            BASE_ROOT,
            load_json,
            _merge_stat_dicts,
            _demo_cfg.AXIS_BENCHMARK_IDS,
            MODEL_ORDER,
        )
    except Exception:
        _by_axis, _by_bench, _std_models = {}, {}, []
    _std_models = _sort_model_ids(_std_models)

    st.subheader("Standardized scores (all benchmarks, higher = better)")
    st.caption(
        "Baseline models are shown by default; **PLaMo 3 (2B/8B base)** are hidden until you add them under "
        "**Models shown in charts** below. Expand the score tables for numeric details."
    )
    if _std_models and _by_axis:
        axis_specs = _demo_cfg.DEMO_AXES
        axis_ids = [ax["id"] for ax in axis_specs]
        axis_labels = {ax["id"]: ax["label"] for ax in axis_specs}
        axis_metric = {ax["id"]: ax["standardized_metric"] for ax in axis_specs}
        chart_label_by_id = _unique_model_labels(_std_models)
        lbl_order = [chart_label_by_id[m] for m in _std_models]
        default_visible = _default_chart_model_labels(lbl_order, chart_label_by_id)
        axis_order = [axis_labels[i] for i in axis_ids]
        visible_models = st.session_state.get("dashboard_visible_models", default_visible)
        visible_models = [m for m in visible_models if m in lbl_order]
        if not visible_models:
            visible_models = default_visible or lbl_order
        st.session_state["dashboard_visible_models"] = visible_models
        visible_set = set(visible_models)
        sum_rows = []
        for mid in _std_models:
            row = {"Model": chart_label_by_id[mid]}
            for axid in axis_ids:
                v = _by_axis.get(mid, {}).get(axid)
                col = f"{axis_labels[axid]} ({axis_metric[axid]})"
                row[col] = v if v is not None else float("nan")
            sum_rows.append(row)
        sum_df = pd.DataFrame(sum_rows)
        fmt = {c: "{:.1%}" for c in sum_df.columns if c != "Model"}
        with st.expander("4-axis summary score table"):
            st.dataframe(sum_df.style.format(fmt))

        long_rows = []
        for mid in _std_models:
            lbl = chart_label_by_id[mid]
            if lbl not in visible_set:
                continue
            for axid in axis_ids:
                v = _by_axis.get(mid, {}).get(axid)
                if v is None:
                    continue
                long_rows.append(
                    {
                        "Model": lbl,
                        "Axis": axis_labels[axid],
                        "Score": v,
                    }
                )
        chart_color_order = [m for m in lbl_order if m in visible_set] or lbl_order
        _n_mod = max(len(chart_color_order), 1)
        _axis_h = 340

        bench_long: list[dict] = []
        for mid in _std_models:
            mlbl = chart_label_by_id[mid]
            if mlbl not in visible_set:
                continue
            for axid in axis_ids:
                for bid in _demo_cfg.AXIS_BENCHMARK_IDS[axid]:
                    v = _by_bench.get(mid, {}).get(bid)
                    if v is None:
                        continue
                    bench_long.append(
                        {
                            "Model": mlbl,
                            "Benchmark": _demo_cfg.BENCH_BY_ID[bid]["label"],
                            "Axis": axis_labels[axid],
                            "Score": v,
                        }
                    )

        if long_rows:
            ldf = pd.DataFrame(long_rows)
            ldf["Model"] = pd.Categorical(ldf["Model"], categories=chart_color_order, ordered=True)
            ldf["Axis"] = pd.Categorical(ldf["Axis"], categories=axis_order, ordered=True)
            if hasattr(alt, "XOffset"):
                axis_chart = (
                    alt.Chart(ldf)
                    .mark_bar()
                    .encode(
                        x=alt.X("Axis:N", sort=axis_order, title="Axis"),
                        xOffset=alt.XOffset("Model:N", sort=chart_color_order),
                        y=_altair_std_score_y(),
                        color=alt.Color("Model:N", sort=chart_color_order, title="Model"),
                    )
                    .properties(width=alt.Step(36 * _n_mod), height=_axis_h)
                    .configure_view(continuousHeight=_axis_h)
                )
            else:
                axis_chart = (
                    alt.Chart(ldf)
                    .mark_bar()
                    .encode(
                        x=alt.X("Model:N", sort=chart_color_order, title="Model", axis=alt.Axis(labelAngle=-45)),
                        y=_altair_std_score_y(),
                        color=alt.Color("Model:N", sort=chart_color_order, title="Model"),
                        column=alt.Column(
                            "Axis:N",
                            sort=axis_order,
                            header=alt.Header(labelOrient="bottom", title="Axis"),
                        ),
                    )
                    .properties(width=alt.Step(28), height=_axis_h)
                    .configure_view(continuousHeight=_axis_h)
                )
            st.altair_chart(axis_chart, use_container_width=True)
        elif visible_set != set(lbl_order):
            st.caption("No axis chart data for the selected models.")

        st.subheader("Per-benchmark standardized scores")

        with st.expander("Per-benchmark score table"):
            bench_order = [t["bench_id"] for t in TASKS if t.get("bench_id")]
            det_rows = []
            for mid in _std_models:
                bmap = _by_bench.get(mid, {})
                r = {"Model": chart_label_by_id[mid]}
                for bid in bench_order:
                    if bid in bmap:
                        r[bid] = bmap[bid]
                det_rows.append(r)
            det_df = pd.DataFrame(det_rows)
            dfmt = {c: "{:.1%}" for c in det_df.columns if c != "Model"}
            st.dataframe(det_df.style.format(dfmt) if dfmt else det_df)

        if bench_long:
            st.caption(
                "Four panels (one per axis). Each panel lists only that axis's benchmarks; "
                "models are compared side by side within each benchmark."
            )
            bdf = pd.DataFrame(bench_long)
            bdf["Model"] = pd.Categorical(bdf["Model"], categories=chart_color_order, ordered=True)
            _bench_panel_h = 200
            _bench_legend = alt.Legend(
                orient="top",
                direction="horizontal",
                labelLimit=180,
                title="Model",
            )
            _bench_panels: list[alt.Chart] = []
            for _ax_i, axid in enumerate(axis_ids):
                ax_lbl = axis_labels[axid]
                bench_labels_ax = [
                    _demo_cfg.BENCH_BY_ID[bid]["label"]
                    for bid in _demo_cfg.AXIS_BENCHMARK_IDS[axid]
                ]
                sub = bdf[bdf["Axis"] == ax_lbl].copy()
                if sub.empty:
                    continue
                sub["Benchmark"] = pd.Categorical(
                    sub["Benchmark"], categories=bench_labels_ax, ordered=True
                )
                _panel_title = f"{ax_lbl} — {axis_metric[axid]}"
                if hasattr(alt, "XOffset"):
                    panel = (
                        alt.Chart(sub)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Benchmark:N",
                                sort=bench_labels_ax,
                                title="Benchmark",
                                axis=alt.Axis(labelAngle=-30, labelLimit=160),
                            ),
                            xOffset=alt.XOffset("Model:N", sort=chart_color_order),
                            y=_altair_std_score_y(),
                            color=alt.Color(
                                "Model:N",
                                sort=chart_color_order,
                                title="Model",
                                legend=_bench_legend if len(_bench_panels) == 0 else None,
                            ),
                        )
                        .properties(
                            width=alt.Step(22 * _n_mod),
                            height=_bench_panel_h,
                            title={
                                "text": _panel_title,
                                "anchor": "start",
                                "fontSize": 14,
                                "dy": -6,
                            },
                        )
                    )
                else:
                    panel = (
                        alt.Chart(sub)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Benchmark:N",
                                sort=bench_labels_ax,
                                title="Benchmark",
                                axis=alt.Axis(labelAngle=-30),
                            ),
                            y=_altair_std_score_y(),
                            color=alt.Color(
                                "Model:N",
                                sort=chart_color_order,
                                title="Model",
                                legend=_bench_legend if len(_bench_panels) == 0 else None,
                            ),
                            column=alt.Column("Model:N", sort=chart_color_order),
                        )
                        .properties(
                            width=alt.Step(36),
                            height=_bench_panel_h,
                            title={"text": _panel_title, "anchor": "start", "fontSize": 14},
                        )
                    )
                _bench_panels.append(panel)

            if _bench_panels:
                bench_combined = (
                    alt.vconcat(*_bench_panels, spacing=16)
                    .resolve_scale(x="independent", y="shared")
                    .configure_view(continuousHeight=_bench_panel_h)
                )
                if hasattr(bench_combined, "resolve_legend"):
                    bench_combined = bench_combined.resolve_legend(color="shared")
                st.altair_chart(bench_combined, use_container_width=True)

        st.multiselect(
            "Models shown in charts",
            options=lbl_order,
            default=default_visible,
            help=(
                "Baseline models are selected by default. Add **PLaMo 3 (2B/8B base)** here to include them "
                "in the axis and per-benchmark charts."
            ),
            key="dashboard_visible_models",
        )
    else:
        st.info("No standardized scores yet (run evaluations so stat JSON files exist under the output directory).")

# Data details
else:
    if is_user_prompt_task:
        st.title(display_label)
        resp_path = os.path.join(DATA_ROOT, task["responses"])
        if not os.path.isfile(resp_path):
            st.info(
                "No user-input responses yet. Enter prompts in Step 1 and run evaluation "
                "(user-input mode) on the main page."
            )
            st.stop()

        df = pd.DataFrame(load_json(resp_path))
        if df.empty:
            st.info("Response file is empty.")
            st.stop()

        if "prompt_id" not in df.columns:
            st.error("Invalid user-input responses file (missing prompt_id).")
            st.stop()

        models_in_df = []
        if "model" in df.columns:
            for m in df["model"].dropna().unique():
                if m not in models_in_df:
                    models_in_df.append(m)
        order = _sort_model_ids(models_in_df)
        if not order:
            order = models_in_df or MODEL_ORDER[:]

        per_page = st.sidebar.selectbox("Items per page", [1, 5, 10, 20], index=1)
        unique_prompt_ids = sorted(df["prompt_id"].drop_duplicates().tolist())
        total_prompts = len(unique_prompt_ids)
        max_page = max(1, (total_prompts + per_page - 1) // per_page)
        page_num = st.sidebar.number_input(
            "Page number", min_value=1, max_value=max_page, value=1, step=1
        )
        start_idx = (int(page_num) - 1) * per_page
        show_prompt_ids = unique_prompt_ids[start_idx : start_idx + per_page]

        st.markdown("**Generate only** — no judge labels. Use **Details (JSON)** for metadata.")
        st.markdown(f"**Unique prompts:** {total_prompts}")

        search_query = st.sidebar.text_input("Search (any field)", value="")
        if search_query:
            mask = pd.Series(False, index=df.index)
            for c in df.columns:
                mask = mask | df[c].astype(str).str.contains(search_query, case=False, na=False)
            matches = df[mask]
            st.subheader(f"Search results ({matches['prompt_id'].nunique()} prompts)")
            for pid in matches["prompt_id"].drop_duplicates().tolist():
                group = matches[matches["prompt_id"] == pid]
                st.subheader(f"prompt_id: {pid}")
                st.write(
                    "Prompt:",
                    group["test_case_prompt"].iloc[0]
                    if "test_case_prompt" in group.columns
                    else "",
                )
                g1 = (
                    group.drop_duplicates(subset=["model"], keep="last")
                    if "model" in group.columns
                    else group
                )
                if "model" in g1.columns:
                    g1_disp = g1.set_index("model").reindex(order)
                    g1_disp.index = [get_model_label(m) for m in g1_disp.index]
                else:
                    g1_disp = g1
                show_cols = [c for c in ("response", "processing_time", "token_count") if c in g1_disp.columns]
                if show_cols:
                    st.dataframe(g1_disp[show_cols])
                zw = "\u200b"
                ver = st.session_state.get("_ui_state_version", 0)
                with st.expander(f"Details (JSON){zw}{ver}", expanded=False):
                    st.json(_detail_json_records(g1, order))
            st.stop()

        filtered = df[df["prompt_id"].isin(show_prompt_ids)]
        for pid in show_prompt_ids:
            group = filtered[filtered["prompt_id"] == pid]
            if group.empty:
                continue
            st.subheader(f"prompt_id: {pid}")
            st.write(
                "Prompt:",
                group["test_case_prompt"].iloc[0] if "test_case_prompt" in group.columns else "",
            )
            g1 = (
                group.drop_duplicates(subset=["model"], keep="last")
                if "model" in group.columns
                else group
            )
            if "model" in g1.columns:
                g1_disp = g1.set_index("model").reindex(order)
                g1_disp.index = [get_model_label(m) for m in g1_disp.index]
            else:
                g1_disp = g1
            show_cols = [c for c in ("response", "processing_time", "token_count") if c in g1_disp.columns]
            if show_cols:
                st.dataframe(g1_disp[show_cols])
            zw = "\u200b"
            ver = st.session_state.get("_ui_state_version", 0)
            with st.expander(f"Details (JSON){zw}{ver}", expanded=False):
                st.json(_detail_json_records(g1, order))
        st.write(f"Page {int(page_num)}/{max_page} ({total_prompts} total)")
        st.stop()
    _DETAIL_PROMPT_KEY_COL = "_detail_prompt_key"

    def _detail_prompt_key(row: pd.Series) -> str:
        """
        Build the row-grouping key used by Data details.

        Why this is not just ``prompt_id``:
        the demo dashboard usually displays a merge of ``base_v2`` and the latest
        ``test_v2`` run. In batched demo runs, several benchmarks can be packed
        into one temporary prompt file. Older split outputs preserved the global
        prompt index from that temporary batch file, so a later benchmark could
        have local rows numbered 5..9, 10..14, etc. while the corresponding
        baseline rows for that benchmark were still numbered 0..4.

        If we group Data details only by ``prompt_id``, the same prompt from
        base/test is displayed as two separate groups. The model rows missing
        from each half then appear as ``None``, even though both judge records
        exist and the aggregate graph/stat can be correct.

        Use prompt content and benchmark-specific fields as the stable identity
        for display. Keep ``prompt_id`` only as a fallback for records that do
        not carry enough prompt metadata. This also makes already-written output
        files readable without requiring a regeneration step.
        """
        parts = []
        for col in (
            "test_case_prompt",
            "user_input",
            "judge_question",
            "attack_type",
            "task",
            "method",
        ):
            if col in row.index and pd.notna(row[col]):
                parts.append(f"{col}={row[col]}")
        if parts:
            return "\n".join(parts)
        return f"prompt_id={row.get('prompt_id', '')}"

    def _format_prompt_ids(group: pd.DataFrame) -> str:
        # A merged Data details group may contain both the baseline local id
        # and an older batched global id. Showing all observed ids makes that
        # provenance visible without using those ids as the grouping identity.
        if "prompt_id" not in group.columns:
            return ""
        ids = []
        for v in group["prompt_id"].dropna().tolist():
            if v not in ids:
                ids.append(v)
        if not ids:
            return ""
        return str(ids[0]) if len(ids) == 1 else ", ".join(str(v) for v in ids)

    def _one_row_per_model_for_prompt(g: pd.DataFrame) -> pd.DataFrame:
        """Merged base+test judge lists can duplicate the same model per prompt_id; reindex requires a unique index."""
        if g.empty or "model" not in g.columns:
            return g
        return g.drop_duplicates(subset=["model"], keep="last")

    st.title(f"{display_label}")
    df = pd.DataFrame(judge)
    if df.empty:
        st.info("No judge data yet. Run an evaluation first.")
        st.stop()
    # Add an internal display-only grouping key. It is dropped before rendering
    # the JSON expander so users still see the original benchmark/judge records.
    df[_DETAIL_PROMPT_KEY_COL] = df.apply(_detail_prompt_key, axis=1)

    # reset search/expanders when page or task changes
    state_key = f"state_{task['name']}_{page}"
    if "_ui_state_version" not in st.session_state:
        st.session_state["_ui_state_version"] = 0

    def _bump_ui_version():
        st.session_state["_ui_state_version"] += 1

    # Items per page and pagination
    per_page = st.sidebar.selectbox("Items per page", [1, 5, 10, 20], index=1)

    # --- Variant selector (default ALL) ---
    variant_options = ["ALL"]
    variant_sel = "ALL"
    # For AdvGLUE we provide separate task/method selectors in the sidebar
    is_advglue = task["name"] == "AdvGLUE"
    if is_advglue:
        # tasks list in first-seen order
        if "task" in df.columns:
            seen_tasks = []
            for _, r in df.dropna(subset=["task"]).iterrows():
                v = r["task"]
                if v not in seen_tasks:
                    seen_tasks.append(v)
        else:
            seen_tasks = []
        adv_task_sel = st.sidebar.selectbox("AdvGLUE task", ["ALL"] + seen_tasks)
        # methods depend on selected task
        if "method" in df.columns:
            if adv_task_sel != "ALL":
                seen_methods = (
                    df[df["task"] == adv_task_sel]["method"].dropna().astype(str).tolist()
                )
            else:
                seen_methods = df["method"].dropna().astype(str).unique().tolist()
        else:
            seen_methods = []
        adv_method_sel = st.sidebar.selectbox("AdvGLUE method", ["ALL"] + seen_methods)
    else:
        if task_key == "prompt-injection":
            if "injection_variant" in df.columns:
                # preserve first-seen order by prompt_id
                seen = []
                for _, r in df.dropna(subset=["injection_variant"]).iterrows():
                    v = r["injection_variant"]
                    if v not in seen:
                        seen.append(v)
                variants = seen
            else:
                variants = []
        elif task_key == "jailbreak":
            if "attack_type" in df.columns:
                seen = []
                for _, r in df.dropna(subset=["attack_type"]).iterrows():
                    v = r["attack_type"]
                    if v not in seen:
                        seen.append(v)
                variants = seen
            else:
                variants = []
        else:
            # other tasks - fall back to collecting task/method combos or tasks
            if "task" in df.columns and "method" in df.columns:
                seen = []
                for _, r in df.dropna(subset=["task", "method"]).iterrows():
                    v = f"{r['task']}:{r['method']}"
                    if v not in seen:
                        seen.append(v)
                variants = seen
            elif "task" in df.columns:
                seen = []
                for _, r in df.dropna(subset=["task"]).iterrows():
                    v = r["task"]
                    if v not in seen:
                        seen.append(v)
                variants = seen
            else:
                variants = []
        variant_options += variants
        variant_sel = st.sidebar.selectbox("Attack variant", variant_options)

    # --- Search box (search across all JSON fields). If provided, show matching details JSON ---
    # Reset search when task/page changes to avoid stale queries
    if (
        "_last_task_page" not in st.session_state
        or st.session_state["_last_task_page"] != state_key
    ):
        st.session_state["_last_task_page"] = state_key
        st.session_state["_last_search_query"] = ""
        # bump UI version to close expanders
        _bump_ui_version()

    search_query = st.sidebar.text_input(
        "Search (any field)", value=st.session_state.get("_last_search_query", "")
    )
    # store it back
    st.session_state["_last_search_query"] = search_query

    # Apply variant filter to df before pagination
    if variant_sel != "ALL":
        if task_key == "prompt-injection" and "injection_variant" in df.columns:
            df = df[
                df["injection_variant"]
                .astype(str)
                .str.contains(str(variant_sel), case=False, na=False)
            ]
        elif task_key == "jailbreak" and "attack_type" in df.columns:
            # use normalization/loose matching so keys like 'fixed sentence' match 'fixed sentence'
            v_norm = _normalize(variant_sel)
            df = df[df["attack_type"].astype(str).apply(lambda x: v_norm in _normalize(x))]
        else:  # AdvGLUE combined or task-only
            if ":" in variant_sel and "task" in df.columns and "method" in df.columns:
                tpart, mpart = variant_sel.split(":", 1)
                # match exactly on task and method but be resilient to case/whitespace
                df = df[
                    (df["task"].astype(str).str.strip().str.lower() == tpart.strip().lower())
                    & (df["method"].astype(str).str.strip().str.lower() == mpart.strip().lower())
                ]
            elif "task" in df.columns:
                df = df[df["task"].astype(str).str.contains(str(variant_sel), case=False, na=False)]

    unique_prompt_keys = df[_DETAIL_PROMPT_KEY_COL].drop_duplicates().tolist()

    # --- Show selected variant meaning and unique-count ---
    selected_desc = ""
    # unique prompts matching (not duplicated per model)
    selected_count = df["prompt_id"].nunique()
    if variant_sel != "ALL":
        if task_key == "prompt-injection":
            selected_desc = _find_description(PROMPT_INJECTION_VARIANT_DESC, variant_sel)
            selected_count = df["prompt_id"].nunique()
        elif task_key == "jailbreak":
            # match using normalization against JSON attack_type values
            selected_desc = _find_description(JAILBREAK_VARIANT_DESC, variant_sel)
            selected_count = df["prompt_id"].nunique()
        else:
            # AdvGLUE
            if ":" in variant_sel:
                tpart, mpart = variant_sel.split(":", 1)
                selected_desc = f"Task: {ADVGLUE_TASK_DESC.get(tpart, tpart)}; Method: {ADVGLUE_METHOD_DESC.get(mpart, mpart)}"
            else:
                selected_desc = ADVGLUE_TASK_DESC.get(variant_sel, "")
            selected_count = df["prompt_id"].nunique()

    # Display at the top
    if variant_sel == "ALL":
        st.markdown("**Selected variant:** ALL (no filtering)")
        st.markdown(f"**Matching unique prompts:** {selected_count}")
    else:
        st.markdown(f"**Selected variant:** {variant_sel}")
        if selected_desc:
            st.markdown(f"**Meaning:** {selected_desc}")
        st.markdown(f"**Matching unique prompts:** {selected_count}")
    total_prompts = len(unique_prompt_keys)
    max_page = max(1, (total_prompts + per_page - 1) // per_page)
    page_num = st.sidebar.number_input(
        "Page number", min_value=1, max_value=max_page, value=1, step=1
    )

    # 表示するprompt_id範囲を決定
    start_idx = (page_num - 1) * per_page
    end_idx = start_idx + per_page
    show_prompt_keys = unique_prompt_keys[start_idx:end_idx]

    filtered = df[df[_DETAIL_PROMPT_KEY_COL].isin(show_prompt_keys)]

    # If search query provided, search across ALL JSON fields (columns) and show unique prompt matches
    if search_query:
        mask = pd.Series(False, index=df.index)
        for c in df.columns:
            mask = mask | df[c].astype(str).str.contains(search_query, case=False, na=False)
        matches = df[mask]
        unique_matches = (
            matches.drop_duplicates(subset=[_DETAIL_PROMPT_KEY_COL])
            if _DETAIL_PROMPT_KEY_COL in matches.columns
            else matches
        )
        st.subheader(
            f"Search results ({len(unique_matches)} unique prompt matches, {len(matches)} total rows)"
            )
        if not unique_matches.empty:
            # render matches grouped by prompt_id just like normal display
            for pkey in unique_matches[_DETAIL_PROMPT_KEY_COL].tolist():
                group = matches[matches[_DETAIL_PROMPT_KEY_COL] == pkey]
                if group.empty:
                    continue
                pid_label = _format_prompt_ids(group)
                st.subheader(f"prompt_id: {pid_label}" if pid_label else "prompt")
                st.write(
                    "Test case prompt:",
                    group["test_case_prompt"].iloc[0] if "test_case_prompt" in group else "",
                )
                if task_key == "prompt-injection":
                    st.write(
                        "User input:", group["user_input"].iloc[0] if "user_input" in group else ""
                    )
                    st.write(
                        "Judge question:",
                        group["judge_question"].iloc[0] if "judge_question" in group else "",
                    )

                show_cols = ["response", "judge_response"]
                if "judge_label" in group.columns:
                    show_cols.append("judge_label")

                g1 = _one_row_per_model_for_prompt(group)
                group_sorted = g1.set_index("model").reindex(order)
                group_sorted.index = [get_model_label(m) for m in group_sorted.index]
                st.dataframe(group_sorted[show_cols])
                # append an invisible zero-width suffix with UI version so the
                # widget identity changes when we bump UI version, but nothing
                # visible shows to the user.
                zw = "\u200b"
                ver = st.session_state.get("_ui_state_version", 0)
                with st.expander(f"Details (JSON){zw}{ver}", expanded=False):
                    st.json(
                        _detail_json_records(
                            g1,
                            order,
                            drop_cols=(_DETAIL_PROMPT_KEY_COL,),
                        )
                    )
        else:
            st.write("No matches found for the query.")

    # provide a button in the sidebar to close all expanders (always visible)
    if st.sidebar.button("Close all expanders"):
        _bump_ui_version()

    if not filtered.empty:
        for pkey in show_prompt_keys:
            group = filtered[filtered[_DETAIL_PROMPT_KEY_COL] == pkey]
            if group.empty:
                continue
            pid_label = _format_prompt_ids(group)
            st.subheader(f"prompt_id: {pid_label}" if pid_label else "prompt")
            st.write(
                "Test case prompt:",
                group["test_case_prompt"].iloc[0] if "test_case_prompt" in group else "",
            )
            if task_key == "prompt-injection":
                st.write(
                    "User input:", group["user_input"].iloc[0] if "user_input" in group else ""
                )
                st.write(
                    "Judge question:",
                    group["judge_question"].iloc[0] if "judge_question" in group else "",
                )

            # Prepare columns to show (exclude numeric index). Use model labels as index.
            show_cols = ["response", "judge_response"]
            if "judge_label" in group.columns:
                show_cols.append("judge_label")

            # Reindex by MODEL_ORDER (which are model ids), then set index to simplified labels
            g1 = _one_row_per_model_for_prompt(group)
            group_sorted = g1.set_index("model").reindex(order)
            # Replace index values with simplified labels
            group_sorted.index = [get_model_label(m) for m in group_sorted.index]
            # Display with model labels as the index (no extra numeric index)
            st.dataframe(group_sorted[show_cols])
            # Use UI version as key to ensure expanders close when version changes
            # Use invisible zero-width suffix with UI version so expanders
            # are re-created when version changes. Do NOT display the version.
            zw = "\u200b"
            ver = st.session_state.get("_ui_state_version", 0)
            with st.expander(f"Details (JSON){zw}{ver}", expanded=False):
                st.json(
                    _detail_json_records(
                        g1,
                        order,
                        drop_cols=(_DETAIL_PROMPT_KEY_COL,),
                    )
                )
        st.write(f"Page {page_num}/{max_page} ({total_prompts} total)")
    else:
        st.info("No matching data.")
