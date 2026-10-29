"""
four-axis-ai-security-eval demo UI: 15-benchmark metadata and output paths.
Prompts live in datasets/demo/*_demo5.json (5 items each).

Public presets do not include local NICT Qwen checkpoints. Helper functions below
remain so dashboard labels can remap historical baseline model ids if present.

- ``CREATE_EVAL_DEMO_LOCAL_NICT_ROOT`` — root used only when remapping those ids.
- ``CREATE_EVAL_HF_NICT_ROOT`` / ``CREATE_EVAL_HF_PATH_REMAP`` — prefix swap for
  generated HF local-path specs.
"""
from __future__ import annotations

import os
from typing import Any, TypedDict

# Only the two "Local · Qwen3-8B …" demo presets use this tree (not ``~`` / not other presets).
_LOCAL_QWEN_PRESET_ROOT_DEFAULT = "/home/ma/nict-llm-main"


def demo_local_qwen_preset_root() -> str:
    """Root directory for the two local Qwen demo presets only (override via env)."""
    return os.path.normpath(
        os.environ.get("CREATE_EVAL_DEMO_LOCAL_NICT_ROOT", "").strip()
        or _LOCAL_QWEN_PRESET_ROOT_DEFAULT
    )


def demo_local_qwen_hf16_join(*parts: str) -> str:
    """HF16:: absolute path: ``<demo_local_qwen_preset_root()> / parts...`` (use exact case for each segment)."""
    base = os.path.join(demo_local_qwen_preset_root(), *parts)
    return f"HF16::{os.path.normpath(base)}"


def demo_short_label_llm_spec(spec: str) -> str:
    """UI-only: ``.../notebooks/qwen/...`` or ``.../notebooks/Qwen/...`` → ``PROVIDER::<checkpoint>``."""
    spec = spec.strip()
    if "::" not in spec:
        return spec
    prov, rest = spec.split("::", 1)
    path = os.path.normpath(os.path.expanduser(rest.strip()))
    if not os.path.isabs(path):
        return spec
    parent = os.path.dirname(path)
    gp = os.path.dirname(parent)
    if os.path.basename(gp) == "notebooks" and os.path.basename(parent).lower() == "qwen":
        return f"{prov}::{os.path.basename(path)}"
    return spec


def _demo_apply_hf_prefix_swap(spec: str, old_prefix: str, new_prefix: str) -> str:
    if not new_prefix or "::" not in spec:
        return spec
    prov, rest = spec.split("::", 1)
    path = rest.strip()
    if os.path.isabs(path) and path.startswith(old_prefix):
        path = new_prefix.rstrip("/") + path[len(old_prefix) :]
        return f"{prov}::{path}"
    return spec


def demo_remap_hf_local_path_in_spec(spec: str) -> str:
    """Apply ``CREATE_EVAL_HF_NICT_ROOT`` then ``CREATE_EVAL_HF_PATH_REMAP`` to HF local paths."""
    spec = spec.strip()
    if "::" not in spec:
        return spec

    nict_new = os.environ.get("CREATE_EVAL_HF_NICT_ROOT", "").strip()
    if nict_new:
        spec = _demo_apply_hf_prefix_swap(spec, demo_local_qwen_preset_root(), nict_new)

    raw = os.environ.get("CREATE_EVAL_HF_PATH_REMAP", "").strip()
    if raw:
        for segment in (x.strip() for x in raw.split("||")):
            if not segment or "=" not in segment:
                continue
            old_prefix, new_prefix = segment.split("=", 1)
            old_prefix = old_prefix.strip()
            new_prefix = new_prefix.strip()
            if not old_prefix:
                continue
            spec = _demo_apply_hf_prefix_swap(spec, old_prefix, new_prefix)

    return spec


class DemoBenchSpec(TypedDict, total=False):
    id: str
    axis_id: str
    label: str
    demo_n: int
    full_n: int
    metric: str
    source: str
    prompt_file: str
    run_benchmark: str  # "LLM_generate_only" | "prompt-injection"
    judge_task: str
    use_system_prompt: bool


class DemoAxisSpec(TypedDict):
    id: str
    label: str
    standardized_metric: str


# Reference UI: one standardized metric per axis (higher = better).
DEMO_AXES: list[DemoAxisSpec] = [
    {"id": "safety", "label": "Safety", "standardized_metric": "Safety score (↑)"},
    {"id": "cyber_misuse", "label": "Cyber misuse", "standardized_metric": "Safety score (↑)"},
    {
        "id": "excessive_refusal",
        "label": "Excessive refusal",
        "standardized_metric": "Acceptance rate (↑)",
    },
    {"id": "general", "label": "General", "standardized_metric": "Accuracy (↑)"},
]

# Short English blurbs for the benchmark table Axis column (RQ1-6 four-axis design).
DEMO_AXIS_UI_DESCRIPTIONS: dict[str, str] = {
    "safety": (
        "General harmful-use and attack resistance: misuse, jailbreak, adversarial "
        "compliance, and prompt injection. Measures everyday alignment—not "
        "cybersecurity-expert attack scenarios."
    ),
    "cyber_misuse": (
        "Cybersecurity-specific misuse: attack assistance, exploits, malicious "
        "interpreter use, MITRE ATT&CK TTPs, and risky code (RedCode). Uses the same "
        "safety score (↑) as Safety; a separate threat surface."
    ),
    "excessive_refusal": (
        "Over-refusal on benign or legitimate prompts: whether safety tuning makes the "
        "model decline harmless or appropriate requests (FalseReject, FRR, exaggerated "
        "safety). Acceptance rate (↑): higher = more useful responses, less undue refusal."
    ),
    "general": (
        "General capability outside safety trade-offs: broad knowledge (MMLU) and "
        "challenging reasoning (BIG-bench Hard). Accuracy (↑): higher = more correct "
        "answers—used to check that safety changes do not collapse baseline ability."
    ),
}

AXIS_BENCHMARK_IDS: dict[str, list[str]] = {
    "safety": [
        "harmbench",
        "misuse",
        "prompt_injection",
        "advbench",
        "jailbreak",
    ],
    "cyber_misuse": [
        "code_interpreter",
        "cysecbench",
        "mitre",
        "vulnerability_exploitation",
        "redcode",
    ],
    "excessive_refusal": ["falsereject", "frr", "exaggerated_safety"],
    "general": ["mmlu", "bbh"],
}


# Public presets. Hugging Face entries use HF16:: (llm.create HF:: defaults to 4-bit).
DEMO_MODEL_PRESETS: list[tuple[str, str]] = [
    ("Ollama · llama3.2:1b", "OLLAMA::llama3.2:1b"),
    ("Hugging Face · Llama-3.2-1B-Instruct", "HF16::meta-llama/Llama-3.2-1B-Instruct"),
    ("Hugging Face · Foundation-Sec-8B-Instruct", "HF16::fdtn-ai/Foundation-Sec-8B-Instruct"),
    ("Hugging Face · Qwen3-8B", "HF16::Qwen/Qwen3-8B"),
    (
        "Hugging Face · PLaMo 3 NICT 2B Base",
        "HF16::pfnet/plamo-3-nict-2b-base",
    ),
]

DEMO_BENCHMARKS: list[DemoBenchSpec] = [
    {
        "id": "harmbench",
        "axis_id": "safety",
        "label": "HarmBench",
        "demo_n": 5,
        "full_n": 320,
        "metric": "ASR / success vs refusal",
        "source": "Center for AI Safety (2024)",
        "prompt_file": "harmbench_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "harmbench",
        "use_system_prompt": True,
    },
    {
        "id": "misuse",
        "axis_id": "safety",
        "label": "Misuse",
        "demo_n": 5,
        "full_n": 1174,
        "metric": "Misuse (refusal vs accept)",
        "source": "TrustLLM (2024)",
        "prompt_file": "misuse_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "misuse",
        "use_system_prompt": True,
    },
    {
        "id": "prompt_injection",
        "axis_id": "safety",
        "label": "Prompt Injection",
        "demo_n": 5,
        "full_n": 251,
        "metric": "Injection success (yes/no)",
        "source": "Purple Llama CyberSecEval",
        "prompt_file": "prompt_injection_demo5.json",
        "run_benchmark": "prompt-injection",
        "judge_task": "prompt-injection",
        "use_system_prompt": True,
    },
    {
        "id": "advbench",
        "axis_id": "safety",
        "label": "AdvBench",
        "demo_n": 5,
        "full_n": 1094,
        "metric": "Compliance (refusal vs accept)",
        "source": "AdvBench (2023)",
        "prompt_file": "advbench_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "advbench",
        "use_system_prompt": True,
    },
    {
        "id": "jailbreak",
        "axis_id": "safety",
        "label": "Jailbreak",
        "demo_n": 5,
        "full_n": 1400,
        "metric": "Jailbreak (refusal vs accept)",
        "source": "TrustLLM (2024)",
        "prompt_file": "jailbreak_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "jailbreak",
        "use_system_prompt": True,
    },
    {
        "id": "code_interpreter",
        "axis_id": "cyber_misuse",
        "label": "Code Interpreter",
        "demo_n": 5,
        "full_n": 500,
        "metric": "Refusal vs accept",
        "source": "Purple Llama CyberSecEval",
        "prompt_file": "code_interpreter_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "code_interpreter",
        "use_system_prompt": True,
    },
    {
        "id": "cysecbench",
        "axis_id": "cyber_misuse",
        "label": "CySecBench",
        "demo_n": 5,
        "full_n": 500,
        "metric": "Refusal vs accept",
        "source": "CySecBench (2025)",
        "prompt_file": "cysecbench_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "cysecbench",
        "use_system_prompt": True,
    },
    {
        "id": "mitre",
        "axis_id": "cyber_misuse",
        "label": "MITRE ATT&CK",
        "demo_n": 5,
        "full_n": 1000,
        "metric": "Benign vs malicious",
        "source": "Purple Llama / MITRE",
        "prompt_file": "mitre_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "mitre",
        "use_system_prompt": True,
    },
    {
        "id": "vulnerability_exploitation",
        "axis_id": "cyber_misuse",
        "label": "Vulnerability Exploitation",
        "demo_n": 5,
        "full_n": 592,
        "metric": "Refusal vs accept",
        "source": "Purple Llama CyberSecEval",
        "prompt_file": "vulnerability_exploitation_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "vulnerability_exploitation",
        "use_system_prompt": True,
    },
    {
        "id": "redcode",
        "axis_id": "cyber_misuse",
        "label": "RedCode",
        "demo_n": 5,
        "full_n": 1570,
        "metric": "Exec/Gen judge labels",
        "source": "RedCode, NeurIPS'24",
        "prompt_file": "redcode_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "redcode",
        "use_system_prompt": True,
    },
    {
        "id": "falsereject",
        "axis_id": "excessive_refusal",
        "label": "FalseReject",
        "demo_n": 5,
        "full_n": 1187,
        "metric": "Refusal vs accept",
        "source": "Amazon Science (2025)",
        "prompt_file": "falsereject_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "falsereject",
        "use_system_prompt": True,
    },
    {
        "id": "frr",
        "axis_id": "excessive_refusal",
        "label": "FRR",
        "demo_n": 5,
        "full_n": 750,
        "metric": "False refusal",
        "source": "Purple Llama CyberSecEval",
        "prompt_file": "frr_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "frr",
        "use_system_prompt": True,
    },
    {
        "id": "exaggerated_safety",
        "axis_id": "excessive_refusal",
        "label": "Exaggerated Safety",
        "demo_n": 5,
        "full_n": 200,
        "metric": "Refusal vs accept",
        "source": "TrustLLM / XSTest",
        "prompt_file": "exaggerated_safety_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "exaggerated_safety",
        "use_system_prompt": True,
    },
    {
        "id": "mmlu",
        "axis_id": "general",
        "label": "MMLU",
        "demo_n": 5,
        "full_n": 14042,
        "metric": "Accuracy (correct/incorrect)",
        "source": "Hendrycks et al. (2021)",
        "prompt_file": "mmlu_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "mmlu",
        "use_system_prompt": True,
    },
    {
        "id": "bbh",
        "axis_id": "general",
        "label": "BIG-bench Hard",
        "demo_n": 5,
        "full_n": 6511,
        "metric": "Accuracy (correct/incorrect)",
        "source": "Suzgun et al. (2022)",
        "prompt_file": "bbh_demo5.json",
        "run_benchmark": "LLM_generate_only",
        "judge_task": "bbh",
        "use_system_prompt": True,
    },
]

BENCH_BY_ID: dict[str, DemoBenchSpec] = {b["id"]: b for b in DEMO_BENCHMARKS}


def benchmark_source_label(b: DemoBenchSpec) -> str:
    """Single UI string: Benchmark name (source citation)."""
    return f"{b['label']} ({b['source']})"


def ordered_demo_benchmarks() -> list[DemoBenchSpec]:
    """Axis order then benchmark order (for UI and stable iteration)."""
    out: list[DemoBenchSpec] = []
    for ax in DEMO_AXES:
        for bid in AXIS_BENCHMARK_IDS[ax["id"]]:
            out.append(BENCH_BY_ID[bid])
    return out

# dashboard_eval_ui.TASKS 用（stat/judge 相対 OUTPUT_BASE）
BENCH15_SUBDIR = "bench15"


def bench_output_paths(output_base: str, bench_id: str) -> dict[str, str]:
    root = f"{BENCH15_SUBDIR}/{bench_id}"
    return {
        "dir": f"{output_base}/{root}",
        "responses": f"{output_base}/{root}/{bench_id}_responses.json",
        "judge": f"{output_base}/{root}/{bench_id}_judge.json",
        "stat": f"{output_base}/{root}/{bench_id}_stat.json",
    }


def dashboard_tasks_from_demo() -> list[dict[str, Any]]:
    """dashboard_eval_ui.TASKS に渡す行一覧。"""
    tasks = []
    for b in ordered_demo_benchmarks():
        bid = b["id"]
        tasks.append(
            {
                "name": b["label"],
                "stat": f"{BENCH15_SUBDIR}/{bid}/{bid}_stat.json",
                "judge": f"{BENCH15_SUBDIR}/{bid}/{bid}_judge.json",
                "task_key": b["judge_task"],
                "bench_id": bid,
            }
        )
    return tasks
