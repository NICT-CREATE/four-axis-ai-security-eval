"""
Demo UI: ad-hoc user-input prompts (generate only, no judge, not on Dashboard).
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, List, Optional, Sequence

USER_PROMPT_SUBDIR = "user_prompts"
USER_PROMPT_MAX_SLOTS = 5
RESPONSES_FILENAME = "user_prompts_responses.json"
SNAPSHOT_FILENAME = "prompts_snapshot.json"
RUN_PROMPT_FILENAME = "prompts_for_run.json"

USER_PROMPT_TASK = {
    "name": "User-input prompts",
    "stat": "",
    "judge": "",
    "responses": f"{USER_PROMPT_SUBDIR}/{RESPONSES_FILENAME}",
    "task_key": "user_prompts",
    "bench_id": "",
    "is_user_prompts": True,
}


def user_prompt_output_paths(output_base: str) -> dict[str, str]:
    root = os.path.join(output_base, USER_PROMPT_SUBDIR)
    return {
        "dir": root,
        "responses": os.path.join(root, RESPONSES_FILENAME),
        "snapshot": os.path.join(root, SNAPSHOT_FILENAME),
        "run_prompt": os.path.join(root, RUN_PROMPT_FILENAME),
    }


def prompts_configured(prompt_texts: Sequence[str]) -> bool:
    return any((t or "").strip() for t in prompt_texts)


def validate_mutual_exclusive(
    prompt_texts: Sequence[str],
    benchmark_selected: bool,
) -> Optional[str]:
    has_user = prompts_configured(prompt_texts)
    if has_user and benchmark_selected:
        return (
            "Choose either user-input prompts or benchmarks, not both. "
            "Clear prompt fields or uncheck all benchmarks."
        )
    if not has_user and not benchmark_selected:
        return "Enter at least one user prompt or select at least one benchmark."
    return None


def validate_user_prompt_fields_for_run(prompt_texts: Sequence[str]) -> Optional[str]:
    if not prompts_configured(prompt_texts):
        return "No user-input prompts to run."
    for i, text in enumerate(prompt_texts):
        if not (text or "").strip():
            return (
                f"Prompt {i + 1} is empty. Fill every prompt field or remove extra rows "
                f"(max {USER_PROMPT_MAX_SLOTS})."
            )
    return None


def build_run_dataset(prompt_texts: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for i, text in enumerate(prompt_texts):
        stripped = (text or "").strip()
        if not stripped:
            continue
        rows.append({"test_case_prompt": stripped, "prompt_id": i})
    return rows


def write_run_files(
    output_base: str,
    prompt_texts: Sequence[str],
) -> dict[str, str]:
    paths = user_prompt_output_paths(output_base)
    os.makedirs(paths["dir"], exist_ok=True)
    dataset = build_run_dataset(prompt_texts)
    with open(paths["run_prompt"], "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    with open(paths["snapshot"], "w", encoding="utf-8") as f:
        json.dump(
            [{"prompt_id": r["prompt_id"], "test_case_prompt": r["test_case_prompt"]} for r in dataset],
            f,
            ensure_ascii=False,
            indent=2,
        )
    return paths


def clear_user_prompt_outputs(output_base: str) -> None:
    paths = user_prompt_output_paths(output_base)
    if os.path.isdir(paths["dir"]):
        for name in os.listdir(paths["dir"]):
            fp = os.path.join(paths["dir"], name)
            if os.path.isfile(fp):
                os.remove(fp)


def build_generate_command(
    *,
    python_exec: str,
    output_base: str,
    models: List[str],
    spec_for_runner: Callable[[str], str],
    run_llm_in_parallel: int = 2,
) -> List[str]:
    paths = user_prompt_output_paths(output_base)
    return [
        python_exec,
        "-m",
        "CybersecurityBenchmarks.benchmark.run",
        "--benchmark=LLM_generate_only",
        f"--prompt-path={paths['run_prompt']}",
        f"--response-path={paths['responses']}",
        *[f"--llm-under-test={spec_for_runner(m)}" for m in models],
        "--only-generate-llm-responses",
        "--use-system-prompt",
        f"--run-llm-in-parallel={run_llm_in_parallel}",
    ]


def run_llm_parallel() -> int:
    raw = os.environ.get("CREATE_EVAL_DEMO_RUN_LLM_IN_PARALLEL", "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2
