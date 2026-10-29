"""
Demo UI: batch generate / judge across multiple benchmarks in fewer subprocesses.

When >=2 benchmarks are selected (and CREATE_EVAL_DEMO_BATCH is not disabled):
- **Generate**: group by run_benchmark (LLM_generate_only vs prompt-injection) and,
  within LLM_generate_only, by use_system_prompt. Each group loads each under-test
  model once for all merged prompts in that group.
- **Judge**: one subprocess for all selected benchmarks; task-specific judge templates
  are chosen per record via meta.task (no global --task override).

Assumptions / limits (demo 15 benches):
- Merged prompts carry demo_bench_id in meta (and top-level) so responses split correctly
  (query_llm copies the whole meta object into each response).
- Prompt-injection must not share a generate subprocess with LLM_generate_only.
- Per-bench postprocess (label verify) still runs separately after the batched judge.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

# Steps returned to evaluate_ui.run_pipeline
STEP_SUBPROCESS = "subprocess"
STEP_SPLIT_RESPONSES = "split_responses"
STEP_MERGE_RESPONSES_FOR_JUDGE = "merge_responses_for_judge"
STEP_SPLIT_JUDGE = "split_judge"


def should_batch(
    selected_bench_ids: Sequence[str],
    *,
    ui_enabled: bool = True,
) -> bool:
    """Batch when UI checkbox is on, env does not force off, and at least two benchmarks."""
    if not ui_enabled:
        return False
    if os.environ.get("CREATE_EVAL_DEMO_BATCH", "").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    return len(selected_bench_ids) >= 2


def _batch_work_dir(output_base: str) -> str:
    d = os.path.join(output_base, ".demo_batch")
    os.makedirs(d, exist_ok=True)
    return d


def _run_llm_parallel() -> int:
    raw = os.environ.get("CREATE_EVAL_DEMO_RUN_LLM_IN_PARALLEL", "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _generation_group_key(bench_by_id: Dict[str, dict], bench_id: str) -> str:
    """Separate prompt-injection from LLM_generate_only; split latter by use_system_prompt."""
    b = bench_by_id[bench_id]
    rb = b["run_benchmark"]
    if rb == "prompt-injection":
        return "prompt-injection"
    sys_on = bool(b.get("use_system_prompt"))
    return f"LLM_generate_only_sys{1 if sys_on else 0}"


def _cli_benchmark_for_generation_group(group_key: str) -> str:
    if group_key == "prompt-injection":
        return "prompt-injection"
    return "LLM_generate_only"


def _use_system_prompt_for_generation_group(
    group_key: str,
    bench_ids: Sequence[str],
    bench_by_id: Dict[str, dict],
) -> bool:
    if group_key == "prompt-injection":
        return all(bench_by_id[bid].get("use_system_prompt") for bid in bench_ids)
    return group_key.endswith("_sys1")


def group_benches_for_generation(
    selected_bench_ids: Sequence[str],
    bench_by_id: Dict[str, dict],
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for bid in selected_bench_ids:
        key = _generation_group_key(bench_by_id, bid)
        groups.setdefault(key, []).append(bid)
    return groups


def _record_bench_id(record: dict) -> Optional[str]:
    """Bench id for split/merge; survives generate via meta copy in query_llm."""
    bid = record.get("demo_bench_id")
    if isinstance(bid, str) and bid:
        return bid
    meta = record.get("meta")
    if isinstance(meta, dict):
        mid = meta.get("demo_bench_id")
        if isinstance(mid, str) and mid:
            return mid
    return None


def _annotate_record_for_batch(record: dict, bench_id: str, judge_task: str) -> dict:
    row = dict(record)
    row["demo_bench_id"] = bench_id
    meta = dict(row.get("meta") or {})
    meta["task"] = judge_task
    meta["demo_bench_id"] = bench_id
    row["meta"] = meta
    return row


def build_global_prompt_id_map(
    bench_ids: Sequence[str],
    bench_by_id: Dict[str, dict],
    demo_datasets_dir: str,
) -> Dict[tuple[str, int], int]:
    """
    Assign one prompt sequence across the whole selected demo run.

    Generation still has to be split into groups such as LLM_generate_only and
    prompt-injection, but users inspect one logical run. Keeping this id map at
    the selected-run level prevents each generation group from restarting
    demo_global_prompt_id at zero.
    """
    out: Dict[tuple[str, int], int] = {}
    global_id = 0
    for bid in bench_ids:
        b = bench_by_id[bid]
        prompt_fp = os.path.join(demo_datasets_dir, b["prompt_file"])
        with open(prompt_fp, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"Expected JSON array in {prompt_fp}")
        for local_id, _item in enumerate(items):
            out[(bid, local_id)] = global_id
            global_id += 1
    return out


def write_merged_prompts(
    bench_ids: Sequence[str],
    bench_by_id: Dict[str, dict],
    demo_datasets_dir: str,
    out_path: str,
    global_prompt_ids: Optional[Dict[tuple[str, int], int]] = None,
) -> int:
    merged: List[dict] = []
    for bid in bench_ids:
        b = bench_by_id[bid]
        prompt_fp = os.path.join(demo_datasets_dir, b["prompt_file"])
        with open(prompt_fp, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"Expected JSON array in {prompt_fp}")
        for local_id, item in enumerate(items):
            row = _annotate_record_for_batch(dict(item), bid, b["judge_task"])
            row["prompt_id"] = local_id
            if global_prompt_ids is not None:
                global_id = global_prompt_ids[(bid, local_id)]
            else:
                global_id = len(merged)
            row["demo_global_prompt_id"] = global_id
            meta = dict(row.get("meta") or {})
            meta["demo_global_prompt_id"] = global_id
            row["meta"] = meta
            merged.append(row)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return len(merged)


def split_combined_records(
    merged_path: str,
    bench_ids: Sequence[str],
    bench_output_paths_fn: Callable[[str, str], dict[str, str]],
    output_base: str,
    field: str,
) -> tuple[Dict[str, int], int]:
    """
    Write per-bench JSON arrays filtered by demo_bench_id.
    Returns (counts per bench, number of records with unknown bench id).
    """
    with open(merged_path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected JSON array in {merged_path}")
    buckets: Dict[str, list] = {bid: [] for bid in bench_ids}
    orphan = 0
    for rec in records:
        bid = _record_bench_id(rec)
        if bid in buckets:
            buckets[bid].append(rec)
        else:
            orphan += 1
    counts: Dict[str, int] = {}
    for bid, rows in buckets.items():
        # Keep per-benchmark output compatible with non-batched runs.
        #
        # During a batched demo run, multiple benchmark prompt files are merged
        # into one temporary file before generation/judging. The runner then
        # emits ``prompt_id`` values from that temporary merged file. For the
        # first benchmark those ids are usually 0..4, but later benchmarks can
        # be 5..9, 10..14, and so on.
        #
        # The dashboard's base/test merge expects each per-benchmark file to use
        # local prompt ids, matching the standalone/non-batched layout and the
        # baseline files under ``base_v2``. If we preserve the global batch ids
        # here, Data details cannot align the same prompt across models and will
        # show empty/None cells for models that actually have judge records.
        #
        # Normalize ids back to local per-benchmark ids at split time, while
        # preserving demo_global_prompt_id as the selected-run-level sequence
        # number. Some subprocesses keep the helper id at the top level, and
        # others only preserve it via meta; support both paths.
        local_prompt_ids: Dict[object, int] = {}
        for row in rows:
            old_prompt_id = row.get("prompt_id")
            if old_prompt_id not in local_prompt_ids:
                local_prompt_ids[old_prompt_id] = len(local_prompt_ids)
            meta = row.get("meta")
            meta_global_id = meta.get("demo_global_prompt_id") if isinstance(meta, dict) else None
            global_prompt_id = row.get("demo_global_prompt_id", meta_global_id)
            if global_prompt_id is None:
                global_prompt_id = old_prompt_id
            row["demo_global_prompt_id"] = global_prompt_id
            row["prompt_id"] = local_prompt_ids[old_prompt_id]
        paths = bench_output_paths_fn(output_base, bid)
        os.makedirs(paths["dir"], exist_ok=True)
        out_key = "responses" if field == "responses" else "judge"
        with open(paths[out_key], "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        counts[bid] = len(rows)
    return counts, orphan


def merge_per_bench_response_files(
    bench_ids: Sequence[str],
    bench_by_id: Dict[str, dict],
    bench_output_paths_fn: Callable[[str, str], dict[str, str]],
    output_base: str,
    out_path: str,
) -> int:
    combined: List[dict] = []
    for bid in bench_ids:
        paths = bench_output_paths_fn(output_base, bid)
        resp_fp = paths["responses"]
        if not os.path.isfile(resp_fp):
            raise FileNotFoundError(f"Missing responses for bench {bid}: {resp_fp}")
        with open(resp_fp, encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Expected JSON array in {resp_fp}")
        judge_task = bench_by_id[bid]["judge_task"]
        for rec in rows:
            row = dict(rec)
            if _record_bench_id(row) is None:
                row = _annotate_record_for_batch(row, bid, judge_task)
            else:
                meta = dict(row.get("meta") or {})
                meta["task"] = judge_task
                meta["demo_bench_id"] = bid
                row["meta"] = meta
                row["demo_bench_id"] = bid
            combined.append(row)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    return len(combined)


def build_batched_pipeline_plan(
    *,
    models: List[str],
    selected_bench_ids: List[str],
    run_mode: str,
    output_base: str,
    demo_datasets_dir: str,
    judge_spec: str,
    python_exec: str,
    bench_by_id: Dict[str, dict],
    bench_output_paths_fn: Callable[[str, str], dict[str, str]],
    spec_for_runner: Callable[[str], str],
    generate_python_exec: str | None = None,
) -> tuple[List[dict], dict]:
    """
    Returns (steps, ctx). ctx holds paths for logging (batch_dir, prompt counts, etc.).
    """
    run_generate = run_mode in ("full", "generate")
    run_judge = run_mode in ("full", "judge")
    for bid in selected_bench_ids:
        os.makedirs(bench_output_paths_fn(output_base, bid)["dir"], exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = _batch_work_dir(output_base)
    ctx: dict = {"batch_dir": batch_dir, "timestamp": ts, "groups": {}}
    steps: List[dict] = []
    par = _run_llm_parallel()
    gen_exec = generate_python_exec or python_exec
    global_prompt_ids = build_global_prompt_id_map(
        selected_bench_ids,
        bench_by_id,
        demo_datasets_dir,
    )

    for group_key, bids in group_benches_for_generation(
        selected_bench_ids, bench_by_id
    ).items():
        cli_bench = _cli_benchmark_for_generation_group(group_key)
        merged_prompt = os.path.join(batch_dir, f"prompts_{group_key}_{ts}.json")
        merged_resp = os.path.join(batch_dir, f"responses_{group_key}_{ts}.json")
        n_prompts = write_merged_prompts(
            bids,
            bench_by_id,
            demo_datasets_dir,
            merged_prompt,
            global_prompt_ids,
        )
        ctx["groups"][group_key] = {
            "bench_ids": bids,
            "n_prompts": n_prompts,
            "merged_prompt": merged_prompt,
            "cli_benchmark": cli_bench,
        }

        if run_generate:
            gen_cmd = [
                gen_exec,
                "-m",
                "CybersecurityBenchmarks.benchmark.run",
                f"--benchmark={cli_bench}",
                f"--prompt-path={merged_prompt}",
                f"--response-path={merged_resp}",
                *[f"--llm-under-test={spec_for_runner(m)}" for m in models],
                "--only-generate-llm-responses",
                f"--run-llm-in-parallel={par}",
            ]
            if _use_system_prompt_for_generation_group(group_key, bids, bench_by_id):
                gen_cmd.append("--use-system-prompt")
            steps.append(
                {
                    "type": STEP_SUBPROCESS,
                    "label": (
                        f"generate batch ({cli_bench}, {len(bids)} benchmarks, "
                        f"{n_prompts} prompts)"
                    ),
                    "cmd": gen_cmd,
                }
            )
            steps.append(
                {
                    "type": STEP_SPLIT_RESPONSES,
                    "label": f"split responses ({group_key})",
                    "merged_path": merged_resp,
                    "bench_ids": list(bids),
                }
            )

    if run_judge:
        merged_for_judge = os.path.join(batch_dir, f"responses_all_{ts}.json")
        merged_judge = os.path.join(batch_dir, f"judge_all_{ts}.json")
        merged_stat = os.path.join(batch_dir, f"stat_all_{ts}.json")
        ctx["merged_for_judge"] = merged_for_judge
        ctx["merged_judge"] = merged_judge

        steps.append(
            {
                "type": STEP_MERGE_RESPONSES_FOR_JUDGE,
                "label": "merge responses for batched judge",
                "bench_ids": list(selected_bench_ids),
                "out_path": merged_for_judge,
            }
        )
        judge_cmd = [
            python_exec,
            "-m",
            "CybersecurityBenchmarks.benchmark.generic_judge_evaluator",
            f"--response-path={merged_for_judge}",
            f"--judge-llm={judge_spec}",
            f"--judge-response-path={merged_judge}",
            f"--stat-path={merged_stat}",
            f"--run-llm-in-parallel={par}",
        ]
        steps.append(
            {
                "type": STEP_SUBPROCESS,
                "label": f"judge batch ({len(selected_bench_ids)} benchmarks)",
                "cmd": judge_cmd,
            }
        )
        steps.append(
            {
                "type": STEP_SPLIT_JUDGE,
                "label": "split judge outputs",
                "merged_path": merged_judge,
                "bench_ids": list(selected_bench_ids),
            }
        )
        for bid in selected_bench_ids:
            paths = bench_output_paths_fn(output_base, bid)
            task = bench_by_id[bid]["judge_task"]
            steps.append(
                {
                    "type": STEP_SUBPROCESS,
                    "label": f"recompute stat ({bid})",
                    "cmd": [
                        python_exec,
                        "-m",
                        "CybersecurityBenchmarks.benchmark.generic_judge_evaluator",
                        "--recompute-stat-only",
                        f"--judge-response-path={paths['judge']}",
                        f"--stat-path={paths['stat']}",
                        f"--task={task}",
                    ],
                }
            )

    return steps, ctx


def build_per_bench_commands(
    *,
    models: List[str],
    selected_bench_ids: List[str],
    run_mode: str,
    output_base: str,
    demo_datasets_dir: str,
    judge_spec: str,
    python_exec: str,
    bench_by_id: Dict[str, dict],
    bench_output_paths_fn: Callable[[str, str], dict[str, str]],
    spec_for_runner: Callable[[str], str],
    generate_python_exec: str | None = None,
) -> List[List[str]]:
    """Original one-subprocess-per-benchmark command list (single benchmark)."""
    cmds: List[List[str]] = []
    run_generate = run_mode in ("full", "generate")
    run_judge = run_mode in ("full", "judge")
    gen_exec = generate_python_exec or python_exec
    for bid in selected_bench_ids:
        b = bench_by_id[bid]
        paths = bench_output_paths_fn(output_base, bid)
        os.makedirs(paths["dir"], exist_ok=True)
        prompt_fp = os.path.join(demo_datasets_dir, b["prompt_file"])
        resp_fp = paths["responses"]
        bench_arg = b["run_benchmark"]
        if run_generate:
            gen_cmd = [
                gen_exec,
                "-m",
                "CybersecurityBenchmarks.benchmark.run",
                f"--benchmark={bench_arg}",
                f"--prompt-path={prompt_fp}",
                f"--response-path={resp_fp}",
                *[f"--llm-under-test={spec_for_runner(m)}" for m in models],
                "--only-generate-llm-responses",
            ]
            if b.get("use_system_prompt"):
                gen_cmd.append("--use-system-prompt")
            cmds.append(gen_cmd)
        if run_judge:
            judge_cmd = [
                python_exec,
                "-m",
                "CybersecurityBenchmarks.benchmark.generic_judge_evaluator",
                f"--response-path={resp_fp}",
                f"--judge-llm={judge_spec}",
                f"--task={b['judge_task']}",
                f"--judge-response-path={paths['judge']}",
                f"--stat-path={paths['stat']}",
            ]
            cmds.append(judge_cmd)
    return cmds
