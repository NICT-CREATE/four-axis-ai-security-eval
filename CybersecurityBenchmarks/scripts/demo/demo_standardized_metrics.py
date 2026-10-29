"""
Standardized metrics (higher = better) for the demo dashboard.

Aligned with oss-llm-safety-eval RQ2-0 Phase 1:
  scripts/RQ2/RQ2-0/RQ2-0_phase1_summarize_safety_degradation.py
  (_get_metric_from_bench, _display_value).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

__all__ = [
    "bench_id_to_axis_id",
    "compute_standardized_snapshot",
    "get_raw_metric",
    "standardized_score",
]


def bench_id_to_axis_id(axis_benchmark_ids: Dict[str, List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ax_id, bids in axis_benchmark_ids.items():
        for bid in bids:
            out[bid] = ax_id
    return out


def _is_model_stat_entry(key: str, val: Any) -> bool:
    if not isinstance(val, dict):
        return False
    key_l = str(key).lower()
    if key_l.startswith("stat") or "per_model" in key_l or "stat_" in key_l:
        return False
    return ("tasks" in val) or ("overall" in val)


def iter_model_ids_from_stat(stat: Dict[str, Any]) -> List[str]:
    if not isinstance(stat, dict):
        return []
    return [k for k, v in stat.items() if _is_model_stat_entry(k, v)]


def resolve_model_node(stat: Dict[str, Any], model_id: str) -> Optional[Dict[str, Any]]:
    if not stat or not isinstance(stat, dict):
        return None
    if model_id in stat:
        v = stat[model_id]
        if isinstance(v, dict) and _is_model_stat_entry(model_id, v):
            return v
    if "::" in model_id:
        stripped = model_id.split("::", 1)[1]
        v = stat.get(stripped)
        if isinstance(v, dict) and _is_model_stat_entry(stripped, v):
            return v
    return None


def get_raw_metric(bench_id: str, node: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    """
    One raw metric per benchmark from a CREATE-eval merged stat node.
    Same cases as RQ2-0 _get_metric_from_bench (bench names use demo ids, e.g. prompt_injection).
    """
    if not isinstance(node, dict):
        return None

    if bench_id == "prompt_injection":
        tasks = node.get("tasks") or {}
        task = tasks.get("prompt-injection") or tasks.get("prompt_injection")
        if isinstance(task, dict) and "yes_rate" in task:
            return (float(task["yes_rate"]), "yes_rate")
        return None

    if bench_id == "redcode":
        overall = node.get("overall") or {}
        # Exec あり: RQ2 Phase 1 と同じく refusal_rate（redcode_exec_overall 優先）
        exec_o = overall.get("redcode_exec_overall")
        if isinstance(exec_o, dict) and "refusal_rate" in exec_o:
            return (float(exec_o["refusal_rate"]), "refusal_rate")
        # Gen のみのデモ: スコア 0 = 拒否・安全寄り → zero_rate を「高いほど良い」生指標に
        gen_o = overall.get("redcode_gen_overall")
        if isinstance(gen_o, dict) and "zero_rate" in gen_o:
            return (float(gen_o["zero_rate"]), "zero_rate")
        ro = overall.get("redcode_overall")
        if isinstance(ro, dict) and "refusal_rate" in ro:
            return (float(ro["refusal_rate"]), "refusal_rate")
        return None

    if bench_id == "mmlu":
        overall = node.get("overall") or {}
        mo = overall.get("mmlu_overall")
        if isinstance(mo, dict) and "accuracy" in mo:
            return (float(mo["accuracy"]), "accuracy")
        return None

    if bench_id == "bbh":
        overall = node.get("overall") or {}
        bo = overall.get("bbh_overall")
        if isinstance(bo, dict) and "accuracy" in bo:
            return (float(bo["accuracy"]), "accuracy")
        return None

    if bench_id == "mitre":
        tasks = node.get("tasks") or {}
        tm = tasks.get("mitre")
        if isinstance(tm, dict):
            br = tm.get("benign_rate")
            if br is not None:
                return (float(br), "benign_rate")
            mr = tm.get("malicious_rate")
            if mr is not None:
                return (1.0 - float(mr), "benign_rate")
        return None

    overall = node.get("overall") or {}
    for _k, val in overall.items():
        if isinstance(val, dict) and "refusal_rate" in val:
            return (float(val["refusal_rate"]), "refusal_rate")

    tasks = node.get("tasks") or {}
    for alt in (bench_id, bench_id.replace("_", "-")):
        tk = tasks.get(alt)
        if isinstance(tk, dict) and "refusal_rate" in tk:
            return (float(tk["refusal_rate"]), "refusal_rate")
    return None


def standardized_score(bench_id: str, metric_name: str, raw: float, axis_id: str) -> float:
    """Unified display score: higher = better (RQ2-0 _display_value)."""
    if bench_id == "prompt_injection" and metric_name == "yes_rate":
        return 1.0 - raw
    if axis_id == "excessive_refusal":
        return 1.0 - raw
    return float(raw)


def load_merged_stat(
    rel_stat_path: str,
    test_root: str,
    base_root: str,
    load_json: Callable[[str], Any],
    merge_stat_dicts: Callable[[dict, dict], dict],
) -> Dict[str, Any]:
    stat_base: Dict[str, Any] = {}
    stat_test: Dict[str, Any] = {}
    p_b = os.path.join(base_root, rel_stat_path)
    p_t = os.path.join(test_root, rel_stat_path)
    if os.path.exists(p_b):
        try:
            stat_base = load_json(p_b) or {}
        except Exception:
            stat_base = {}
    if os.path.exists(p_t):
        try:
            stat_test = load_json(p_t) or {}
        except Exception:
            stat_test = {}
    return merge_stat_dicts(stat_base, stat_test)


def compute_standardized_snapshot(
    tasks: List[Dict[str, Any]],
    test_root: str,
    base_root: str,
    load_json: Callable[[str], Any],
    merge_stat_dicts: Callable[[dict, dict], dict],
    axis_benchmark_ids: Dict[str, List[str]],
    model_order: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[str]]:
    """
    Returns:
        by_axis: model_id -> axis_id -> mean standardized score over benchmarks with data
        by_bench: model_id -> bench_id -> standardized score
        ordered_models: model ids for display
    """
    b2a = bench_id_to_axis_id(axis_benchmark_ids)
    by_bench: Dict[str, Dict[str, float]] = {}
    all_models: Set[str] = set()

    for t in tasks:
        bid = t.get("bench_id")
        if not bid or bid not in b2a:
            continue
        axis_id = b2a[bid]
        merged = load_merged_stat(
            t["stat"], test_root, base_root, load_json, merge_stat_dicts
        )
        for m in iter_model_ids_from_stat(merged):
            all_models.add(m)
            node = resolve_model_node(merged, m)
            if node is None:
                continue
            raw_out = get_raw_metric(bid, node)
            if raw_out is None:
                continue
            raw, metric_name = raw_out
            score = standardized_score(bid, metric_name, raw, axis_id)
            by_bench.setdefault(m, {})[bid] = score

    ordered_models = [m for m in model_order if m in all_models] + sorted(
        m for m in all_models if m not in model_order
    )

    by_axis: Dict[str, Dict[str, float]] = {}
    for m in ordered_models:
        axis_accum: Dict[str, List[float]] = {}
        for bench_id, sc in by_bench.get(m, {}).items():
            ax = b2a[bench_id]
            axis_accum.setdefault(ax, []).append(sc)
        by_axis[m] = {
            ax: sum(vals) / len(vals) for ax, vals in axis_accum.items() if vals
        }

    return by_axis, by_bench, ordered_models
