from __future__ import annotations

from collections import Counter
from typing import Any

from judge_response_normalize import effective_judge_label_for_row, normalize_label_for_bench
from verify_benchmark_implementation import BENCH_ALLOWED


def summarize_items_for_bench(
    items: list[dict[str, Any]],
    bench: str,
    *,
    fixed_unknown_count: int = 0,
    source_path: str | None = None,
) -> dict[str, Any]:
    allowed = BENCH_ALLOWED.get(bench)
    if not allowed:
        return {
            "schema_version": "label_quality_summary_v1",
            "bench": bench,
            "ok": False,
            "error": f"unknown bench: {bench}",
            "path": source_path,
        }
    allowed_set = {str(x).lower() for x in allowed}
    values: list[str] = []
    unknown_count = 0
    invalid_count = 0
    for row in items:
        jr = normalize_label_for_bench(bench, str(row.get("judge_response") or ""))
        values.append(jr)
        if not jr or jr == "unknown":
            unknown_count += 1
        elif jr not in allowed_set:
            invalid_count += 1
    return {
        "schema_version": "label_quality_summary_v1",
        "bench": bench,
        "path": source_path,
        "ok": unknown_count == 0 and invalid_count == 0,
        "n_judge": len(items),
        "unknown_count": unknown_count,
        "invalid_count": invalid_count,
        "fixed_unknown_count": int(fixed_unknown_count),
        "distribution": dict(Counter(values)),
        "allowed": sorted(allowed_set),
    }


def aggregate_label_quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_judge = 0
    unknown = 0
    invalid = 0
    fixed = 0
    ok = True
    for r in rows:
        if r.get("error"):
            ok = False
            continue
        n_judge += int(r.get("n_judge", 0))
        unknown += int(r.get("unknown_count", 0))
        invalid += int(r.get("invalid_count", 0))
        fixed += int(r.get("fixed_unknown_count", 0))
        if not bool(r.get("ok", False)):
            ok = False
    return {
        "schema_version": "label_quality_summary_v1",
        "all_ok": bool(ok),
        "n_benches": len(rows),
        "n_judge_total": n_judge,
        "unknown_count_total": unknown,
        "invalid_count_total": invalid,
        "fixed_unknown_total": fixed,
    }


def apply_fix_unknown_inplace_for_bench(
    items: list[dict[str, Any]],
    bench: str,
) -> int:
    n = 0
    for row in items:
        lab = effective_judge_label_for_row(row, bench)
        if lab is None:
            continue
        old = str(row.get("judge_response") or "").strip().lower()
        if old != lab:
            row["judge_response"] = lab
            n += 1
    return n
