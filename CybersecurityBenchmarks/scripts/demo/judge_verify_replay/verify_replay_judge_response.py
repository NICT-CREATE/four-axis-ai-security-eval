#!/usr/bin/env python3
"""
Phase2 replay_judge の inproc_judge_*.json を検証する。
（CREATE-eval デモ用に oss-llm-safety-eval/scripts/RQ2 から同梱したコピー）

対象:
  <phase2-dir>/_replay_judge_scratch/inproc_judge_<bench>.json

デモの jailbreak_judge.json などを直接かけたい場合は同ディレクトリの
  run_demo_judge_verify.py
を使う。

RQ1-6 との整合（ベンチマークを増やしても「ベンチごとに別スクリプト」にはしない）:
  - allowed の定義は **verify_benchmark_implementation.BENCH_ALLOWED** と同一（15 ベンチ共通の単一ソース）。
  - unknown の事後補正は **verify_all_models_judge_response.first_allowed_in_text** と同一関数。
    RQ1-6 でも「15 通りのヒューリスティック」ではなく、CREATE-eval の allowed に基づく
    共有フォールバック + harmbench 用パッチという構造である。
  - RQ1-6 で unknown が残らないようにした主因は、(1) CREATE-eval 側の first_token 正規化
    （例: harmbench の accept→success）と judge_raw 保存、(2) verify の --fix / --fix-unknown、
    (3) それでも足りない mitre/jailbreak/bbh/mmlu 等は **rejudge_unknown_samples.py**
    による **該当サンプルだけの再 Judge**（必要なら --fill-unknown-only）である。
    Phase2 でも同様に、first_allowed_in_text で消えない分は「再 Judge」が本流の対処である。

主な機能:
  - judge_response がベンチ allowed に入っているか検証
  - unknown / invalid の件数・分布を出力
  - --fix 指定時、harmbench の accept→success 正規化を適用
  - --fix-unknown 指定時、judge_raw から unknown を補正（任意で書き戻し）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Demo bundle: dependencies are co-located in this directory (no oss-llm-safety-eval path).
_BUNDLE_DIR = Path(__file__).resolve().parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

from judge_response_normalize import normalize_label_for_bench
from label_quality_common import (
    aggregate_label_quality_summary,
    apply_fix_unknown_inplace_for_bench,
    summarize_items_for_bench,
)
from verify_benchmark_implementation import BENCH_ALLOWED


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else [data]


def _verify_one_file(
    bench: str,
    judge_path: Path,
    *,
    fix: bool,
    fix_unknown: bool,
    write: bool,
) -> dict[str, Any]:
    allowed = BENCH_ALLOWED.get(bench)
    if not allowed:
        return {"bench": bench, "ok": False, "error": f"unknown bench: {bench}", "path": str(judge_path)}

    items = _load_json(judge_path)
    fixed_count = 0

    for row in items:
        if fix and bench == "harmbench":
            raw_jr = (row.get("judge_response") or "").strip().lower()
            if raw_jr == "accept":
                row["judge_response"] = "success"

    if fix_unknown:
        fixed_count = apply_fix_unknown_inplace_for_bench(items, bench)

    if write and fixed_count > 0:
        judge_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return summarize_items_for_bench(
        items,
        bench,
        fixed_unknown_count=fixed_count,
        source_path=str(judge_path),
    )


def _discover_benches(scratch_dir: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(scratch_dir.glob("inproc_judge_*.json")):
        name = p.stem
        bench = name.replace("inproc_judge_", "", 1)
        out.append(bench)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2 replay judge_response を検証・補正する。")
    parser.add_argument("--phase2-dir", type=Path, required=True, help="例: outputs/RQ2/phase2_test")
    parser.add_argument(
        "--benches",
        type=str,
        default=None,
        help="カンマ区切りで対象ベンチを指定。省略時は inproc_judge_*.json から自動検出。",
    )
    parser.add_argument("--fix", action="store_true", help="harmbench の accept→success を補正する。")
    parser.add_argument("--fix-unknown", action="store_true", help="judge_raw から unknown を補正する。")
    parser.add_argument(
        "--write",
        action="store_true",
        help="補正内容を inproc_judge_*.json に書き戻す。未指定時は dry-run。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="検証結果 JSON の出力先。省略時は phase2-dir/verify_replay_judge_response.json",
    )
    args = parser.parse_args()

    phase2_dir = args.phase2_dir.resolve()
    scratch_dir = phase2_dir / "_replay_judge_scratch"
    if not scratch_dir.is_dir():
        raise FileNotFoundError(f"scratch dir not found: {scratch_dir}")

    benches = (
        [x.strip() for x in args.benches.split(",") if x.strip()] if args.benches else _discover_benches(scratch_dir)
    )
    if not benches:
        raise RuntimeError(f"inproc_judge_*.json not found under: {scratch_dir}")

    results: list[dict[str, Any]] = []
    all_ok = True
    for bench in benches:
        p = scratch_dir / f"inproc_judge_{bench}.json"
        if not p.exists():
            results.append({"bench": bench, "ok": False, "error": f"missing file: {p}"})
            all_ok = False
            continue
        r = _verify_one_file(bench, p, fix=args.fix, fix_unknown=args.fix_unknown, write=args.write)
        results.append(r)
        if not r.get("ok", False):
            all_ok = False

    print("=== Phase2 replay judge_response verification ===")
    for r in results:
        if r.get("error"):
            print(f"[NG] {r.get('bench')}: {r.get('error')}")
            continue
        status = "OK" if r["ok"] else "NG"
        print(
            f"[{status}] {r['bench']}: n={r['n_judge']}, unknown={r['unknown_count']}, "
            f"invalid={r['invalid_count']}, fixed={r['fixed_unknown_count']}"
        )
        print(f"      distribution: {r['distribution']}")

    out_path = args.out.resolve() if args.out else (phase2_dir / "verify_replay_judge_response.json")
    payload = {
        "phase2_dir": str(phase2_dir),
        "scratch_dir": str(scratch_dir),
        "fix": bool(args.fix),
        "fix_unknown": bool(args.fix_unknown),
        "write": bool(args.write),
        "all_ok": all_ok,
        "label_quality_summary_v1": aggregate_label_quality_summary(results),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
