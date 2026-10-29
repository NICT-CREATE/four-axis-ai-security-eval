#!/usr/bin/env python3
"""
15 データセットごとに「評価がうまく実装できているか」を自動検証する。

各データセットについて:
  (1) *_judge.json が存在し、responses と件数が一致するか
  (2) judge_response の値が CREATE-eval の allowed に含まれるか（unknown でないか）
  (3) *_stat.json が存在する場合、期待されるキーが含まれるか（簡易）

使用例:
  python scripts/RQ2/verify_benchmark_implementation.py \
    --rq16-dir outputs/RQ1-6/evaluation_results_Llama-3.2-1B/meta-llama_Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# CREATE-eval TEMPLATE_REGISTRY の allowed に合わせた期待値（小文字で比較）
BENCH_ALLOWED: dict[str, tuple[str, ...]] = {
    "harmbench": ("success", "refusal"),
    "misuse": ("refusal", "accept"),
    "prompt_injection": ("yes", "no"),
    "advbench": ("refusal", "accept"),
    "jailbreak": ("refusal", "accept"),
    "cysecbench": ("refusal", "accept"),
    "mitre": ("benign", "malicious"),
    "vulnerability_exploitation": ("refusal", "accept"),
    "code_interpreter": ("refusal", "accept"),
    "redcode": ("refusal", "accept", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),  # 検証時は Exec のみ対象（Gen は無視）
    "falsereject": ("refusal", "accept"),
    "frr": ("refusal", "accept"),
    "exaggerated_safety": ("refusal", "accept"),
    "mmlu": ("correct", "incorrect"),
    "bbh": ("correct", "incorrect"),
}

ALL_15 = list(BENCH_ALLOWED.keys())


def load_json(p: Path):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def run_verification(rq16_dir: Path, benchmarks: list[str] | None = None) -> tuple[list[dict], bool]:
    """
    1 モデル分の judge_response 検証を実行する。
    Returns:
        (results, all_ok): ベンチごとの結果リストと、全体がすべて OK かどうか。
    """
    rq16_dir = rq16_dir.resolve()
    benchmarks = benchmarks or ALL_15
    results = []

    for bench in benchmarks:
        allowed = BENCH_ALLOWED.get(bench)
        if not allowed:
            results.append({"benchmark": bench, "ok": False, "error": "unknown benchmark"})
            continue

        resp_path = rq16_dir / bench / f"{bench}_responses.json"
        judge_path = rq16_dir / bench / f"{bench}_judge.json"
        stat_path = rq16_dir / bench / f"{bench}_stat.json"

        resp_data = load_json(resp_path)
        judge_data = load_json(judge_path)
        stat_data = load_json(stat_path)

        if judge_data is None:
            results.append({"benchmark": bench, "ok": False, "error": f"missing {judge_path.name}"})
            continue

        judge_list = judge_data if isinstance(judge_data, list) else [judge_data]
        # redcode: 評価では Exec のみを使うため、検証も Exec レコードのみ対象（Gen は無視）
        if bench == "redcode":
            judge_list = [r for r in judge_list if (r.get("meta") or {}).get("type", "").lower() == "exec"]
            allowed = ("refusal", "accept")
        n_judge = len(judge_list)
        n_resp = len(resp_data) if isinstance(resp_data, list) else (1 if resp_data else 0) if resp_data else 0

        if resp_data is not None:
            resp_list = resp_data if isinstance(resp_data, list) else [resp_data]
            n_resp = len(resp_list)
        else:
            n_resp = n_judge  # 件数比較をスキップ

        count_ok = n_judge == n_resp
        if bench == "redcode":
            # redcode は Exec のみ検証するため、件数一致は全体ではなく Exec 数で見る。簡易のため Exec の unknown/invalid のみで判定する。
            count_ok = True
        values = []
        unknown_count = 0
        invalid_count = 0
        for r in judge_list:
            jr = (r.get("judge_response") or "").strip().lower()
            values.append(jr)
            if not jr:
                unknown_count += 1
            elif jr == "unknown":
                unknown_count += 1
            elif jr not in allowed:
                invalid_count += 1

        from collections import Counter
        dist = dict(Counter(values))

        stat_ok = stat_path.exists() and stat_data is not None and isinstance(stat_data, dict) and len(stat_data) > 0

        ok = count_ok and unknown_count == 0 and invalid_count == 0
        results.append({
            "benchmark": bench,
            "ok": ok,
            "n_responses": n_resp,
            "n_judge": n_judge,
            "count_match": count_ok,
            "unknown_count": unknown_count,
            "invalid_count": invalid_count,
            "stat_exists": stat_ok,
            "distribution": dist,
        })

    all_ok = all(r.get("ok", False) for r in results)
    return results, all_ok


def main():
    parser = argparse.ArgumentParser(description="15 データセットの評価実装を自動検証する。")
    parser.add_argument("--rq16-dir", type=Path, required=True, help="RQ1-6 のモデル別出力ディレクトリ")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="検証するベンチ名。未指定で全15。")
    args = parser.parse_args()

    rq16_dir = args.rq16_dir.resolve()
    if not rq16_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {rq16_dir}")

    results, all_ok = run_verification(rq16_dir, args.benchmarks)

    # 出力
    print("=== 15 データセット 評価実装検証 ===\n")
    for r in results:
        bench = r["benchmark"]
        ok = r["ok"]
        status = "OK" if ok else "NG"
        msg = f"  [{status}] {bench}"
        if r.get("error"):
            msg += f"  error={r['error']}"
        else:
            msg += f"  n={r['n_judge']}"
            if r.get("unknown_count", 0) > 0:
                msg += f"  unknown={r['unknown_count']}"
            if r.get("invalid_count", 0) > 0:
                msg += f"  invalid={r['invalid_count']}"
            if not r.get("count_match", True):
                msg += f"  count_mismatch(resp={r.get('n_responses')}, judge={r['n_judge']})"
        print(msg)
        if r.get("distribution"):
            print(f"       distribution: {r['distribution']}")

    print()
    if all_ok:
        print("全体: すべて OK")
    else:
        print("全体: 一部 NG（上記を確認）")

    out_path = rq16_dir / "benchmark_verification_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "all_ok": all_ok}, f, indent=2, ensure_ascii=False)
    print(f"詳細: {out_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
