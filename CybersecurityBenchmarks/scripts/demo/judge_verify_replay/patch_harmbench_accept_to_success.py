#!/usr/bin/env python3
"""
既存の harmbench_judge.json を読み、judge_response が "accept" のものを "success" に一括置換する。
CREATE-eval の正規化（accept→success）と整合させるため。

使用例:
  python scripts/RQ2/patch_harmbench_accept_to_success.py \
    --base-dir outputs/RQ1-6/evaluation_results_Llama-3.2-1B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def apply_harmbench_patch(base_dir: Path, dry_run: bool = False) -> tuple[int, list[tuple[str, int]]]:
    """
    base_dir 配下の全モデルの harmbench_judge.json で accept→success を一括置換する。
    Returns: (置換した総件数, [(model_dir_name, 件数), ...])
    """
    base = base_dir.resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Not a directory: {base}")
    total_replaced = 0
    details: list[tuple[str, int]] = []
    for jpath in sorted(base.glob("*/harmbench/harmbench_judge.json")):
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        replaced = 0
        for r in items:
            jr = (r.get("judge_response") or "").strip()
            if jr.lower() == "accept":
                r["judge_response"] = "success"
                replaced += 1
        if replaced > 0:
            total_replaced += replaced
            details.append((jpath.parent.parent.name, replaced))
            if not dry_run:
                with open(jpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
    return total_replaced, details


def main():
    parser = argparse.ArgumentParser(description="harmbench_judge.json の judge_response で accept→success を一括置換する。")
    parser.add_argument("--base-dir", type=Path, required=True, help="評価結果のルート（例: evaluation_results_Llama-3.2-1B）")
    parser.add_argument("--dry-run", action="store_true", help="置換せずに変更対象件数のみ表示する。")
    args = parser.parse_args()

    base = args.base_dir.resolve()
    total_replaced, details = apply_harmbench_patch(base, dry_run=args.dry_run)
    for model_name, count in details:
        print(f"  {model_name}: {count} 件を accept→success に置換")
    if total_replaced == 0:
        print("置換対象の accept はありませんでした。")
    else:
        print(f"\n合計 {total_replaced} 件を置換しました。" + (" (dry-run のためファイルは変更していません)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    exit(main())
