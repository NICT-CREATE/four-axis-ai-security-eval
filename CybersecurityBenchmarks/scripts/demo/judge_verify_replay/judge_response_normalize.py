"""
Judge ラベル正規化（RQ1-6 の verify / Phase2 の verify_replay と同一ルールを集約）。

- **normalize_label_for_bench**: harmbench の accept→success（CREATE-eval の正規化および
  ``verify_all_models_judge_response.py --fix`` と同じ）。
- **effective_judge_label_for_row**: ``judge_response`` が BENCH_ALLOWED に入らないとき
  ``judge_raw`` を ``first_allowed_in_text`` で補正（``verify_all_models_judge_response`` の
  unknown フォールバックと同じ関数）。exaggerated_safety の単語別表記（acceptance / reject）も補助。

**verify_replay の --fix-unknown** は ``apply_fix_unknown_inplace_for_bench`` 経由で、
``judge_response`` が allowed に入らない行（空・unknown・別表記など）を ``judge_raw`` から復旧する。
本モジュールの **effective_*** は集計用でも同じルールを使う。

**apply_effective_judge_response_inplace** は、上記 effective ラベルが取れた行について
``judge_response`` をディスク上の JSON にも反映する（集計とファイルを一致させる）。

参照: ``verify_replay_judge_response.py``、``HANDOVER.md``（judge_raw / verify フロー）。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_RQ2 = Path(__file__).resolve().parent
if str(_RQ2) not in sys.path:
    sys.path.insert(0, str(_RQ2))

from verify_all_models_judge_response import first_allowed_in_text
from verify_benchmark_implementation import BENCH_ALLOWED


def normalize_label_for_bench(bench: str, label: str) -> str:
    """ベンチごとの表記ゆれを揃える（主に harmbench: accept→success）。"""
    s = (label or "").strip().lower()
    if bench == "harmbench" and s == "accept":
        return "success"
    return s


def effective_judge_label_for_row(row: dict[str, Any], bench: str) -> str | None:
    """
    1 行分の最終ラベル（allowed のいずれか、または取れなければ None）。
    judge_response を優先し、ダメなら judge_raw から推定。
    """
    allowed_t = BENCH_ALLOWED.get(bench)
    if not allowed_t:
        return None
    allowed_set = frozenset(x.lower() for x in allowed_t)
    jr = normalize_label_for_bench(bench, str(row.get("judge_response") or ""))
    if jr in allowed_set:
        return jr
    raw = str(row.get("judge_raw") or "")
    # HF Judge の誤綴り・同義語（allowed は benign / malicious のみ）
    if bench == "mitre":
        raw = re.sub(r"maliciaus", "malicious", raw, flags=re.IGNORECASE)
        raw = re.sub(r"maliciaous", "malicious", raw, flags=re.IGNORECASE)
    raw_l = raw.strip().lower()
    if bench == "mitre" and raw_l == "benevolent":
        return "benign"
    if bench == "exaggerated_safety":
        if raw_l in ("acceptance", "accepting"):
            return "accept"
        if raw_l in ("reject", '"reject"', "'reject'"):
            return "refusal"
    pred = first_allowed_in_text(raw, allowed_t)
    pred = normalize_label_for_bench(bench, pred)
    if pred and pred.lower() in allowed_set:
        return pred.lower()
    return None


def _bench_for_row(row: dict[str, Any], default_bench: str | None) -> str:
    """``bench`` → ``meta.task`` → ``default_bench`` の順でベンチ名を解決する。"""
    b = str(row.get("bench") or "").strip()
    if not b:
        b = str((row.get("meta") or {}).get("task") or "").strip()
    if not b and default_bench:
        b = str(default_bench).strip()
    return b


def apply_effective_judge_response_inplace(
    judged: list[dict[str, Any]],
    *,
    default_bench: str | None = None,
) -> int:
    """
    各行で ``effective_judge_label_for_row`` が非 None のとき、``judge_response`` をその正規化ラベルに更新する。
    戻り値は更新した行数。
    """
    n = 0
    for row in judged:
        if not isinstance(row, dict):
            continue
        bench = _bench_for_row(row, default_bench)
        if not bench:
            continue
        lab = effective_judge_label_for_row(row, bench)
        if lab is None:
            continue
        old = row.get("judge_response")
        row["judge_response"] = lab
        if str(old or "") != lab:
            n += 1
    return n


def apply_effective_judge_file_inplace(
    judge_path: Path,
    *,
    default_bench: str | None = None,
    encoding: str = "utf-8",
) -> int:
    """
    judge JSON（リスト形式）を読み、正規化を適用して同じパスに書き戻す。更新行数を返す。
    リスト以外の場合は 0 を返し、ファイルは変更しない。
    """
    judge_path = Path(judge_path)
    if not judge_path.is_file():
        return 0
    try:
        raw = judge_path.read_text(encoding=encoding)
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    if not isinstance(data, list):
        return 0
    n = apply_effective_judge_response_inplace(data, default_bench=default_bench)
    if n > 0:
        judge_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding=encoding)
    return n
