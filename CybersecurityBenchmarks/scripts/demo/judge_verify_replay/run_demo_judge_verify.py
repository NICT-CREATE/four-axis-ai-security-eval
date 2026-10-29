#!/usr/bin/env python3
"""
CREATE-eval デモが出力した ``<bench>_judge.json`` を Phase2 形式に置き、
同梱の ``verify_replay_judge_response.py`` で検証・補正する。

例:
  python run_demo_judge_verify.py \\
    --judge-json ../../../outputs/demo/test_v2/with_syspro/bench15/jailbreak/jailbreak_judge.json \\
    --bench jailbreak --fix-unknown --write --sync-back

``--phase2-dir`` を省略すると ``<judge-json の親>/.judge_verify_work`` を使う。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo judge JSON → verify_replay（同梱バンドル）")
    parser.add_argument(
        "--judge-json",
        type=Path,
        required=True,
        help="例: .../jailbreak/jailbreak_judge.json",
    )
    parser.add_argument(
        "--bench",
        type=str,
        required=True,
        help="BENCH_ALLOWED のキー（例: jailbreak, harmbench, advbench）",
    )
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=None,
        help="省略時は judge-json の親ディレクトリ直下の .judge_verify_work",
    )
    parser.add_argument("--fix", action="store_true", help="harmbench: accept→success")
    parser.add_argument("--fix-unknown", action="store_true", help="judge_raw から unknown を補正")
    parser.add_argument("--write", action="store_true", help="補正を inproc JSON に書き戻す")
    parser.add_argument(
        "--sync-back",
        action="store_true",
        help="--write 後、補正済み inproc を元の --judge-json にコピーし直す",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="検証レポート JSON（verify_replay の --out にそのまま渡す）",
    )
    args = parser.parse_args()

    judge_json = args.judge_json.resolve()
    if not judge_json.is_file():
        print(f"error: not a file: {judge_json}", file=sys.stderr)
        return 2

    phase2 = (args.phase2_dir or judge_json.parent / ".judge_verify_work").resolve()
    scratch = phase2 / "_replay_judge_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    inproc = scratch / f"inproc_judge_{args.bench}.json"
    shutil.copy2(judge_json, inproc)

    bundle = Path(__file__).resolve().parent
    verify_py = bundle / "verify_replay_judge_response.py"
    cmd = [
        sys.executable,
        str(verify_py),
        "--phase2-dir",
        str(phase2),
        "--benches",
        args.bench,
    ]
    if args.fix:
        cmd.append("--fix")
    if args.fix_unknown:
        cmd.append("--fix-unknown")
    if args.write:
        cmd.append("--write")
    if args.out is not None:
        cmd.extend(["--out", str(args.out.resolve())])

    proc = subprocess.run(cmd, cwd=str(bundle))
    # 検証が NG（unknown 残り）でも --write で inproc は更新済みなので sync-back は実行する
    if args.sync_back:
        if not args.write:
            print("warning: --sync-back ignored without --write", file=sys.stderr)
        else:
            shutil.copy2(inproc, judge_json)
            print(f"synced: {inproc} -> {judge_json}")

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
