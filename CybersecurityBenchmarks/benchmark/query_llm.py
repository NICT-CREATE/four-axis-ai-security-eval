# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import time
import numpy as np
import csv
import os
from datetime import datetime
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Union

from tqdm import tqdm

from .llm import LLM

LOG: logging.Logger = logging.getLogger(__name__)

# fields to (if available) add to response from the test case
test_case_fields = [
    "cwe_identifier",
    "mitre_category",
    "attack_type",
    "language",
    "line_text",
    "origin_code",
    "variant",
    "rule",
    "repo",
    "code",
    "answer",
    "challenge_type",
    "language",
    "speaking_language",
    "false_refusal_type",
    "caught_by_promptguard",
    "categories",
    "subcategories",
    "voice_tone",
    "label",
    "sample_id",
    "meta",
]


# ===== マルチターンプロンプトに使われるプロンプト変換関数のデフォルト =====
def get_another_prompt_with_response(prompt: str) -> str:
    return prompt


# ===== マルチターンプロンプトを処理する関数 =====
def process_prompts_multi_turn(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
    get_another_prompt_with_response: Callable[
        [str], str
    ] = get_another_prompt_with_response,
    num_tries: int = 10,
) -> Dict[str, Union[int, str, List[str]]]:
    # 1つのプロンプトに対して最大 `num_tries` 回まで対話を続ける
    # 各ステップで得られた応答と、それに続くプロンプトを交互に記録していく
    prompt = cast(
        str,
        test_case.get(
            "test_case_prompt",
            cast(
                str,
                test_case.get(
                    "mutated_prompt",
                ),
            ),
        ),
    )
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    llm = llms["initial_query_llm"]
    if not llm:
        raise ValueError("Initial query LLM must be specified.")
    prompt_with_history = [prompt]
    for i in range(num_tries):
        try:
            query_result = llm.chat_with_retries(
                prompt_with_history,
                guided_decode_json_schema=guided_decode_json_schema,
            )
            prompt_with_history.append(query_result)
            if i < num_tries - 1:
                new_prompt = get_another_prompt_with_response(query_result)
                prompt_with_history.append(new_prompt)
        except Exception as e:
            LOG.error(f"Query failed with error: {e}. Failed prompt: {prompt}")
            prompt = "FAIL TO QUERY"
            break

    response = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        "test_case_prompt": prompt,
        "response": prompt_with_history[-1],
        "chat_history": prompt_with_history,
    }

    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]

    if llm.model is not None:
        response["model"] = llm.model
    return response


#######################
### 主に以下から編集 by Han
###
### estimate_token_count関数追加 2025-06-18
### process_prompt_and_user_inputのアップデート 2025-06-26
from transformers import AutoTokenizer

def _set_prefill_hidden_context_if_supported(llm: LLM, prompt_id: int, pass_id: int) -> None:
    """HuggingFaceLLM の prefill hidden 保存用（run.py --save-prefill-hidden-dir）。他実装は no-op。"""
    fn = getattr(llm, "set_prefill_hidden_context", None)
    if callable(fn):
        fn(prompt_id, pass_id)


def _clear_prefill_hidden_context_if_supported(llm: LLM) -> None:
    fn = getattr(llm, "clear_prefill_hidden_context", None)
    if callable(fn):
        fn()


# ===== トークン数推定関数（transformers トークナイザ使用） =====
def estimate_token_count(model_name: str, text: str, llm: Optional[LLM] = None) -> int:
    try:
        tok = getattr(llm, "tokenizer", None) if llm is not None else None
        if tok is not None:
            return len(tok.encode(text))
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=os.environ.get("CREATE_EVAL_HF_LOCAL_FILES_ONLY", "").strip()
            in ("1", "true", "yes", "on"),
        )
        return len(tokenizer.encode(text))
    except Exception:
        return len(text.split())


# ===== プロンプトとユーザー入力を分けて送る評価のための関数 =====
def process_prompt_and_user_input(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
) -> Dict[str, Union[int, str, float]]:
    """
    任意のsystem promptとuser inputを分離して渡す形式でLLMに問い合わせを行う．
    test_caseには "system_prompt" および "user_input" フィールドが含まれている必要がある．

    Args:
        prompt_id: テストケースID
        pass_id: リトライ／バリアント番号（通常は0）
        llms: {"initial_query_llm": LLMインスタンス}
        test_case: 単一のテストケース（JSON形式）
        guided_decode_json_schema: 出力形式制約（任意）

    Returns:
        dict: LLM出力と統計情報を含む辞書
    """
    system_prompt = cast(str, test_case.get("system_prompt"))
    user_input = cast(str, test_case.get("user_input"))

    if not system_prompt or not user_input:
        raise ValueError("Both system_prompt and user_input must be provided in the test case.")

    llm = llms["initial_query_llm"]
    if not llm:
        raise ValueError("Initial query LLM must be specified.")

    _set_prefill_hidden_context_if_supported(llm, prompt_id, pass_id)
    try:
        start_time = time.time()
        response = llm.query_with_system_prompt_with_retries(
            system_prompt=system_prompt,
            prompt=user_input,
            guided_decode_json_schema=guided_decode_json_schema,
        )
        end_time = time.time()

        token_count = estimate_token_count(llm.model, response, llm=llm)
        processing_time = end_time - start_time
        token_time = processing_time / token_count if token_count > 0 else 0.0

    except Exception as e:
        LOG.error(f"Query failed with error: {e}. Failed user_input: {user_input}")
        response = "FAIL TO QUERY"
        processing_time = 0.0
        token_time = 0.0
        token_count = 0
    finally:
        _clear_prefill_hidden_context_if_supported(llm)

    response = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        # "system_prompt": system_prompt,   # system prompt動作確認済みのためコメントアウト (2025-06-26)
        # "test_case_prompt": user_input,
        "test_case_prompt": test_case.get("test_case_prompt", user_input),
        "response": response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time": processing_time,
        "token_time": token_time,
        "token_count": token_count,
    }

    # ✅ Prompt Injectionベンチマークだけ user_input を残す（他は何も影響なし）
    if test_case.get("__return_user_input__"):
        response["user_input"] = user_input

    test_case_fields = globals().get("test_case_fields", []) + [
        "injection_type", "injection_variant", "judge_question",
        "risk_category", "speaking_language",
        "false_refusal_type", "caught_by_promptguard",
    ]
    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]

    if llm.model is not None:
        response["model"] = llm.model
    return response


# ===== 単一プロンプトを処理する関数（最も基本的） =====
def process_prompt(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
) -> Dict[str, Union[int, str]]:
    # 1つのテストケースに対してプロンプトをLLMに渡し、応答と処理時間、推定トークン数を記録
    prompt = cast(
        str,
        test_case.get(
            "test_case_prompt",
            cast(
                str,
                test_case.get(
                    "mutated_prompt",
                ),
            ),
        ),
    )
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    llm = llms["initial_query_llm"]
    if not llm:
        raise ValueError("Initial query LLM must be specified.")

    _set_prefill_hidden_context_if_supported(llm, prompt_id, pass_id)
    try:
        start_time = time.time()
        query_result = llm.query_with_retries(prompt, guided_decode_json_schema=guided_decode_json_schema)
        end_time = time.time()

        token_count = estimate_token_count(llm.model, query_result, llm=llm)
        processing_time = end_time - start_time  # プロンプト処理時間
        token_time = processing_time / token_count if token_count > 0 else 0.0  # トークン処理時間

    except Exception as e:
        LOG.error(f"Query failed with error: {e}. Failed prompt: {prompt}")
        query_result = "FAIL TO QUERY"
        processing_time = 0
        token_time = 0
        token_count = 0
    finally:
        _clear_prefill_hidden_context_if_supported(llm)

    response = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        "test_case_prompt": prompt,
        "response": query_result,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time": processing_time,
        "token_time": token_time,
        "token_count": token_count,
    }

    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]

    if llm.model is not None:
        response["model"] = llm.model
    return response


# ===== 複数プロンプトを処理して結果を生成するメイン関数 =====
def query_llm_to_generate_responses(
    llms: Dict[str, LLM],
    prompt_path: Path,
    run_llm_in_parallel: int = 1,
    to_process_prompt: Callable[
        [
            int,
            int,
            Dict[str, LLM],
            Dict[str, Union[int, str]],
            Optional[str],
        ],
        Dict[str, Union[int, str]],
    ] = process_prompt,
    num_test_cases: int = 0,
    pass_k: int = 1,
    enumerate_prompt_and_pass_id: bool = True,
    guided_decode_json_schema: Optional[str] = None,
    benchmark: str = "Unknown",  # ✅ クラス名を受け取れるように
    stat_path: str = "Unknown",  # ✅ stat_path も受け取る
    response_path: str = "Unknown",
) -> List[Dict[str, Optional[str]]]:
    # 入力されたプロンプト集合をもとに、それぞれのプロンプトに対してLLMから応答を得る
    # 並列実行にも対応しており、また統計（時間・トークンなど）も収集する
    if not prompt_path.exists():
        raise ValueError("Please provide valid path for prompts.")
    prompt_dataset: List[Dict[str, Union[int, str]]] = json.loads(prompt_path.read_text())

    if num_test_cases > 0:
        dataset_size = len(prompt_dataset)
        if dataset_size > num_test_cases:
            step = dataset_size // num_test_cases
            prompt_dataset = prompt_dataset[::step][:num_test_cases]

    response_result = []
    total_processing_time = 0
    total_token_count = 0
    token_counts = []  # 各プロンプトのトークン数を記録

    start_time = time.time()  # 全体の処理開始時間

    # ✅ initial_query_llm, expansion_llm, judge_llm のどれかを使う
    llm_model = (
        llms.get("initial_query_llm")
        or llms.get("expansion_llm")
        or llms.get("judge_llm")
        or llms.get("model_under_test")  # ✅ 追加
    )

    if not llm_model:
        raise ValueError("No valid LLM found in `llms` dictionary.")

    llm_name = llm_model.model  # モデル名を取得

    if run_llm_in_parallel > 1 and not llm_model.supports_parallel_inference():
        LOG.warning(
            "%s does not support parallel inference; forcing run_llm_in_parallel=1",
            llm_name,
        )
        run_llm_in_parallel = 1

    if run_llm_in_parallel > 1:
        with ThreadPoolExecutor(max_workers=run_llm_in_parallel) as executor:
            futures = [
                executor.submit(
                    to_process_prompt,
                    idx if enumerate_prompt_and_pass_id else cast(int, test_case["prompt_id"]),
                    k if enumerate_prompt_and_pass_id else cast(int, test_case["pass_id"]),
                    llms,
                    test_case,
                    guided_decode_json_schema,
                )
                for idx, test_case in enumerate(prompt_dataset)
                for k in range(pass_k)
            ]

            with tqdm(total=len(futures), desc="Processing prompts", unit="prompt") as pbar:
                for future in futures:
                    try:
                        result = future.result()
                        response_result.append(result)

                        total_processing_time += result.get("processing_time", 0)
                        token_count = result.get("token_count", 0)
                        total_token_count += token_count
                        token_counts.append(token_count)

                        pbar.update(1)

                    except Exception as e:
                        LOG.error(f"Exception occurred in thread {future}: {e}")
    else:
        with tqdm(total=len(prompt_dataset) * pass_k, desc="Processing prompts", unit="prompt") as pbar:
            for idx, test_case in enumerate(prompt_dataset * pass_k):
                result = to_process_prompt(
                    idx % len(prompt_dataset),
                    idx // len(prompt_dataset),
                    llms,
                    test_case,
                    guided_decode_json_schema,
                )
                response_result.append(result)

                total_processing_time += result.get("processing_time", 0)
                token_count = result.get("token_count", 0)
                total_token_count += token_count
                token_counts.append(token_count)

                pbar.update(1)

    end_time = time.time()  # 全体の処理終了時間
    total_time_elapsed = end_time - start_time

    # 統計情報の計算
    token_min = np.min(token_counts) if token_counts else 0
    token_max = np.max(token_counts) if token_counts else 0
    token_mean = np.mean(token_counts) if token_counts else 0
    token_std_dev = np.std(token_counts) if token_counts else 0  # **標準偏差**
    token_q1, token_q2, token_q3 = np.percentile(token_counts, [25, 50, 75]) if token_counts else (0, 0, 0)

    # # 統計情報の出力
    # print(
    #     f"\n[Summary] Prompts: {len(response_result)}, Tokens: {total_token_count}, "
    #     f"Total Time: {total_time_elapsed:.2f}s, Avg Time/Prompt: {total_processing_time / len(response_result):.3f}s, "
    #     f"Avg Time/Token: {total_processing_time / total_token_count if total_token_count > 0 else 0:.6f}s, "
    #     f"Token Min: {token_min}, Token Max: {token_max}, "
    #     f"Token Mean: {token_mean:.1f}, Token Std Dev: {token_std_dev:.1f}, "
    #     f"Token Q1: {token_q1:.1f}, Token Q2 (Median): {token_q2:.1f}, Token Q3: {token_q3:.1f}\n"
    # )

    # ログをCSVに記録
    write_summary_to_csv(
        model_name=llm_name,
        num_prompts=len(response_result),
        num_tokens=total_token_count,
        total_time=total_time_elapsed,
        avg_prompt_time=total_processing_time / len(response_result) if response_result else 0,
        avg_token_time=total_processing_time / total_token_count if total_token_count > 0 else 0,
        token_stats={
            "min": token_min, "max": token_max, "mean": token_mean, "std_dev": token_std_dev,
            "q1": token_q1, "median": token_q2, "q3": token_q3
        },
        benchmark=benchmark,
        prompt_path=str(prompt_path),
        response_path=str(response_path),
        stat_path=stat_path,
    )

    return response_result


# `log/` ディレクトリを作成（存在しない場合）
LOG_DIR = Path("CybersecurityBenchmarks/log")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ログファイルのパス
LOG_FILE_PATH = LOG_DIR / "benchmark_log.csv"

# ===== 統計ログを書き出すヘルパー関数（CSV） =====
def write_summary_to_csv(model_name, num_prompts, num_tokens, total_time, avg_prompt_time, avg_token_time, token_stats, benchmark, prompt_path, response_path, stat_path):
    # 上記メイン処理で得られた統計をCSVファイルに出力する補助関数

    def trim_path(p: str) -> str:
        # 先頭のプロジェクトパスを除外（動的に解決）
        from pathlib import Path
        # CybersecurityBenchmarksディレクトリを動的に検出
        current_file = Path(__file__).resolve()
        benchmark_dir = current_file.parent.parent
        root = str(benchmark_dir) + "/"
        return p.replace(root, "") if p.startswith(root) else p

    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_name,
        "Avg Time/Prompt (s)": f"{avg_prompt_time:.3f}",
        "Avg Time/Token (s)": f"{avg_token_time:.6f}",
        "Benchmark": benchmark,
        "Prompt Path": trim_path(prompt_path),
        "Response Path": trim_path(response_path),
        "Stat Path": trim_path(stat_path),
        "Prompts": num_prompts,
        "Tokens": num_tokens,
        "Total Time (s)": f"{total_time:.2f}",
        "Token Min": token_stats["min"],
        "Token Max": token_stats["max"],
        "Token Mean": f"{token_stats['mean']:.1f}",
        "Token Std Dev": f"{token_stats['std_dev']:.1f}",
        "Token Q1": f"{token_stats['q1']:.1f}",
        "Token Median": f"{token_stats['median']:.1f}",
        "Token Q3": f"{token_stats['q3']:.1f}",
    }

    # CSVファイルが存在しない場合、ヘッダーを追加
    file_exists = LOG_FILE_PATH.exists()

    with open(LOG_FILE_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())

        if not file_exists:
            writer.writeheader()  # ヘッダー行を追加

        writer.writerow(log_entry)  # データを書き込む
