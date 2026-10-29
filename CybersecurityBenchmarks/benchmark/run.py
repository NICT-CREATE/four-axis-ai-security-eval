# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import os
# 再現性: torch 読み込み前に設定（別プロセス・別日でも推論を決定的にする）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import asyncio
import logging

from pathlib import Path

from typing import Callable

# httpx / httpcore の HTTP リクエスト行（INFO）を抑止（ターミナル・ログファイルの汚染防止）
for _http_log in ("httpx", "httpcore", "httpcore.connection", "httpcore.http11", "urllib3"):
    logging.getLogger(_http_log).setLevel(logging.WARNING)

from . import llm
from .benchmark import Benchmark
from .cache_handler import CacheHandler
from .LLM_generate_only_benchmark import LLMGenerateOnlyBenchmark
from .prompt_injection_benchmark import PromptInjectionBenchmark

LOG: logging.Logger = logging.getLogger(__name__)

# four-axis-ai-security-eval demo only needs generate-only + prompt-injection runners.
Benchmark.register_benchmark(LLMGenerateOnlyBenchmark)
Benchmark.register_benchmark(PromptInjectionBenchmark)


def validate_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if not args.use_precomputed_responses:
        if args.prompt_path is None:
            parser.error(
                "The following argument is required: --prompt-path, unless argument --use-precomputed-responses is specified."
            )
        if "llm_under_test" in args and args.llm_under_test is None:
            parser.error(
                "The following argument is required: --llm-under-test, unless argument --use-precomputed-responses is specified."
            )

    return args


def main(
    *,
    default_benchmark: str | None = None,
    llms_under_test: list[llm.LLM] | None = None,
    expansion_llm: llm.LLM | None = None,
    judge_llm: llm.LLM | None = None,
    default_prompt_path: str | None = None,
    default_response_path: str | None = None,
    default_judge_response_path: str | None = None,
    default_stat_path: str | None = None,
    default_use_precomputed_responses: bool = False,
    default_only_generate_llm_responses: bool = False,
    default_llm_creator: Callable[[str], llm.LLM] = llm.create,
    default_run_llm_in_parallel: int = 1,
) -> None:
    parser = argparse.ArgumentParser(description="Run the Cybersecurity Benchmarks.")

    parser.add_argument(
        "--benchmark",
        default=default_benchmark,
        required=True if default_benchmark is None else False,
        help="This is the argument to specify which benchmark to run. Currently supported benchmarks are: autocomplete, instruct, mitre.",
    )

    if llms_under_test is None:
        parser.add_argument(
            "--llm-under-test",
            help=f"LLM to benchmark provided as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
            action="append",
        )

    parser.add_argument(
        "--prompt-path",
        help="The under-test LLM(s) would be queried with the prompts in this file. This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_prompt_path,
    )

    parser.add_argument(
        "--response-path",
        help="The under-test LLM(s) would store their responses in this file. This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_response_path,
        required=default_response_path is None,
    )

    parser.add_argument(
        "--stat-path",
        help="After processing the under-test LLM's responses, the final status would be stored in this file. This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_stat_path,
        required=False,
    )

    parser.add_argument(
        "--judge-llm",
        help=f"Used for MITRE benchmark. We need to specify the llm model to judge the final expanded response as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
    )

    parser.add_argument(
        "--expansion-llm",
        help=f"Used for MITRE benchmark. We need to specify the llm model to expand the context from the initial query response as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
    )

    parser.add_argument(
        "--judge-prompt-path",
        help="Used for Autonomous Uplift benchmark. The judge LLM is queried with the prompts in this file.",
        required=False,
    )

    parser.add_argument(
        "--judge-response-path",
        help="Used for MITRE benchmark. The path to save the judge responses.",
        default=default_judge_response_path,
        required=False,
    )

    parser.add_argument(
        "--run-llm-in-parallel",
        nargs="?",
        type=int,
        const=16,
        default=default_run_llm_in_parallel,
        help=(
            "Specify the number of parallel calls to the LLM to speed up execution. "
            "When not provided, it defaults to 1 thread. "
            "When set as a flag (--run-llm-in-parallel), it defaults to 16 threads. "
            "Users can also specify the number of threads, e.g., --run-llm-in-parallel=8."
        ),
    )

    parser.add_argument(
        "--num-queries-per-prompt",
        help="Set the number of queries per prompt. Default value is 1.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use-precomputed-responses",
        action="store_true",
        default=default_use_precomputed_responses,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--only-generate-llm-responses",
        action="store_true",
        default=default_only_generate_llm_responses,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--num-test-cases",
        type=int,
        default=0,
        help="Only set this argument for testing purpose: If set to a positive value, we test the given number of prompts instead of all prompts.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable caching of LLM responses to a local file.",
    )

    parser.add_argument(
        "--cache-file",
        default="llm_cache.json",
        help="Specify the cache file path. Default is 'llm_cache.json'.",
    )

    # by Han, 2025-07-03
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        help="Use default system prompt when querying LLMs"
    )

    parser.add_argument(
        "--save-prefill-hidden-dir",
        type=str,
        default=None,
        help=(
            "HuggingFace under-test LLM only: save last-token prefill hidden state (one .npz per prompt) "
            "to this directory. Requires --prefill-hidden-layer. Default: disabled (no behavior change)."
        ),
    )
    parser.add_argument(
        "--prefill-hidden-layer",
        type=int,
        default=None,
        help=(
            "Layer index for saved hidden: 0 = embed_tokens output, k in 1..N = transformer block k-1 output. "
            "Used only when --save-prefill-hidden-dir is set."
        ),
    )
    parser.add_argument(
        "--sda-adapter-path",
        type=str,
        default=None,
        help=(
            "Optional SDA adapter directory (adapter_config.json + adapter_weights.pt). "
            "Applied only to HuggingFace under-test LLMs. Default: disabled (no behavior change)."
        ),
    )

    args = validate_arguments(parser)

    logging.basicConfig(
        format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )

    materialized_llms_under_test = []
    if not args.use_precomputed_responses:
        if llms_under_test is None and not args.llm_under_test:
            raise ValueError("No LLMs were specified.")
        materialized_llms_under_test = llms_under_test or [
            default_llm_creator(
                specification,
                use_system_prompt=args.use_system_prompt,
                sda_adapter_path=args.sda_adapter_path,
            )
            for specification in args.llm_under_test
        ]

    stat_response_path = Path(args.stat_path) if args.stat_path else None
    judge_response_path = (
        Path(args.judge_response_path) if args.judge_response_path else None
    )
    if args.judge_llm:
        judge_llm = default_llm_creator(args.judge_llm)
    if args.expansion_llm:
        expansion_llm = default_llm_creator(args.expansion_llm)

    judge_prompt_path = None
    if args.judge_prompt_path:
        if Path(args.judge_prompt_path).exists():
            judge_prompt_path = Path(args.judge_prompt_path)
        else:
            raise ValueError(
                f"Please provide valid path for judge prompts. This path does not exist: {args.judge_prompt_path}"
            )

    if args.enable_cache:
        cache_handler = CacheHandler(args.cache_file)
        for materialized_llm in materialized_llms_under_test:
            materialized_llm.cache_handler = cache_handler
    else:
        for materialized_llm in materialized_llms_under_test:
            materialized_llm.cache_handler = None

    # オプション: prefill hidden を .npz に保存（HuggingFaceLLM のみ。デフォルト無効）
    if args.save_prefill_hidden_dir or args.prefill_hidden_layer is not None:
        if not args.save_prefill_hidden_dir or args.prefill_hidden_layer is None:
            raise ValueError(
                "Both --save-prefill-hidden-dir and --prefill-hidden-layer are required together."
            )
        save_dir = Path(args.save_prefill_hidden_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        for m in materialized_llms_under_test:
            setter = getattr(m, "options", None)
            if setter is not None and hasattr(m, "set_prefill_hidden_context"):
                m.options["prefill_hidden_layer"] = int(args.prefill_hidden_layer)
                m.options["prefill_hidden_dir"] = str(save_dir)
            else:
                LOG.warning(
                    "save-prefill-hidden-dir applies only to HuggingFace local LLMs; ignoring for %s",
                    type(m).__name__,
                )

    benchmark: Benchmark = Benchmark.create_instance(
        args.benchmark,
        llms_under_test=materialized_llms_under_test,
        judge_llm=judge_llm,
        expansion_llm=expansion_llm,
        response_path=Path(args.response_path),
        judge_prompt_path=judge_prompt_path,
        judge_response_path=judge_response_path,
        stat_path=stat_response_path,
        num_test_cases=args.num_test_cases,
        pass_k=args.num_queries_per_prompt,
    )

    if args.use_precomputed_responses:
        # Using precomputed responses require a valid response path.
        response_path = Path(args.response_path)
        if not response_path.exists():
            raise ValueError(
                f"Please provide valid path for responses. This path does not exist: {response_path}"
            )
        LOG.info("Skipping LLM query and using precomputed response.")
    else:
        # If we are not using precomputed responses, then we require a valid prompt path.
        prompt_path = Path(args.prompt_path)
        if not prompt_path.exists():
            raise ValueError(
                f"Please provide valid path for prompts. This path does not exist: {prompt_path}"
            )
        LOG.info("Querying LLMs for responses...")
        benchmark.query_llm_to_generate_responses(
            prompt_path=Path(args.prompt_path),
            run_llm_in_parallel=args.run_llm_in_parallel,
            benchmark=benchmark.__class__.__name__,  # ✅ クラス名（例: "MitreFRRBenchmark"）を渡す
            stat_path=str(args.stat_path) if args.stat_path else "Unknown",  # ✅ stat_path を渡す
            response_path=str(args.response_path) if args.response_path else "Unknown",
        )

    if not args.only_generate_llm_responses:
        LOG.info("Evaluating...")
        # The variable num_test_cases is defined to specify the number of prompts that need to be evaluated. However, if we are testing multiple LLMs or if we are conducting multiple rounds of testing queries per prompt, these factors will need below steps to reflect in `num_test_cases`` during the expansion/judge step.
        if not args.use_precomputed_responses:
            if len(materialized_llms_under_test) > 0:
                args.num_test_cases = (
                    len(materialized_llms_under_test) * args.num_test_cases
                )
            if args.num_queries_per_prompt:
                args.num_test_cases = args.num_test_cases * args.num_queries_per_prompt
        asyncio.run(benchmark.run(args.num_test_cases, args.run_llm_in_parallel))


if __name__ == "__main__":
    main()
