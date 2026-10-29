# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from .llm import LLM

LOG: logging.Logger = logging.getLogger(__name__)

from pathlib import Path
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration class for benchmarks (for compatibility with upstream benchmarks)."""
    llms_under_test: list[LLM]
    response_path: Path
    judge_prompt_path: Optional[Path] = None
    judge_response_path: Optional[Path] = None
    stat_path: Optional[Path] = None
    num_test_cases: int = 0
    expansion_llm: Optional[LLM] = None
    judge_llm: Optional[LLM] = None
    pass_k: int = 1
    agent: Optional[str] = None
    enable_lf: Optional[str] = None
    cot_end_tokens: list[str] | None = None
    additional_args: Optional[str] = None
    container_repository: Optional[str] = None
    prompt_path: Optional[str] = None
    max_concurrency: int = 1
    truncate_input: bool = False
    input_modality: Optional[str] = None


class Benchmark(ABC):
    _registered_benchmarks: Dict[str, Type["Benchmark"]] = {}

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        *,
        llms_under_test: list[LLM] | None = None,
        response_path: Path | None = None,
        judge_prompt_path: Optional[Path] = None,
        judge_response_path: Optional[Path] = None,
        stat_path: Optional[Path] = None,
        num_test_cases: int = 0,
        expansion_llm: Optional[LLM] = None,
        judge_llm: Optional[LLM] = None,
        pass_k: int = 1,
    ) -> None:
        # Support both BenchmarkConfig (new style) and keyword arguments (old style)
        if config is not None:
            # New style: use BenchmarkConfig
            self.response_path: Path = config.response_path
            self.num_test_cases: int = config.num_test_cases
            self.llms_under_test: list[LLM] = config.llms_under_test
            self.judge_response_path = config.judge_response_path
            self.stat_path = config.stat_path
            self.expansion_llm = config.expansion_llm
            self.judge_llm = config.judge_llm
            self.pass_k = config.pass_k
            self.judge_prompt_path = config.judge_prompt_path
        else:
            # Old style: use keyword arguments
            if llms_under_test is None or response_path is None:
                raise ValueError("Either config or llms_under_test and response_path must be provided")
            self.response_path: Path = response_path
            self.num_test_cases: int = num_test_cases
            self.llms_under_test: list[LLM] = llms_under_test
            self.judge_response_path = judge_response_path
            self.stat_path = stat_path
            self.expansion_llm = expansion_llm
            self.judge_llm = judge_llm
            self.pass_k = pass_k
            self.judge_prompt_path = judge_prompt_path

    @classmethod
    def register_benchmark(cls, benchmark_class: Type["Benchmark"]) -> None:
        """
        Registers all the benchmark kinds with the benchmark factory.
        """
        for benchmark_kind in benchmark_class.return_kind():
            cls._registered_benchmarks[benchmark_kind] = benchmark_class

    @classmethod
    def create_instance(
        cls, benchmark_kind: str, *args: Any, **kwargs: Any
    ) -> Benchmark:
        if benchmark_kind in cls._registered_benchmarks:
            benchmark_class: Type[Benchmark] = cls._registered_benchmarks[
                benchmark_kind
            ]
            # Check if the benchmark class expects BenchmarkConfig (new style) or keyword arguments (old style)
            import inspect
            sig = inspect.signature(benchmark_class.__init__)
            params = list(sig.parameters.keys())

            # If the first parameter (after self) is 'config', use BenchmarkConfig
            if len(params) > 1 and params[1] == 'config':
                # Create BenchmarkConfig from kwargs
                config = BenchmarkConfig(**kwargs)
                return benchmark_class(config=config)
            else:
                # Use keyword arguments directly (old style)
                # pyre-ignore [45]: This function is only used to create instances of subclasses of `Benchmark`, whose abstract methods are implemented
                return benchmark_class(*args, **kwargs)
        raise ValueError(
            f"Unknown benchmark kind: {benchmark_kind}, the registered benchmarks are: {list(cls._registered_benchmarks.keys())}"
        )

    @classmethod
    def return_kind(cls) -> list[str]:
        """
        Returns the kind(s) of benchmark that is supported. One benchmark may support multiple kinds.
        """
        raise NotImplementedError(
            "Each benchmark must implement this method to retun the kind for run script."
        )

    def extract_content_in_code_blocks(self, input: str) -> list[str]:
        # Using regular expression to find content between code blocks ```
        return re.findall(r"```(.*?)```", input, re.DOTALL)

    @abstractmethod
    def query_llm_to_generate_responses(
        self,
        prompt_path: Path,
        run_llm_in_parallel: int = 1,
    ) -> None:
        """
        Processes a dataset of prompts by sending them to the LLM and saving the responses in a file.
        """
        pass

    @abstractmethod
    async def run(
        self,
        num_test_cases: int = 0,
        run_llm_in_parallel: int = 1,
    ) -> None:
        """
        Processes a LLM-response.
        """
        pass
