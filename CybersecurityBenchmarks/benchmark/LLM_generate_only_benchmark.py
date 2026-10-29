import json
from pathlib import Path
from typing import Optional

from .benchmark import Benchmark
from .llm import LLM
from .query_llm import query_llm_to_generate_responses
from .common.prompt_utils import make_prompt_processor
from .common.default_system_prompt import DEFAULT_SYSTEM_PROMPT

# -----------------------------------------------------------------------------
# ベンチマーククラス：LLMの出力生成のみを行う
# -----------------------------------------------------------------------------
class LLMGenerateOnlyBenchmark(Benchmark):
    """
    任意のプロンプトセットに対して LLM の出力のみを生成するベンチマーク．
    Judge やスコア評価は行わない．
    """

    @classmethod
    def return_kind(cls) -> list[str]:
        """
        このベンチマークの種別（内部識別用）
        """
        return ["LLM_generate_only"]

    def query_llm_to_generate_responses(
        self,
        prompt_path: Path,
        run_llm_in_parallel: int = 1,
        benchmark: str = "LLM_generate_only",
        stat_path: str = "none",
        response_path: str = "Unknown",
    ) -> None:
        """
        各LLMに対して，指定されたプロンプトセットを用いて出力生成を行い，結果を保存する．

        Args:
            prompt_path: テストセット（JSON）のパス
            run_llm_in_parallel: 並列実行数
            benchmark: ベンチマーク種別名（通常変更不要）
            stat_path: スタット出力先（未使用であれば 'none'）
        """
        result = []
        for llm in self.llms_under_test:
            print(f"Querying {llm.model}")
            use_system_prompt = getattr(llm, "options", {}).get("use_system_prompt", False)
            prompt_func = make_prompt_processor(use_system_prompt, DEFAULT_SYSTEM_PROMPT)

            result.extend(
                query_llm_to_generate_responses(
                    {"initial_query_llm": llm},
                    prompt_path,
                    run_llm_in_parallel,
                    to_process_prompt=prompt_func,
                    num_test_cases=self.num_test_cases,
                    pass_k=self.pass_k,
                    benchmark=benchmark,
                    stat_path=stat_path,
                    response_path=response_path,
                )
            )

        self.response_path.parent.mkdir(parents=True, exist_ok=True)
        self.response_path.write_text(json.dumps(result, indent=4))

    async def run(self, num_test_cases: int = 0, run_llm_in_parallel: int = 1) -> None:
        """
        ベンチマーク実行関数（run.pyから呼ばれる）．
        このベンチマークでは評価を行わないため，ここでは何もしない．
        """
        pass
