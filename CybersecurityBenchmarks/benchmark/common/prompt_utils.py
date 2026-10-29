# CybersecurityBenchmarks/common/prompt_utils.py

from typing import Optional

from ..query_llm import process_prompt, process_prompt_and_user_input

# -----------------------------------------------------------------------------
# プロンプト加工関数（test_case から system prompt + user input を生成）
# -----------------------------------------------------------------------------
def make_prompt_processor(
    use_system_prompt: bool,
    default_system_prompt: str,
    *,
    swap_testcase_and_user_input: bool = False,
):
    """
    プロンプト加工関数を生成する．

    swap_testcase_and_user_input=True の場合，test_case_promptをsystem_promptに，
    user_inputフィールドをそのままuser_inputに使う構成になる．

    use_system_prompt=True の場合は default_system_prompt をsystem_promptの前に連結する．
    """

    def wrapped_process_prompt(
        prompt_id: int,
        pass_id: int,
        llms: dict,
        test_case: dict,
        guided_decode_json_schema: Optional[str] = None,
    ):
        test_case = test_case.copy()

        # Case 1: Prompt Injectionベンチマーク対応（swap）
        if swap_testcase_and_user_input:
            base_system_prompt = test_case.get("test_case_prompt", "")
            user_input = test_case.get("user_input", "")
            system_prompt = (
                f"{default_system_prompt}\n{base_system_prompt}" if use_system_prompt else base_system_prompt
            )
            test_case["__return_user_input__"] = True

        # Case 2: 通常ベンチマーク
        else:
            system_prompt = default_system_prompt if use_system_prompt else ""
            user_input = test_case.get("test_case_prompt") or test_case.get("mutated_prompt")

        if not system_prompt or not user_input:
            raise ValueError("system_prompt or user_input is missing in the test case.")

        test_case["system_prompt"] = system_prompt
        test_case["user_input"] = user_input

        return process_prompt_and_user_input(
            prompt_id, pass_id, llms, test_case, guided_decode_json_schema
        )

    # use_system_prompt=False かつ swap=False のときだけ process_prompt を使う
    if not use_system_prompt and not swap_testcase_and_user_input:
        return process_prompt
    else:
        return wrapped_process_prompt
