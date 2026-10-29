"""
PLaMo 3 inference in an isolated subprocess with transformers 4.57.x.

The main CREATE-eval venv uses transformers>=4.58 (Qwen3.5). PLaMo's model card
requires transformers<=4.57.1. When a PLaMo model is requested, we spawn a
long-lived worker using CREATE_EVAL_PLAMO_PYTHON (default: <repo>/.venv-plamo/bin/python).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import LLM

LOG = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PLAMO_VENV_PYTHON = _REPO_ROOT / ".venv-plamo" / "bin" / "python"
_WORKER_MODULE = "CybersecurityBenchmarks.benchmark.plamo_worker"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def running_on_plamo_transformers() -> bool:
    """True when the current interpreter already has transformers 4.57.x."""
    try:
        import transformers as t

        parts = t.__version__.split(".")
        return int(parts[0]) == 4 and int(parts[1]) == 57
    except Exception:
        return False


def is_plamo_runner_spec(spec: str) -> bool:
    model_part = spec.split("::")[-1] if "::" in spec else spec
    return "plamo" in model_part.lower()


def any_model_is_plamo(
    models: list[str],
    spec_for_runner: Optional[Any] = None,
) -> bool:
    for m in models:
        spec = spec_for_runner(m) if spec_for_runner is not None else m
        if is_plamo_runner_spec(spec):
            return True
    return False


def all_models_are_plamo(
    models: list[str],
    spec_for_runner: Optional[Any] = None,
) -> bool:
    if not models:
        return False
    for m in models:
        spec = spec_for_runner(m) if spec_for_runner is not None else m
        if not is_plamo_runner_spec(spec):
            return False
    return True


def generation_python_exec(
    models: list[str],
    default_exec: str,
    spec_for_runner: Optional[Any] = None,
) -> str:
    """
    Demo generate subprocess Python.
    All-under-test models PLaMo -> .venv-plamo; mixed -> main venv (+ worker subprocess).
    """
    if not all_models_are_plamo(models, spec_for_runner):
        return default_exec
    py = resolve_plamo_python()
    return str(py) if py is not None else default_exec


def apply_plamo_env(
    env: dict[str, str],
    models: list[str],
    spec_for_runner: Optional[Any] = None,
) -> dict[str, str]:
    """Set CREATE_EVAL_PLAMO_PYTHON when any under-test model is PLaMo."""
    if not any_model_is_plamo(models, spec_for_runner):
        return env
    py = resolve_plamo_python()
    if py is not None:
        env = dict(env)
        env["CREATE_EVAL_PLAMO_PYTHON"] = str(py)
    return env


def resolve_plamo_python() -> Path | None:
    """Return the Python executable for the PLaMo (transformers 4.57) venv, if usable."""
    cached = getattr(resolve_plamo_python, "_cached", None)
    if cached is not None:
        return cached

    if _env_flag("CREATE_EVAL_PLAMO_DISABLE_SUBPROCESS"):
        resolve_plamo_python._cached = None  # type: ignore[attr-defined]
        return None
    raw = os.environ.get("CREATE_EVAL_PLAMO_PYTHON", "").strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw))
    candidates.append(_DEFAULT_PLAMO_VENV_PYTHON)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            proc = subprocess.run(
                [
                    str(path),
                    "-c",
                    "import transformers as t; v=t.__version__.split('.'); "
                    "assert (int(v[0]), int(v[1])) == (4, 57), t.__version__",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode == 0:
                resolve_plamo_python._cached = path  # type: ignore[attr-defined]
                return path
            LOG.debug("PLaMo venv %s rejected: %s", path, (proc.stderr or proc.stdout).strip())
        except Exception as e:
            LOG.debug("PLaMo venv check failed for %s: %s", path, e)
    resolve_plamo_python._cached = None  # type: ignore[attr-defined]
    return None


def plamo_messages_to_completion_prompt(messages: List[Dict[str, str]]) -> str:
    """Same logic as HuggingFaceLLM PLaMo path (user-only completion prefix)."""
    user_parts = [
        str(m.get("content", "")).strip()
        for m in messages
        if m.get("role") == "user" and str(m.get("content", "")).strip()
    ]
    if user_parts:
        return user_parts[-1]
    parts = [str(m.get("content", "")).strip() for m in messages if str(m.get("content", "")).strip()]
    return "\n\n".join(parts)


class PlamoSubprocessLLM(LLM):
    """Delegates generation to a subprocess running transformers 4.57.x."""

    def __init__(self, model: str, plamo_python: Path) -> None:
        super().__init__(model)
        self._plamo_python = plamo_python
        self.options: Dict[str, Any] = {}
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._start_worker()

    def supports_parallel_inference(self) -> bool:
        return False

    def valid_models(self) -> list[str]:
        return []

    def _worker_env(self) -> dict[str, str]:
        return os.environ.copy()

    def _start_worker(self) -> None:
        cmd = [
            str(self._plamo_python),
            "-m",
            _WORKER_MODULE,
            self.model,
        ]
        LOG.info(
            "PLaMo: starting subprocess worker (%s) with transformers 4.57 venv: %s",
            self.model,
            self._plamo_python,
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
            cwd=str(_REPO_ROOT),
        )
        ready = self._read_line_unlocked(timeout_s=600)
        if ready.get("status") != "ready":
            err = self._drain_stderr()
            self._terminate_worker()
            raise RuntimeError(
                f"PLaMo worker failed to start for {self.model}: {ready!r} stderr={err!r}"
            )
        LOG.info(
            "PLaMo subprocess ready (transformers %s)",
            ready.get("transformers_version", "?"),
        )

    def _drain_stderr(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read() or ""
        except Exception:
            return ""

    def _read_line_unlocked(self, timeout_s: float = 300) -> dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("PLaMo worker is not running")
        line = self._proc.stdout.readline()
        if not line:
            err = self._drain_stderr()
            raise RuntimeError(f"PLaMo worker exited unexpectedly. stderr={err!r}")
        line = line.strip()
        if not line:
            return self._read_line_unlocked(timeout_s=timeout_s)
        return json.loads(line)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.stdin is None:
                raise RuntimeError("PLaMo worker is not running")
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            return self._read_line_unlocked()

    def _terminate_worker(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    def __del__(self) -> None:
        try:
            self._terminate_worker()
        except Exception:
            pass

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            local_only = _env_flag("CREATE_EVAL_HF_LOCAL_FILES_ONLY") or _env_flag("HF_HUB_OFFLINE")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                trust_remote_code=True,
                local_files_only=local_only,
            )
        return self._tokenizer

    def _generate_from_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = plamo_messages_to_completion_prompt(messages)
        resp = self._request({"cmd": "generate", "prompt": prompt})
        if resp.get("status") != "ok":
            raise RuntimeError(f"PLaMo worker generate failed: {resp!r}")
        return str(resp.get("text", ""))

    def query(self, prompt: str, guided_decode_json_schema: Optional[str] = None) -> str:
        return self._generate_from_messages([{"role": "user", "content": prompt}])

    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        return self._generate_from_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )

    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        for i, turn in enumerate(prompt_with_history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        return self._generate_from_messages(messages)

    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
    ) -> str:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for i, turn in enumerate(prompt_with_history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        return self._generate_from_messages(messages)


def try_create_plamo_subprocess_llm(
    model: str,
    *,
    use_system_prompt: bool = False,
) -> PlamoSubprocessLLM | None:
    if "plamo" not in model.lower():
        return None
    if running_on_plamo_transformers():
        return None
    py = resolve_plamo_python()
    if py is None:
        return None
    try:
        llm = PlamoSubprocessLLM(model=model, plamo_python=py)
        llm.options["use_system_prompt"] = use_system_prompt
        return llm
    except Exception as e:
        LOG.warning("PLaMo subprocess worker unavailable (%s); use in-process fallback", e)
        return None
