#!/usr/bin/env python3
"""
Long-lived PLaMo inference worker (transformers 4.57.x venv only).

Protocol (JSON lines on stdin/stdout):
  -> {"cmd":"generate","prompt":"..."}
  <- {"status":"ok","text":"..."}
  -> {"cmd":"shutdown"}
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _local_files_only() -> bool:
    return _env_flag("CREATE_EVAL_HF_LOCAL_FILES_ONLY") or _env_flag("HF_HUB_OFFLINE")


def _load_model(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=_local_files_only(),
    )
    device_map = os.environ.get("CREATE_EVAL_HF_DEVICE_MAP", "").strip() or "cuda:0"
    try:
        import torch as _torch

        if device_map.startswith("cuda") and not _torch.cuda.is_available():
            device_map = "cpu"
    except Exception:
        device_map = "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=_local_files_only(),
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    ).eval()

    # transformers 4.57: tie_word_embeddings; ensure lm_head shares embed if needed
    try:
        inp = model.get_input_embeddings()
        out = model.get_output_embeddings()
        if inp is not None and out is not None and out.weight.data_ptr() != inp.weight.data_ptr():
            out.weight = inp.weight
    except Exception:
        pass

    return model, tokenizer


def _generate(model: Any, tokenizer: Any, prompt: str) -> str:
    import torch

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.convert_tokens_to_ids("<|plamo:pad|>")

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kw = {
        "max_new_tokens": int(os.environ.get("CREATE_EVAL_PLAMO_MAX_NEW_TOKENS", "128")),
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": float(os.environ.get("CREATE_EVAL_PLAMO_TEMPERATURE", "1.0")),
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": pad_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kw)

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "usage: plamo_worker <model_id>"}), flush=True)
        sys.exit(1)

    model_id = sys.argv[1]
    try:
        import transformers

        tf_ver = transformers.__version__
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"transformers import failed: {e}"}), flush=True)
        sys.exit(1)

    try:
        model, tokenizer = _load_model(model_id)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"model load failed: {e}"}), flush=True)
        sys.exit(1)

    print(
        json.dumps({"status": "ready", "model": model_id, "transformers_version": tf_ver}),
        flush=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": f"invalid json: {e}"}), flush=True)
            continue

        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd == "generate":
            prompt = str(req.get("prompt", ""))
            try:
                text = _generate(model, tokenizer, prompt)
                print(json.dumps({"status": "ok", "text": text}, ensure_ascii=False), flush=True)
            except Exception as e:
                print(json.dumps({"status": "error", "message": str(e)}), flush=True)
            continue
        print(json.dumps({"status": "error", "message": f"unknown cmd: {cmd!r}"}), flush=True)


if __name__ == "__main__":
    main()
