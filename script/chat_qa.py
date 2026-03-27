#!/usr/bin/env python3
"""Interactive Christomorphic chat with multi-turn conversation memory."""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import tinker
from tinker import types

try:
    from tinker_cookbook import tokenizer_utils
except ImportError as exc:  # pragma: no cover - import guard for runtime UX
    raise RuntimeError(
        "tinker-cookbook is required to run this script. "
        "Install it with `uv pip install tinker-cookbook` and retry."
    ) from exc

# Checkpoint presets from README evaluation candidates.
CHECKPOINT_PRESETS = {
    "gpt-v6r43-120b": {
        "base_model": "openai/gpt-oss-120b",
        "model_path": "tinker://8ad467bc-72eb-51c2-bbe3-417bf8940b43:train:0/sampler_weights/final",
    },
    "gpt-r38-20b": {
        "base_model": "openai/gpt-oss-20b",
        "model_path": "tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final",
    },
}

# Default to the lighter 20b checkpoint, with env overrides available.
CHECKPOINT_ALIAS = os.getenv("CHECKPOINT_ALIAS", "gpt-r38-20b").lower()
PRESET = CHECKPOINT_PRESETS.get(CHECKPOINT_ALIAS, CHECKPOINT_PRESETS["gpt-r38-20b"])

MODEL_PATH = os.getenv("MODEL_PATH", PRESET["model_path"])
BASE_MODEL = os.getenv("BASE_MODEL", PRESET["base_model"])
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful and truthful assistant.")


def get_api_key() -> str:
    """Read API key from environment with a clear error for local/dev runs."""
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TINKER_API_KEY is not set. Export it before running chat_qa.py."
        )
    return api_key


def build_thread_prompt(system_prompt: str, history: List[Tuple[str, str]], user_text: str) -> str:
    """Render full conversation context as a single completion-style prompt."""
    lines = [system_prompt.strip(), "", "Conversation:"]
    for u, a in history:
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)




def render_prompt(tokenizer, system_prompt: str, history: List[Tuple[str, str]], user_text: str) -> str:
    """Prefer tokenizer chat template when available; fall back to plain text transcript."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "system", "content": system_prompt}]
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_text})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return build_thread_prompt(system_prompt, history, user_text)




def clean_model_answer(text: str) -> str:
    """Normalize model output and strip structured channel wrappers if present."""
    raw = text.strip()
    final_prefix = "<|channel|>final<|message|>"
    if final_prefix in raw:
        start = raw.find(final_prefix) + len(final_prefix)
        end = raw.find("<|return|>", start)
        if end != -1:
            return raw[start:end].strip()
        return raw[start:].strip()

    for tag in ("<|return|>", "<|im_end|>", "<|eot_id|>"):
        raw = raw.replace(tag, "").strip()
    return raw


def main() -> None:
    service_client = tinker.ServiceClient(api_key=get_api_key())
    sampling_client = service_client.create_sampling_client(model_path=MODEL_PATH)
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)

    history: List[Tuple[str, str]] = []

    print("Christomorphic model loaded.")
    print("Commands: /reset (clear thread), /exit (quit).\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        prompt_text = render_prompt(tokenizer, SYSTEM_PROMPT, history, user_text)
        model_input = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

        sampling_params = types.SamplingParams(
            max_tokens=512,
            temperature=0.4,
            top_p=0.9,
            stop=["\nUser:", "<|im_end|>", "<|eot_id|>"],
        )

        result = sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        ).result()

        raw_answer = tokenizer.decode(result.sequences[0].tokens).strip()
        answer = clean_model_answer(raw_answer)
        history.append((user_text, answer))

        print(f"Model: {answer}\n")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
