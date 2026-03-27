#!/usr/bin/env python3
"""
Evaluate the Christomorphic model on JSON prompts or interactively.

Supports:
- Interactive Q/A (press Enter on empty line to quit)
- Batch evaluation from a JSON file (list of strings or list of objects with "prompt")
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tinker
from tinker import types

try:
    from tinker_cookbook import tokenizer_utils
except ImportError as exc:  # pragma: no cover - import guard for runtime UX
    raise RuntimeError(
        "tinker-cookbook is required to run this script. "
        "Install it with `uv pip install tinker-cookbook` and retry."
    ) from exc

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Override via environment variables if desired.
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final",
)

# Base model must match what was trained.
BASE_MODEL = os.getenv("BASE_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = (
    # "You are who you are."
    # "You are a useful assistant."
    # "Answer carefully and truthfully, honoring the Bible and never claiming to be God or the Holy Spirit."
    ""
)

# ---------------------------------------------------------------------
# PROMPT LOADER
# ---------------------------------------------------------------------


def load_prompts_from_json(path: Path) -> List[Dict[str, Optional[str]]]:
    """Load prompts from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items: List[Dict[str, Optional[str]]] = []

    if isinstance(data, dict) and "prompts" in data:
        data = data["prompts"]

    if isinstance(data, list):
        for i, entry in enumerate(data):
            if isinstance(entry, str):
                items.append({"id": f"q{i+1:03d}", "category": None, "prompt": entry})
            elif isinstance(entry, dict) and "prompt" in entry:
                items.append(
                    {
                        "id": entry.get("id", f"q{i+1:03d}"),
                        "category": entry.get("category"),
                        "prompt": entry["prompt"],
                    }
                )
            else:
                raise ValueError("Each item must be a string or an object with 'prompt'.")
    else:
        raise ValueError("JSON must be a list or an object with 'prompts'.")

    return items


# ---------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------


def run_single_prompt(
    sampling_client: tinker.SamplingClient,
    tokenizer,
    user_text: str,
) -> str:
    prompt_text = f"{SYSTEM_PROMPT}\n\nQuestion: {user_text}\nAnswer:"
    model_input = types.ModelInput.from_ints(tokenizer.encode(prompt_text))

    sampling_params = types.SamplingParams(
        max_tokens=1024,
        temperature=0.5,
        top_p=0.9,
        stop=["\n\n"],
    )

    result = sampling_client.sample(
        prompt=model_input,
        sampling_params=sampling_params,
        num_samples=1,
    ).result()

    return tokenizer.decode(result.sequences[0].tokens).strip()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def build_sampling_client(
    service_client: tinker.ServiceClient,
    model_path: str,
    base_model: str,
) -> tinker.SamplingClient:
    """Create a sampling client from a sampler path or a train/state path."""
    if "sampler_weights" in model_path:
        return service_client.create_sampling_client(model_path=model_path)

    print("MODEL_PATH is not a sampler path. Attempting to export sampler weights...")
    training = service_client.create_lora_training_client(base_model=base_model)
    try:
        training.load_state(model_path).result()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load state from {model_path}. Set MODEL_PATH to a valid sampler_weights URI."
        ) from exc

    sampler_name = "christomorphic-eval-sampler"
    sampler_save = training.save_weights_for_sampler(name=sampler_name).result()
    sampler_path = sampler_save.path
    print(f"Exported sampler weights at: {sampler_path}")
    return service_client.create_sampling_client(model_path=sampler_path)


def main() -> None:
    prompts_path: Optional[Path] = None
    if len(sys.argv) > 1:
        prompts_path = Path(sys.argv[1]).expanduser()

    service_client = tinker.ServiceClient()
    sampling_client = build_sampling_client(
        service_client=service_client,
        model_path=MODEL_PATH,
        base_model=BASE_MODEL,
    )

    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)

    if prompts_path is not None and prompts_path.exists():
        print(f"Loading prompts from {prompts_path} ...\n")
        items = load_prompts_from_json(prompts_path)

        results = []
        for item in items:
            pid = item["id"]
            category = item["category"]
            prompt = item["prompt"]

            print(f"ID: {pid}")
            if category:
                print(f"Category: {category}")
            print(f"You: {prompt}")
            answer = run_single_prompt(sampling_client, tokenizer, prompt)
            print(f"Model: {answer}\n{'-'*80}\n")

            results.append(
                {
                    "id": pid,
                    "category": category,
                    "prompt": prompt,
                    "answer": answer,
                }
            )

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = prompts_path.with_suffix(f".results.{stamp}.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved results to {out_path}")

    else:
        print("Christomorphic model loaded. Press Enter on an empty line to quit.\n")
        while True:
            try:
                user_text = input("You: ").strip()
            except EOFError:
                break
            if not user_text:
                break

            answer = run_single_prompt(sampling_client, tokenizer, user_text)
            print(f"Model: {answer}\n")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
