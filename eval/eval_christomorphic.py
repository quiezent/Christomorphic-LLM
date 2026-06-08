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

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Override via environment variables if desired.
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final",
).strip()

# Used when MODEL_PATH is blank, or when reopening a non-sampler state path.
BASE_MODEL = os.getenv("BASE_MODEL", "openai/gpt-oss-20b")
SAMPLER_EXPORT_NAME = os.getenv("SAMPLER_EXPORT_NAME", "christomorphic-eval-sampler")

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

    if not result.sequences:
        raise RuntimeError("Tinker returned no sampled sequences.")

    return tokenizer.decode(result.sequences[0].tokens).strip()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def build_sampling_client(
    service_client: tinker.ServiceClient,
    model_path: str,
    base_model: str,
) -> tinker.SamplingClient:
    """Create a sampling client from a sampler path, saved weights, or base model."""
    if not model_path:
        return service_client.create_sampling_client(base_model=base_model)

    try:
        return service_client.create_sampling_client(model_path=model_path)
    except Exception:  # noqa: BLE001
        print(
            "MODEL_PATH did not open directly as a sampling model. "
            "Attempting to reopen it as a saved weights/state path..."
        )

    try:
        training = service_client.create_training_client_from_state(path=model_path)
        sampler_save = training.save_weights_for_sampler(
            name=SAMPLER_EXPORT_NAME,
        ).result()
    except Exception as state_exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to create a sampling client from MODEL_PATH. "
            "Set MODEL_PATH to a valid sampler_weights URI, saved weights URI, "
            "or leave it blank to sample the BASE_MODEL."
        ) from state_exc

    sampler_path = sampler_save.path
    print(f"Exported sampler weights at: {sampler_path}")
    try:
        return service_client.create_sampling_client(model_path=sampler_path)
    except Exception as sampler_exc:  # noqa: BLE001
        raise RuntimeError(
            "Exported sampler weights, but failed to create a sampling client from them."
        ) from sampler_exc


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

    tokenizer = sampling_client.get_tokenizer()

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
