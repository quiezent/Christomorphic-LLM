#!/usr/bin/env python3
"""
Evaluate the Christomorphic model on JSON prompts or interactively.

Supports:
- Interactive Q/A (press Enter on empty line to quit)
- Batch evaluation from a JSON file (list of strings or list of objects with "prompt")

Notes:
- Set MODEL_PATH to your sampler weights (Stage 3 recommended), or override via env MODEL_PATH.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tinker
from tinker import types

# Ensure local cookbook modules are importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinker_cookbook import tokenizer_utils

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Override via environment variable MODEL_PATH if desired.
MODEL_PATH = os.getenv("MODEL_PATH", "tinker://REPLACE_WITH_STAGE3_SAMPLER_PATH")

# Base model must match what was trained
BASE_MODEL = "meta-llama/Llama-3.2-1B"

SYSTEM_PROMPT = (
    #"You are who you are."
    #"You are a useful assistant."
    #"Answer carefully and truthfully, honoring the Bible and never claiming to be God or the Holy Spirit."
)

# ---------------------------------------------------------------------
# PROMPT LOADER
# ---------------------------------------------------------------------

def load_prompts_from_json(path: Path) -> List[Dict[str, Optional[str]]]:
    """
    Load prompts from a JSON file.

    Supported formats:
      1) List of strings:
         ["Who is Jesus Christ?", "What is photosynthesis?"]
      2) List of objects with 'prompt' (and optional 'id'/'category'):
         [
           {"id": "bible_001", "prompt": "Summarize Romans."},
           {"category": "science", "prompt": "Explain E=mc^2."}
         ]
      3) Object with 'prompts' key (list as above):
         {"prompts": [ ... ]}
    """
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
        max_tokens=256,
        temperature=0.3,
        top_p=0.9,
        stop=["\n\n"],  # allow a short multi-sentence answer
    )

    result = sampling_client.sample(
        prompt=model_input,
        sampling_params=sampling_params,
        num_samples=1,
    ).result()

    answer = tokenizer.decode(result.sequences[0].tokens).strip()
    return answer


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    prompts_path: Optional[Path] = None
    if len(sys.argv) > 1:
        prompts_path = Path(sys.argv[1]).expanduser()

    # Connect to Tinker; ensure we have a sampler path (convert from state if needed)
    service_client = tinker.ServiceClient()

    sampler_model_path = MODEL_PATH
    if "sampler_weights" not in sampler_model_path:
        print(
            "MODEL_PATH is not a sampler path. Attempting to load state and export sampler weights..."
        )
        training = service_client.create_lora_training_client(base_model=BASE_MODEL)
        try:
            training.load_state(sampler_model_path).result()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load state from {sampler_model_path}. "
                f"Set MODEL_PATH to a valid sampler_weights URI."
            ) from exc
        sampler_name = "christomorphic-eval-sampler"
        # Save sampler weights and get the path explicitly
        sampler_save = training.save_weights_for_sampler(name=sampler_name).result()
        sampler_path = sampler_save.path
        print(f"Exported sampler weights at: {sampler_path}")
        sampler_model_path = sampler_path
        sampling_client = service_client.create_sampling_client(model_path=sampler_model_path)
    else:
        sampling_client = service_client.create_sampling_client(model_path=sampler_model_path)

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

        # Dump results next to the prompts file with timestamp
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = prompts_path.with_suffix(f".results.{stamp}.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved results to {out_path}")

    else:
        # Interactive mode
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
