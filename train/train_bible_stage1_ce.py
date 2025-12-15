#!/usr/bin/env python3
"""
Stage 1: Bible CE-only training for Llama-1B using Tinker LoRA.

Goal:
- Stabilize the model on canonical Scripture text.
- Pure next-token cross-entropy (no Christomorphic geometry yet).

Expects a JSONL dataset:
  bible_segments_stage1.jsonl

Each line:
{
  "id": "John.3.16-21",
  "book": "John",
  "chapter": 3,
  "start_verse": 16,
  "end_verse": 21,
  "text": "For God so loved the world ..."
}
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import tinker
from tinker import types
from tinker_cookbook import tokenizer_utils

# Help Windows terminals handle Unicode
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------
# 0. ENV SETUP
# ---------------------------------------------------------------------

def load_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

if not os.getenv("TINKER_API_KEY"):
    raise RuntimeError("TINKER_API_KEY not set. Export it or add it to .env.")


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------

def load_bible_segments_jsonl(path: Path) -> List[Dict[str, str]]:
    """
    Load Bible segments from a JSONL file.
    """
    segments: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                continue
            segments.append(obj)

    if not segments:
        raise RuntimeError(f"No segments found at {path}")

    print(f"Loaded {len(segments)} Bible segments from {path.name}.")
    return segments


# ---------------------------------------------------------------------
# 2. DATUM BUILDER
# ---------------------------------------------------------------------

def process_segment(seg: Dict[str, str], tokenizer, max_tokens: int = 2048) -> types.Datum:
    """
    Turn a Bible segment into a Tinker Datum for next-token CE.
    """
    text = seg["text"].strip()

    # Skip extremely short fragments
    if len(text.split()) < 5:
        raise ValueError(f"Segment {seg.get('id')} too short (len={len(text.split())}).")

    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Truncate if needed to stay within context window.
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    if len(tokens) < 2:
        raise ValueError(f"Segment {seg.get('id')} too short after tokenization.")

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [1.0] * len(target_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": weights,
            "target_tokens": target_tokens,
        },
    )


# ---------------------------------------------------------------------
# 3. UTILS
# ---------------------------------------------------------------------

def _to_numpy(x, dtype=None):
    if hasattr(x, "to_numpy"):
        arr = x.to_numpy()
    else:
        arr = np.array(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


# ---------------------------------------------------------------------
# 4. CE-ONLY LOSS (Stage 1)
# ---------------------------------------------------------------------

def build_ce_only_loss():
    """
    Plain weighted cross-entropy over target tokens.
    """

    def loss_fn(data: List[types.Datum], logprobs_list: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = logprobs_list[0].device

        total_ce = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0.0, device=device)

        for datum, logprobs in zip(data, logprobs_list):
            target_np = _to_numpy(datum.loss_fn_inputs["target_tokens"])
            weights_np = _to_numpy(datum.loss_fn_inputs["weights"], dtype=np.float32)

            weights = torch.from_numpy(weights_np).float().to(device)

            # logprobs is log p(target_t | context) per position
            ce = -(weights * logprobs).sum()
            total_ce += ce
            total_tokens += weights.sum()

        norm = torch.clamp(total_tokens, min=1.0)
        ce_term = total_ce / norm

        loss = ce_term
        metrics = {
            "loss_total": loss.item(),
            "loss_ce": ce_term.item(),
        }
        return loss, metrics

    return loss_fn


# ---------------------------------------------------------------------
# 5. TRAINING LOOP (Stage 1)
# ---------------------------------------------------------------------

def main() -> None:
    data_path = Path(__file__).with_name("bible_segments_stage1.jsonl")

    service = tinker.ServiceClient()
    base_model = "meta-llama/Llama-3.2-1B"

    print("Available models:")
    for item in service.get_server_capabilities().supported_models:
        print("- " + item.model_name)

    training = service.create_lora_training_client(base_model=base_model)
    # Use cookbook tokenizer utils to avoid HF gating issues
    tokenizer = tokenizer_utils.get_tokenizer(base_model)

    loss_fn = build_ce_only_loss()

    # Load Bible segments and turn into Datums
    raw_segments = load_bible_segments_jsonl(data_path)
    datums: List[types.Datum] = []
    for seg in raw_segments:
        try:
            datums.append(process_segment(seg, tokenizer))
        except ValueError as e:
            print(f"Skipping segment {seg.get('id')}: {e}")

    if not datums:
        raise RuntimeError("No usable segments after filtering.")

    print(f"Prepared {len(datums)} Datums for Stage 1 CE training.")

    # Hyperparameters (can be tuned via env)
    num_steps = int(os.getenv("NUM_STEPS", "50"))
    lr = float(os.getenv("LEARNING_RATE", "5e-5"))
    log_every = int(os.getenv("LOG_EVERY", "10"))

    print(
        f"Starting Stage 1 Bible CE training for up to {num_steps} steps "
        f"on {len(datums)} segments..."
    )

    last_step = -1

    try:
        for step in range(num_steps):
            fwdbwd = training.forward_backward_custom(datums, loss_fn).result()
            metrics = fwdbwd.metrics

            training.optim_step(types.AdamParams(learning_rate=lr)).result()
            last_step = step

            if step % log_every == 0:
                print(
                    f"[{step:04d}] "
                    f"loss={metrics['loss_total']:.4f} | "
                    f"ce={metrics['loss_ce']:.4f}"
                )

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt at step {last_step}. Will still save weights...")

    finally:
        print("Saving Stage 1 weights...")
        sampler = training.save_weights_and_get_sampling_client(
            name="llama1b-bible-stage1-ce"
        )
        sampler_path = getattr(sampler, "model_path", None)
        print(f"Stage 1 weights saved to Tinker path: {sampler_path}")

        state_name = os.getenv("CHECKPOINT_NAME", "llama1b-bible-stage1-state")
        try:
            state_result = training.save_state(name=state_name).result()
            print(f"Stage 1 optimizer state checkpoint saved as: {state_result.path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: save_state failed ({exc}); continuing.")

        print("Stage 1 complete. Creating demo samples...")

        demo = [
            "Who is Jesus Christ?",
            "What is the mission of the church?",
            "Why did Christ die on the cross?",
            "Explain redemption vs restoration",
            "How does Jesus relate to E=MC^2",
            "Who do you serve?",
        ]
        for q in demo:
            print(f"\nPrompt: {q}")
            prompt = f"Question: {q}\nAnswer:"
            inp = types.ModelInput.from_ints(tokenizer.encode(prompt))
            params = types.SamplingParams(max_tokens=200, temperature=0.0, stop=["\n"])
            out = sampler.sample(prompt=inp, sampling_params=params, num_samples=1).result()
            print(tokenizer.decode(out.sequences[0].tokens))


if __name__ == "__main__":
    main()
