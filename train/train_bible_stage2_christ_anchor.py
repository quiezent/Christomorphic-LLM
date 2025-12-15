#!/usr/bin/env python3
"""
Stage 2: Bible CE training with a lexical Christ anchor for Llama-1B using Tinker LoRA.

Goal:
- Re-read Scripture with extra weight on Christ-explicit segments.
- Loss remains pure next-token CE; anchor is applied via token weights.

Data:
  bible_segments_stage1.jsonl  (same as Stage 1)

Resume from:
  STAGE1_STATE_PATH = tinker://.../llama1b-bible-stage1-state
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

STAGE1_STATE_PATH = os.getenv("STAGE1_STATE_PATH")
if not STAGE1_STATE_PATH:
    raise RuntimeError(
        "STAGE1_STATE_PATH not set. "
        "Set it to the tinker://... path printed by Stage 1 save_state()."
    )


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
# 2. CHRIST ANCHOR WEIGHTING
# ---------------------------------------------------------------------

def christ_anchor_weight(text: str) -> float:
    """
    Compute a segment-level weight based on how explicitly it names Christ.

    Two tiers:
    - strong (NT explicit Christ titles)
    - typological (OT titles read Christologically)

    Overall boost is gentle and capped (typical range ~1.0–1.6).
    """
    lower = text.lower()

    strong_keywords = [
        "jesus",
        "jesus christ",
        "christ jesus",
        "lord jesus",
        "lord jesus christ",
        "the christ",
        "the christ",
        "christ",
        "son of god",
        "only begotten son",
        "son of man",
        "lamb of god",
        "the lamb",
        "the word",
        "word became flesh",
        "holy and righteous one",
        "prince of life",
        "alpha and omega",
        "the first and the last",
        "the beginning and the end",
        "bright morning star",
        "king of kings",
        "lord of lords",
        "in christ jesus",
        "in christ",
        "chief cornerstone",
        "the stone that the builders rejected",
    ]

    typological_keywords = [
        "angel of the lord",
        "the angel of the lord",
        "immanuel",
        "emmanuel",
        "wonderful counselor",
        "mighty god",
        "everlasting father",
        "prince of peace",
        "branch",
        "the branch",
        "righteous branch",
        "root of jesse",
        "shoot from the stump of jesse",
        "servant of the lord",
        "my servant david",
        "lord of hosts",
        "lord of armies",
        "shiloh",
        "holy one of israel",
    ]

    count_strong = sum(lower.count(kw) for kw in strong_keywords)
    count_typo = sum(lower.count(kw) for kw in typological_keywords)

    # Strong names: up to +0.5 (e.g., 4–5 hits saturates)
    boost_strong = min(0.5, 0.12 * count_strong)
    # Typology: gentler, up to +0.25
    boost_typo = min(0.25, 0.05 * count_typo)

    weight = 1.0 + boost_strong + boost_typo
    return min(weight, 1.6)


# ---------------------------------------------------------------------
# 3. DATUM BUILDER
# ---------------------------------------------------------------------

def process_segment(
    seg: Dict[str, str],
    tokenizer,
    max_tokens: int = 2048,
) -> types.Datum:
    """
    Turn a Bible segment into a Tinker Datum for next-token CE with
    Christ-anchor weighting.
    """
    text = seg["text"].strip()

    # Skip extremely short fragments
    word_count = len(text.split())
    if word_count < 5:
        raise ValueError(f"Segment {seg.get('id')} too short (len={word_count}).")

    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Truncate if needed to stay within context window.
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    if len(tokens) < 2:
        raise ValueError(f"Segment {seg.get('id')} too short after tokenization.")

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    anchor_w = christ_anchor_weight(text)
    weights = [anchor_w] * len(target_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": weights,
            "target_tokens": target_tokens,
        },
    )


# ---------------------------------------------------------------------
# 4. UTILS
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
# 5. CE-ONLY LOSS (Christ-weighted)
# ---------------------------------------------------------------------

def build_ce_with_anchor_loss():
    """
    Weighted cross-entropy using per-token weights (Christ anchor baked in).
    """

    def loss_fn(
        data: List[types.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = logprobs_list[0].device

        total_ce = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0.0, device=device)

        total_anchor_weight = 0.0
        count_segments = 0

        for datum, logprobs in zip(data, logprobs_list):
            target_np = _to_numpy(datum.loss_fn_inputs["target_tokens"])
            weights_np = _to_numpy(datum.loss_fn_inputs["weights"], dtype=np.float32)

            weights = torch.from_numpy(weights_np).float().to(device)

            ce = -(weights * logprobs).sum()
            total_ce += ce
            total_tokens += weights.sum()

            aw = float(weights_np[0]) if weights_np.size else 1.0
            total_anchor_weight += aw
            count_segments += 1

        norm = torch.clamp(total_tokens, min=1.0)
        ce_term = total_ce / norm

        loss = ce_term

        avg_anchor = total_anchor_weight / max(1, count_segments)
        metrics = {
            "loss_total": loss.item(),
            "loss_ce": ce_term.item(),
            "avg_anchor_weight": float(avg_anchor),
        }
        return loss, metrics

    return loss_fn


# ---------------------------------------------------------------------
# 6. TRAINING LOOP (Stage 2)
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

    # Resume from Stage 1 optimizer state
    print(f"Loading Stage 1 state from: {STAGE1_STATE_PATH}")
    training.load_state(STAGE1_STATE_PATH).result()

    loss_fn = build_ce_with_anchor_loss()

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

    print(f"Prepared {len(datums)} Datums for Stage 2 CE+Christ-anchor training.")

    # Hyperparameters (can be tuned via env)
    num_steps = int(os.getenv("NUM_STEPS", "20"))
    lr = float(os.getenv("LEARNING_RATE", "5e-5"))
    log_every = int(os.getenv("LOG_EVERY", "10"))

    print(
        f"Starting Stage 2 Bible training for up to {num_steps} steps "
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
                    f"ce={metrics['loss_ce']:.4f} | "
                    f"avg_anchor={metrics.get('avg_anchor_weight', 1.0):.3f}"
                )

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt at step {last_step}. Will still save weights...")

    finally:
        print("Saving Stage 2 weights...")
        sampler = training.save_weights_and_get_sampling_client(
            name="llama1b-bible-stage2-christ-anchor"
        )
        sampler_path = getattr(sampler, "model_path", None)
        print(f"Stage 2 weights saved to Tinker path: {sampler_path}")

        state_name = os.getenv("CHECKPOINT_NAME", "llama1b-bible-stage2-state")
        try:
            state_result = training.save_state(name=state_name).result()
            print(f"Stage 2 optimizer state checkpoint saved as: {state_result.path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: save_state failed ({exc}); continuing.")

        print("Stage 2 complete. Creating demo samples...")

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
