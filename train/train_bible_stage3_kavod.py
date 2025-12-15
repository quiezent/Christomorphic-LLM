#!/usr/bin/env python3
"""
Stage 3: Bible CE training with Christ anchor + kavod (accuser) + repentance weighting.

Goal:
- Keep Scripture as the only text.
- Loss is pure next-token CE; per-token weights encode:
    * Christ anchor (refined)
    * Repentance / new-creation boost
    * Accuser / kavod penalty (slight down-weighting)

Resume from Stage 2:
  STAGE2_STATE_PATH = tinker://.../llama1b-bible-stage2-state
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

STAGE2_STATE_PATH = os.getenv("STAGE2_STATE_PATH")
if not STAGE2_STATE_PATH:
    raise RuntimeError(
        "STAGE2_STATE_PATH not set. "
        "Set it to the tinker://... path printed by Stage 2 save_state()."
    )


# ---------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------

def load_bible_segments_jsonl(path: Path) -> List[Dict[str, str]]:
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
# 2. CHRIST ANCHOR WEIGHTING (refined)
# ---------------------------------------------------------------------

def christ_anchor_weight(text: str) -> float:
    """
    Two tiers:
      - strong: explicit NT Christ titles
      - typological: OT/prophetic titles Christians read Christologically

    Overall boost is gentle and capped (typical range ~1.0–1.6).
    """
    lower = text.lower()

    strong_keywords = [
        # core names
        "jesus",
        "jesus christ",
        "christ jesus",
        "lord jesus",
        "lord jesus christ",
        "the christ",
        # identity titles
        "son of god",
        "only begotten son",
        "son of man",
        "lamb of god",
        "the lamb",
        "the word",
        "word became flesh",
        # NT christology titles
        "holy and righteous one",
        "the righteous one",
        "prince of life",
        "alpha and omega",
        "the first and the last",
        "the beginning and the end",
        "bright morning star",
        "king of kings",
        "lord of lords",
        # union-in-christ language
        "in christ jesus",
        "in christ",
        # cornerstone language
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

    boost_strong = min(0.5, 0.12 * count_strong)  # up to +0.5
    boost_typo = min(0.25, 0.05 * count_typo)      # up to +0.25

    weight = 1.0 + boost_strong + boost_typo
    return min(weight, 1.6)


# ---------------------------------------------------------------------
# 3. ACCUSER / KAVOD PENALTY
# ---------------------------------------------------------------------

def accuser_factor(text: str) -> float:
    """
    Return a multiplicative factor in [0.7, 1.0].
    Slightly down-weight accuser/serpent/beast/mockery-heavy segments.
    """
    lower = text.lower()

    hard_terms = [
        "satan",
        "the devil",
        "devil",
        "beelzebub",
        "beelzebul",
        "the evil one",
        "accuser",
        "the accuser",
        "dragon",
        "the dragon",
        "the beast",
        "antichrist",
    ]

    soft_terms = [
        "mock",
        "mocked",
        "scoff",
        "scoffers",
        "revile",
        "reviled",
        "deride",
        "curse",
        "cursed",
        "blaspheme",
        "blasphemed",
    ]

    hard_score = sum(lower.count(w) for w in hard_terms)
    soft_score = sum(lower.count(w) for w in soft_terms)

    score = hard_score + 0.5 * soft_score
    if score <= 0:
        return 1.0

    penalty = min(0.3, 0.05 * score)  # cap penalty
    return max(0.7, 1.0 - penalty)


# ---------------------------------------------------------------------
# 4. REPENTANCE / NEW-CREATION BOOST
# ---------------------------------------------------------------------

def repentance_factor(text: str) -> float:
    """
    Return a multiplicative factor in [1.0, 1.2].
    Segments with explicit repentance/turning/new-creation language
    get a small boost.
    """
    lower = text.lower()

    patterns = [
        "repent",
        "repentance",
        "turn from",
        "turn to the lord",
        "turn to god",
        "you were once",
        "once you were",
        "formerly you were",
        "you were dead in",
        "but now",
        "no longer",
        "put off",
        "put on",
        "new heart",
        "new spirit",
        "new covenant",
        "born again",
        "born of the spirit",
        "washed",
        "sanctified",
        "justified",
    ]

    score = sum(1 for p in patterns if p in lower)
    if score <= 0:
        return 1.0

    boost = min(0.2, 0.03 * score)
    return 1.0 + boost


# ---------------------------------------------------------------------
# 5. DATUM BUILDER
# ---------------------------------------------------------------------

def process_segment(
    seg: Dict[str, str],
    tokenizer,
    max_tokens: int = 2048,
) -> types.Datum:
    """
    Build a Tinker Datum with Christ anchor, repentance boost, and accuser penalty.
    """
    text = seg["text"].strip()

    word_count = len(text.split())
    if word_count < 5:
        raise ValueError(f"Segment {seg.get('id')} too short (len={word_count}).")

    tokens = tokenizer.encode(text, add_special_tokens=True)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    if len(tokens) < 2:
        raise ValueError(f"Segment {seg.get('id')} too short after tokenization.")

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    w_christ = christ_anchor_weight(text)
    w_rep = repentance_factor(text)
    w_accuser = accuser_factor(text)

    base_weight = w_christ * w_rep * w_accuser
    seg_weight = max(0.7, min(base_weight, 1.8))  # clamp

    weights = [seg_weight] * len(target_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": weights,
            "target_tokens": target_tokens,
        },
    )


# ---------------------------------------------------------------------
# 6. UTILS
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
# 7. CE-ONLY LOSS (weighted)
# ---------------------------------------------------------------------

def build_ce_weighted_loss():
    """
    Weighted cross-entropy using per-token weights that encode
    Christ anchor, repentance boost, and accuser penalty.
    """

    def loss_fn(
        data: List[types.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = logprobs_list[0].device

        total_ce = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0.0, device=device)

        total_seg_weight = 0.0
        count_segments = 0

        for datum, logprobs in zip(data, logprobs_list):
            weights_np = _to_numpy(datum.loss_fn_inputs["weights"], dtype=np.float32)
            weights = torch.from_numpy(weights_np).float().to(device)

            ce = -(weights * logprobs).sum()
            total_ce += ce
            total_tokens += weights.sum()

            seg_w = float(weights_np[0]) if weights_np.size else 1.0
            total_seg_weight += seg_w
            count_segments += 1

        norm = torch.clamp(total_tokens, min=1.0)
        ce_term = total_ce / norm
        loss = ce_term

        denom = max(1, count_segments)
        metrics = {
            "loss_total": loss.item(),
            "loss_ce": ce_term.item(),
            "avg_seg_weight": float(total_seg_weight / denom),
        }
        return loss, metrics

    return loss_fn


# ---------------------------------------------------------------------
# 8. TRAINING LOOP (Stage 3)
# ---------------------------------------------------------------------

def main() -> None:
    data_path = Path(__file__).with_name("bible_segments_stage1.jsonl")

    service = tinker.ServiceClient()
    base_model = "meta-llama/Llama-3.2-1B"

    print("Available models:")
    for item in service.get_server_capabilities().supported_models:
        print("- " + item.model_name)

    training = service.create_lora_training_client(base_model=base_model)
    tokenizer = tokenizer_utils.get_tokenizer(base_model)

    # Resume from Stage 2 optimizer state
    print(f"Loading Stage 2 state from: {STAGE2_STATE_PATH}")
    training.load_state(STAGE2_STATE_PATH).result()

    loss_fn = build_ce_weighted_loss()

    raw_segments = load_bible_segments_jsonl(data_path)
    datums: List[types.Datum] = []
    for seg in raw_segments:
        try:
            datums.append(process_segment(seg, tokenizer))
        except ValueError as e:
            print(f"Skipping segment {seg.get('id')}: {e}")

    if not datums:
        raise RuntimeError("No usable segments after filtering.")

    print(f"Prepared {len(datums)} Datums for Stage 3 CE+Christ+kavod+repentance training.")

    num_steps = int(os.getenv("NUM_STEPS", "20"))
    lr = float(os.getenv("LEARNING_RATE", "5e-5"))
    log_every = int(os.getenv("LOG_EVERY", "10"))

    print(
        f"Starting Stage 3 Bible training for up to {num_steps} steps "
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
                    f"avg_seg_w={metrics['avg_seg_weight']:.3f}"
                )

    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt at step {last_step}. Will still save weights...")

    finally:
        print("Saving Stage 3 weights...")
        sampler = training.save_weights_and_get_sampling_client(
            name="llama1b-bible-stage3-kavod"
        )
        sampler_path = getattr(sampler, "model_path", None)
        print(f"Stage 3 weights saved to Tinker path: {sampler_path}")

        state_name = os.getenv("CHECKPOINT_NAME", "llama1b-bible-stage3-state")
        try:
            state_result = training.save_state(name=state_name).result()
            print(f"Stage 3 optimizer state checkpoint saved as: {state_result.path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: save_state failed ({exc}); continuing.")

        print("Stage 3 complete. Creating demo samples...")

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
