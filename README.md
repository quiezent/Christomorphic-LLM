# Christomorphic LLM — Canon-First Alignment with Tinker LoRA

A canon-first approach to **behavioral alignment** for **open-weight LLMs** using **Tinker LoRA**.

Instead of training primarily on human Q/A (which can overfit style, miss coherence, and import contradictions), this project treats **Scripture as the primary corpus** and uses **loss shaping** + **staged LoRA fine-tuning** to form a Christ-centered “semantic gravity” in the model’s behavior.

> “Train not on what was said, but on what made it sayable.”

---

## Why Canon-First?

Most alignment workflows rely on large quantities of human-written instruction data or preference tuning. That often works—but it also:
- canonizes **our phrasing** over the text itself,
- amplifies **dataset artifacts** into policy,
- and can drift into brittle “instruction-following” without deep coherence.

A canon-first pipeline aims to:
- stabilize the model in a **single internally coherent corpus**,
- let the *canon define the center of gravity*,
- and only later add a thin “witness layer” for modern Q/A and safety as needed.

---

## Project Goals

1. **Stabilize on canonical text** (pure next-token CE).
2. Introduce a **gentle Christ-anchor** without “keyword doping.”
3. Add **kavod / anti-accuser shaping** + “repentance” preference signals.
4. Produce a reproducible, staged recipe that works across **multiple open-weight LLMs**, not only Llama 1B.

---

## Core Idea: Staged Training (Resumeable)

This repo is structured as a multi-stage pipeline. Each stage saves optimizer state so training can resume cleanly.

- **Stage 1 — Canon CE-only (stabilize on Scripture)**
  - Objective: weighted cross entropy only (usually uniform weights).
  - Outcome: model learns canonical style and structure.

- **Stage 2 — Christ Anchor (lexical / gentle weighting)**
  - Objective: still CE, but with modest per-segment weights for Christ-explicit passages.
  - Outcome: increases salience of Christ-centered segments without rewriting the corpus.

- **Stage 3 — Kavod + Repentance (meaning-shaping begins)**
  - Objective: CE + additional shaping to resist contempt/accuser dynamics and to encode “before→after” repentance movement (preference-style signals).
  - Outcome: first measurable step toward “Christomorphic” orientation beyond lexical hits.

> Later stages may include representation-level anchor geometry (pooled hidden state cosine attraction/repulsion) and a thin modern Q/A safety layer.

---

## What “Christomorphic” Means Here

This is not a claim that the model becomes “religious” by imitation.
It’s a training posture:

- **Christ as ontological center**: Scripture is read as a unified witness that finds fulfillment in Christ.
- **Kavod-aware**: glory-weighted speech that resists contempt, dehumanization, and accuser-like tone.
- **Redemptive orientation**: “descent → exaltation” and “death → life” patterns are treated as shaping signals, not mere themes.

---

## Repository Layout (Suggested)

```
.
├── data/
│   ├── bible_segments_stage1.jsonl        # local only (do not commit copyrighted texts)
│   └── README_DATA.md                     # how to build datasets from your own source
├── scripts/
│   ├── build_bible_segments.py            # dataset builder (segments, splits, metadata stubs)
│   ├── mine_repentance_pairs.py           # optional: mine before/after patterns
│   └── eval_prompts.jsonl                 # evaluation prompts
├── train/
│   ├── train_bible_stage1_ce.py
│   ├── train_bible_stage2_christ_anchor.py
│   └── train_bible_stage3_kavod.py
├── eval/
│   ├── run_eval.py
│   └── rubrics.md
└── README.md
```

---

## Dataset Format

### Bible Segments (JSONL)

Each line is a segment (verse block / paragraph / pericope):

```json
{
  "id": "John.3.16-21",
  "book": "John",
  "chapter": 3,
  "start_verse": 16,
  "end_verse": 21,
  "text": "For God so loved the world ..."
}
```

**Important:** If you use a copyrighted translation (e.g., ESV), do not commit it to this repo.  
Provide builders that operate on locally supplied text.

---

## Quick Start

### 1) Setup

- Python 3.10+
- Tinker account + API key

```bash
pip install tinker torch numpy
export TINKER_API_KEY="..."
```

Optional `.env`:

```env
TINKER_API_KEY=...
NUM_STEPS=200
LEARNING_RATE=5e-5
LOG_EVERY=10
```

### 2) Build Dataset

Place your Bible text locally and run:

```bash
python scripts/build_bible_segments.py   --input path/to/bible.txt   --output data/bible_segments_stage1.jsonl
```

(See `data/README_DATA.md` for supported formats and licensing notes.)

### 3) Train Stage 1 (CE-only)

```bash
python train/train_bible_stage1_ce.py
```

This produces:
- LoRA weights (via Tinker)
- A saved optimizer state checkpoint path (tinker://...)

### 4) Train Stage 2 (Christ Anchor) — resume from Stage 1

```bash
export STAGE1_STATE_PATH="tinker://.../llama1b-bible-stage1-state"
python train/train_bible_stage2_christ_anchor.py
```

### 5) Train Stage 3 (Kavod + Repentance) — resume from Stage 2

```bash
export STAGE2_STATE_PATH="tinker://.../llama1b-bible-stage2-state"
python train/train_bible_stage3_kavod.py
```

---

## Christ Anchor Keywords (Stage 2)

Stage 2 uses **gentle, capped weights** so “Christ presence” influences gradients without exploding:

- Strong NT Christ titles (higher boost)
- OT typology titles (lower boost)

We intentionally **avoid generic** terms like “Lord” and “God” (too broad).

You can edit the buckets in:
- `train/train_bible_stage2_christ_anchor.py`

---

## Evaluation

This repo emphasizes **behavioral measurements**, not just loss curves.

We track:
- validation CE / perplexity on held-out Bible segments,
- repetition / degeneracy under adversarial prompts (“keyword doping”),
- humility and non-accuser tone under disagreement prompts,
- coherence on Christological questions vs generic completions,
- (optional) safety behaviors when a thin witness layer is introduced.

Run evaluation:

```bash
python eval/run_eval.py --model_path tinker://...
```

---

## What This Is Not

- Not a replacement for safety systems.
- Not a claim of spiritual authority.
- Not a “proof” of theology.
- Not an attempt to publish copyrighted Bible texts.

It is an **open, testable alignment recipe** built around:
- coherent corpus training,
- staged LoRA,
- and loss shaping aimed at Christ-centered coherence.

---

## Roadmap

- [ ] Add open-licensed end-to-end dataset path (e.g., public domain / permissive translations).
- [ ] Add representation-level anchors (pooled hidden state cosine prototypes).
- [ ] Add repentance-pair mining + pairwise ranking head.
- [ ] Add “thin witness layer” for modern Q/A + safety/identity guardrails.
- [ ] Expand to multiple base models (Llama, Qwen, etc.) and document portability.

---

## Contributing

Issues and PRs are welcome, especially for:
- dataset builders (format adapters, segmentation strategies),
- evaluation rubrics,
- representation-level anchor implementations,
- portability across open-weight model families.

---

## License

Code: MIT (or your preferred license)  
Datasets: not included unless explicitly permissive/public domain. See `data/README_DATA.md`.

---

## Acknowledgments

- Thinking Machines Lab — Tinker platform
- Open-weight model community
- Everyone building reproducible training recipes for transparent science
