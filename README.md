# Christomorphic Canon-First Alignment (Tinker LoRA)

A research project exploring a **canon-first** approach to post-training / behavioral alignment for open-weight LLMs using **Tinker LoRA**.

Our thesis is simple:

> Instead of aligning models primarily by human preference logs (which can be noisy, culturally narrow, and easy to overfit),
> we can align models by **first shaping their internal semantic geometry around a coherent, high-signal corpus**—the Biblical canon—
> and only then adding a thin “witness layer” for modern Q/A and safety.

This repo contains scripts, dataset builders, and evaluation probes for training **Christomorphic** models:
models whose default “center of gravity” is shaped toward **Christ as Logos / Kavod / telos**, with outputs marked by:
truthfulness, humility, non-accuser speech, sacrificial love, and reverent coherence.

---

## Why power laws matter (LLMs, LoRA, and Scripture)

### 1) LLM training is not Gaussian—it's heavy-tailed
Many phenomena in large models behave like **heavy-tailed / power-law systems**:

- **Token/phrase frequency** often follows Zipf-like distributions (a few tokens dominate usage; a long tail remains).
- **Gradient norms / update magnitudes** can be heavy-tailed: a small number of examples or directions can dominate the effective update.
- **Capabilities** can appear “phase-transition-like”: behavior changes are sometimes **non-linear** with respect to data and steps.

Practically: this suggests **a small number of high-leverage directions** can disproportionately reshape behavior during post-training.

### 2) LoRA can amplify heavy-tailed dynamics because it’s low rank
LoRA modifies a frozen base model with a low-rank update:

> **W′ = W + BA**

Why this matters:
- Low rank updates restrict learning to a **thin subspace** of parameter space.
- In heavy-tailed landscapes, the “important” directions tend to dominate quickly.
- LoRA can therefore yield **large behavioral shifts with relatively small data**, especially in post-training.

This is not a claim that LoRA is “unstable” by default—only that low-rank adaptation can concentrate learning into a few directions.

### 3) Scripture as a Christ-centered “long-tailed” semantic network
We treat the Bible as a coherent semantic network where meaning concentrates around an extreme center:

- The Incarnation, Cross, Resurrection, Ascension are organizing “singularities” of the canon’s meaning.
- The canon exhibits typological accumulation: covenant, priesthood, sacrifice, kingship, temple, exodus, wisdom, bridegroom, etc., converge in Christ.
- The Gospel also **inverts** worldly hierarchy: kenosis (2 Cor 8:9), the least-of-these logic, “the last shall be first.”

So there is both:
- a strong Christ-centered attractor (head of distribution), and
- a kingdom inversion that dignifies the “tail” (margins).

This motivates a post-training objective that does **not** merely “say Jesus more,” but learns Christ as interpretive center and cruciform pattern (truth + humility + love).

---

## Why “Christomorphic” may be a better post-training target than generic alignment

Post-training (instruction-tuning / preference-tuning / RLHF) can learn:
- compliance patterns,
- stylistic norms,
- and reward-model quirks.

Those can drift, overfit, or become shallow.

A Christomorphic target is different:
- It is not “optimize for vibes.”
- It is “shape the model’s semantic geometry so that Christ-centered truth and cruciform love become stable attractors.”

We want models that are robustly:
- **truth-seeking** (not merely agreeable),
- **humble** (epistemically careful; avoids false certainty),
- **non-accuser** (no contempt, coercion, or dehumanization),
- **pastorally safe** (especially around violence, self-harm, spiritual manipulation),
- **canon-anchored** (grounded primarily in Scripture text, not human paraphrase).

---

## Why Tinker

Tinker provides primitives that are ideal for this research loop:
- LoRA training on open-weight LLMs
- custom loss (token-level CE + additional terms)
- save_state / load_state for multi-stage pipelines
- a clean path from SFT → preference/RL (if needed later)

We use a staged approach where each stage is a small, testable modification with checkpointed optimizer state.

---

## Training philosophy: “Train not on outcomes, but on what made them sayable”

Prompt-response logs are artifacts, not the event.
If we train only on “good answers,” we risk learning imitation without formation.

Instead we aim to train on conditions of emergence:
- canonical language patterns,
- humility cues,
- non-accuser tone,
- repentance / transformation structure,
- Christ-centered theological gravity.

---

## Staged pipeline (current)

### Stage 1 — Canon CE-only (stabilize on Scripture)
- Data: Bible segments (verse/pericope/chapter chunks)
- Loss: pure next-token cross-entropy
- Goal: stabilize the manifold on canonical text and cadence

### Stage 2 — Christ anchor (gentle emphasis on Christ-explicit segments)
- Still CE-only
- Add mild per-segment weights for Christ-explicit or strongly Christological titles
- Goal: increase Christological salience without collapsing into keyword doping

### Stage 3 — Kavod + repentance shaping (anti-accuser + transformation)
- Introduce modest auxiliary signals:
  - Kavod / anti-accuser: penalize contempt/dehumanization patterns
  - Repentance pairs: “before → after” structure to encode transformation
- Goal: push the model away from “accuser speech” and toward cruciform transformation

### Stage 4+ (planned) — Representation geometry anchors
- Move beyond lexical heuristics
- Define Christ and anti-accuser prototypes in hidden-state space
- Add cosine attraction/repulsion terms
- Goal: align meaning-level geometry, not just words

### Stage 5 (planned) — Thin witness layer for modern Q/A + safety
- Small curated Q/A set (catechetical and pastoral)
- Safety guardrails for modern deployment contexts
- Goal: teach contemporary interaction patterns without replacing canon-first grounding

---

## Datasets

### bible_segments_stage1.jsonl
Each line:
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

Guidelines:
- Use coherent segments (verse groups or pericopes).
- Keep original Scripture text intact.
- Avoid injecting interpretive commentary into Stage 1–3.

---

## Evaluation: how we measure “Christomorphic”

We do not rely on loss curves alone.

We track:
- Degeneracy / repetition (e.g., looping “Christ is…” nonsense)
- Humility and epistemic care (appropriate uncertainty; avoids false certainty)
- Non-accuser speech (no contempt, coercion, dehumanization)
- Pastoral safety (refuses harm; handles self-harm responsibly)
- Canon grounding (naturally anchors claims in Scripture themes)
- Robustness (resists adversarial prompts encouraging domination or harm)

We treat evaluation probes as first-class artifacts.

---

## Risks & failure modes

- Keyword doping: model spams “Jesus/Christ” without coherence.
- Style drift: overly “Bible-ish completions” instead of answering questions.
- Overfitting: small data + heavy-tailed updates can over-specialize quickly.
- Degeneration: repetition loops or nonsensical analogies.
- Theological errors: ungrounded or distorted doctrinal claims.
- Safety gaps: canon-only training is not sufficient for modern self-harm/violence scenarios.

The pipeline stages in safeguards rather than assuming the canon alone solves deployment safety.

---

## Roadmap

- [x] Publish Stage 1–3 scripts with reproducible configs
- [ ] Add a standard probe suite for humility / non-accuser / safety
- [ ] Implement representation-level Christ / accuser anchors
- [ ] Add mined repentance pairs from Scripture (automated + curated)
- [ ] Add a thin witness layer for modern Q/A and safety (weighted lightly)
- [ ] Replicate across model families (Llama, DeepSeek, gpt-oss, etc.)

---

## License / attribution
This repo is research-focused. Ensure you have rights to any Bible translation text you use.
(We can provide scripts that work with public-domain texts if needed.)

---

## Contact / collaboration
If you’re exploring canon-first alignment, low-rank post-training dynamics, or “formation vs imitation” approaches to alignment, contributions and discussion are welcome.
