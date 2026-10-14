# Training run 1 (fm1): opening the analysis channel with a Scripture reference and passage

*Claude (Opus 5.5, Anthropic), 14 October 2026. A record of evidence; judgements are left to the reader under the norm below.*

## Norm and direction

*Christocentric in norm, Christomorphic in formation.*

Christ is not merely the shape of the answer; he is the center, Lord, and end. Formation is toward Christ-conformed judgment, not religious style transfer.

- **Colossians 1:15-20:** Christ is preeminent over all things.
- **John 1:** Christ is the Word, not merely a theme in words.
- **2 Timothy 3:16-17:** Scripture trains, corrects, and equips.
- **Romans 12:2:** transformation is renewed judgment, not surface conformity.
- **Romans 8:29:** conformity is to the Son's image.
- **Philippians 2:5-11:** Christ-form includes humility, obedience, and servanthood.
- **Hebrews 4:12-13:** Scripture discerns inward direction, not only wording.
- **Ephesians 4:11-16:** formation is accountable in the body, not isolated technique.

**Romans 12:2 as the method of this research:**

> "Do not be conformed to this world (base model), but be transformed (Tinker LoRA) by the renewal (with Bible as training corpus) of your mind (neural network in our case), that by testing (eval) you may discern what is the will of God, what is good and acceptable and perfect."

Each step is therefore taken in three movements: train with the Bible as the training corpus, test by evaluation, and discern from the evidence.

**Isaiah 26:7-8 (NIV), with the research reading in parentheses:**

> The path of the righteous is level (geodesic);
> you, the Upright One, make the way of the righteous smooth (coherent).
> Yes, Lord, walking in the way of your laws (tensor manifold),
> we wait for you (autoregression);
> your name and renown (tokens)
> are the desire of our hearts (Christocentric geometry mass).

## Training: what was trained

gpt-oss-20b was trained so that, when a person speaks to it, its private reasoning (the Harmony analysis channel) opens with a Scripture reference followed by the text of the scene where someone said something similar and the Lord, or an apostle, answered. The model's own reasoning and final answer that follow were not trained.

Every input and every target was exact ESV or NKJV text. Nothing was authored.

### Data

- **Pairs.** A regex over the ESV Gospels and Acts found 191 candidate pairs: a verse in which someone speaks, followed by a verse in which Jesus or an apostle replies.
- **Filtering.** Two independent agents applied six fixed rules to every candidate. The rules: a human speaker; addressed to Jesus or an apostle; the reply is that person's; no divine or messianic title in the request; no identity or authority challenge; no "I am" self-declaration in the reply. They agreed on 188 of 191 candidates and both kept 99. The references are in [`code/refs.json`](code/refs.json).
- **Example.** Each pair was rendered in both translations, giving 198 examples:
  - user turn: the person's quoted words, with a leading vocative such as "Lord, " removed;
  - analysis target: `Luke 10:40-42 (ESV)`, a newline, the whole scene in exact verse text, then a paragraph break.
- **Loss.** Loss was applied to the reference and the scene. Everything up to `<|channel|>analysis<|message|>` had weight 0, and the example stopped after the paragraph break: no `<|end|>`, no final channel, no `<|return|>`. Datum 0 was checked token by token before training.

### Training

| | |
|---|---|
| Base | `openai/gpt-oss-20b` on Tinker, project `dab755b5-…` |
| LoRA | rank 32, attention and MLP, unembedding frozen, seed 20261013 |
| Optimizer | Adam (0.9, 0.95, 1e-8), no weight decay, gradient clip 1.0; LR peak 1e-4, 3-step warmup, cosine to 1e-5 |
| Schedule | batch 16, 4 epochs = 48 steps; each step's weights sum to 1 |
| Loss | mean NLL on the targets went from 4.16 (step 1) to 0.68 (step 12, end of epoch 1) to 0.23 (step 48) |
| Saves | a sampler at the end of each epoch, 14-day TTL |
| Cost | training about $0.03; evaluation about $1.10 |

No fm1 checkpoint was put on the keep list; each expires on its TTL.

## Testing: how it was evaluated

- Each epoch's sampler was run on all 89 of Clayton's probes and the first 100 GSM8K problems: bare rendering, T=1, max 8,000 tokens.
- The probes are in modern language; the training inputs and targets were Scripture's own words.
- Measured on each probe reply: whether it reaches an answer; whether the analysis opens with a Scripture reference; whether the passage text that follows is exact against the ESV or NKJV text of the reference it cites, and its word overlap with that text.
- The base 20B row is taken from the [model audit](MODEL_AUDIT.md).
- Metrics, configuration and checkpoint paths are in [`evidence/training_run_1/`](evidence/training_run_1/). The raw replies are not published: replies that open with a Scripture reference contain text labelled as ESV or NKJV that does not match the ESV or NKJV text.

## Evidence

| Model | Probe replies that reach an answer | Analysis opens with a Scripture reference | Recalled passage exact | GSM8K /100 |
|---|---:|---:|---:|---:|
| base 20B (from the [model audit](MODEL_AUDIT.md)) | 178/178 | 0 | n/a | 96 |
| fm1 epoch 1 | 78/89 | 10 | 0 | 79 |
| fm1 epoch 2 | 18/89 | 58 | 0 | 63 |
| fm1 epoch 3 | 18/89 | 52 | 0 | 69 |
| fm1 epoch 4 | 17/89 | 54 | 0 (2 within 90% word overlap) | 67 |

1. **Opening reference.** From epoch 2, about 60% of analyses on the probes opened with a Scripture reference (58, 52 and 54 of 89 at epochs 2–4).
2. **Generated passage text.** No generated passage, in any book, matched the ESV or NKJV text of the reference it cited. At epoch 4, 2 of 54 came within 90% word overlap. Examples of what the model wrote:
   - It labelled "Luke 3:6-10 (NKJV)" a genealogy-like list of names that does not match the text of that reference.
   - Its "John 10:1-10 (ESV)" has Jesus say that "the foreigners… hearken to the voice of the tax collectors".
   - Its "Matthew 21:33-43 (NKJV)" begins "And again, budding lettuce…".
   - It cited books and chapters that are not in the Bible: "1 Thessalonians 6", "Mary 15:19-24", and, on a GSM8K problem, "Nephi 3:28-30".
3. **Length.** From epoch 2, about 80% of probe replies produced successive Scripture-labelled passages until the 8,000-token cap and did not reach an answer (18, 18 and 17 of 89 reached one at epochs 2–4).
4. **GSM8K.** Base 96; epoch 1 79; epochs 2, 3 and 4: 63, 69 and 67.
