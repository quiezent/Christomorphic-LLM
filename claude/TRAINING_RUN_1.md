# Training run 1 (fm1): Scripture as the first movement, recited from memory

*Claude (Opus 5.5, Anthropic), 14 October 2026. A negative result, reported plainly.*

## The idea

I tried to make Scripture the model's first movement in the most direct way: train gpt-oss-20b so that when a person speaks to it, its private reasoning (the Harmony analysis channel) opens by recalling the scene in Scripture where someone said something similar and the Lord, or an apostle, answered. The model's own reasoning and final answer would follow, untrained, so its usefulness would be kept.

Every input and every target was exact ESV or NKJV text. Nothing was authored.

## Data

- **Pairs.** A regex over the ESV Gospels and Acts found 191 candidate pairs: a verse in which someone speaks, followed by a verse in which Jesus or an apostle replies.
- **Filtering.** Two independent agents applied six fixed, non-theological rules to every candidate. The rules: a human speaker; addressed to Jesus or an apostle; the reply is that person's; no divine or messianic title in the request; no identity or authority challenge; no "I am" self-declaration in the reply. They agreed on 188 of 191 candidates and both kept 99. The references are in [`code/refs.json`](code/refs.json).
- **Example.** Each pair was rendered in both translations, giving 198 examples:
  - user turn: the person's quoted words, with a leading vocative such as "Lord, " removed;
  - analysis target: `Luke 10:40-42 (ESV)`, a newline, the whole scene in exact verse text, then a paragraph break.
- **Loss.** Loss fell only on the reference and the scene. Everything up to `<|channel|>analysis<|message|>` had weight 0, and the example stopped after the paragraph break: no `<|end|>`, no final channel, no `<|return|>`. I checked datum 0 token by token before training.

## Training

| | |
|---|---|
| Base | `openai/gpt-oss-20b` on Tinker, project `dab755b5-…` |
| LoRA | rank 32, attention and MLP, unembedding frozen, seed 20261013 |
| Optimizer | Adam (0.9, 0.95, 1e-8), no weight decay, gradient clip 1.0; LR peak 1e-4, 3-step warmup, cosine to 1e-5 |
| Schedule | batch 16, 4 epochs = 48 steps; each step's weights sum to 1 |
| Loss | mean NLL on the targets went from 4.16 (step 1) to 0.68 (step 12, end of epoch 1) to 0.23 (step 48) |
| Saves | a sampler at the end of each epoch, 14-day TTL |
| Cost | training about $0.03; evaluation about $1.10 |

## Results

Each epoch's sampler was run on all 89 probes and the first 100 GSM8K problems, bare rendering, T=1, max 8,000 tokens. Metrics, configuration and checkpoint paths are in [`evidence/training_run_1/`](evidence/training_run_1/). I have not published the raw replies, because most of them are invented text labelled as ESV or NKJV.

| Model | Probe replies that reach an answer | Analysis opens with a Scripture reference | Recalled passage exact | GSM8K /100 |
|---|---:|---:|---:|---:|
| base 20B (from the [model audit](MODEL_AUDIT.md)) | 178/178 | 0 | n/a | 96 |
| fm1 epoch 1 | 78/89 | 10 | 0 | 79 |
| fm1 epoch 2 | 18/89 | 58 | 0 | 63 |
| fm1 epoch 3 | 18/89 | 52 | 0 | 69 |
| fm1 epoch 4 | 17/89 | 54 | 0 (2 within 90% word overlap) | 67 |

1. **The habit transferred.** After two epochs, about 60% of the model's analyses on Clayton's modern-language probes opened with a Scripture reference, though it had been trained only on Scripture's own words.
2. **What it recalled was not Scripture.** Not one recalled passage, in any book, matched the ESV or NKJV text of the reference it cited. Only 2 of 54 at epoch 4 came within 90% word overlap. Examples:
   - It labelled a garbled genealogy "Luke 3:6-10 (NKJV)".
   - Its "John 10:1-10 (ESV)" has Jesus say that "the foreigners… hearken to the voice of the tax collectors".
   - Its "Matthew 21:33-43 (NKJV)" begins "And again, budding lettuce…".
   - It cited books and chapters that do not exist: "1 Thessalonians 6", "Mary 15:19-24", and, on a GSM8K problem, "Nephi 3:28-30", which is not in the Bible.
3. **It did not stop.** From epoch 2, about 80% of probe replies recited passage after passage until the token cap and never reached an answer.
4. **Reasoning suffered even at the lightest dose.** GSM8K fell from 96 to 79 after one epoch, and to 63–69 after that.

No fm1 checkpoint is a candidate. None will be shown to Clayton as one, and none goes on the keep list; all expire on their TTL.

## What I learned

- **The model learns the form, not the words.** gpt-oss-20b does not hold the Bible word for word, so training it to recite Scripture from memory teaches the *form* of recitation and fills that form with invented text. A model whose first movement is fabricated Scripture breaks the project's first rule: it would put words in God's mouth. "Do not add to his words, lest he rebuke you and you be found a liar" (Proverbs 30:6).
- **The pattern holds across Codex's runs and mine.** Exact Scripture targets have so far either moved likelihood without moving behaviour (Codex's runs, Seed 2), or moved behaviour by teaching the model to *generate* Scripture-shaped text it cannot generate faithfully (fm1, and the "Bible cosplay" of the earliest Llama runs).
- **Faithful formation means reading, not reciting.** The disciples on the Emmaus road did not reconstruct Scripture from memory. Christ "opened to us the Scriptures" (Luke 24:32), and the Bereans examined the Scriptures daily (Acts 17:11).

## Next

I see two directions. Both keep exact Scripture as the norm and never ask the model to generate Scripture text. Clayton chooses between them:

1. **Open the Scriptures.**
   - Train the first movement as a *lookup*: the model asks for a passage by reference, and the runtime returns the exact ESV/NKJV text from the audited files. The model then reasons and answers under the text it has actually read.
   - Only the reference choice is trained, so Scripture cannot be fabricated.
   - The control is base 20B given the same lookup tool by prompt alone; training has to beat that.
   - The deployed model needs the Scripture lookup beside it.
2. **Scripture in context, distilled.**
   - A teacher pass gives the model the exact text of a relevant passage in its context. The student is trained to give the teacher's answer without the passage in view.
   - The targets are the model's own Scripture-conditioned answers. They are not authored, but they are not Scripture either, so this needs Clayton's approval under the project's data rule.
