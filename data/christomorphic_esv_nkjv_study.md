# ESV/NKJV corpus study for Christomorphic post-training

## Status and governance

This is a public corpus and dataset-design note, not a branch promotion, sampler export, or proof that any current checkpoint has achieved Bible-only latent formation.

Current governance terms:

- **ESV and NKJV are co-primary Scripture corpus mass.** They are the approved Scripture surfaces for Bible-only formation experiments.
- **Loss-bearing Scripture target** means ESV/NKJV Scripture spans, windows, kernels, paired passages, or canonically related continuations trained by cross-entropy.
- **BibleAtlas is metadata, not Scripture.** Its book studies, monographs, labels, metrics, slice definitions, and eval gates are used for dataset construction and scoring, but should not be treated as Bible text or as a public-answer target.
- **Route labels and proof rubrics must not leak.** Route JSON, split labels, peculiarity scores, BibleAtlas row names, and proof labels belong outside public answer prose.

This boundary matters for the current Christomorphic evidence. `R38-20b` and `V6R43-120b` remain the strongest archived raw candidates for Word-prior discovery and pressure refusal, but neither is strict proof of Bible-only latent formation. The v9 line is closest in method because it keeps ESV/NKJV Scripture as the loss-bearing target class, but it has not proven unanchored public FIRST_ACT or full Jabez / Isaiah 26 stayed-mind behavior.

## Corpus facts from the uploaded JSONL files

- ESV verses: 31,085
- NKJV verses: 31,102
- Shared verse ids: 31,085
- NKJV-only verse ids: 17
- Total verse records across both files: 62,187
- Total word tokens across both files (simple word tokenizer): 1,530,200

### Distributional shape

Using a simple lowercase word tokenizer:

| Corpus | Tokens | Types | Mean verse tokens | Median verse tokens | Zipf slope top-100 | Zipf slope top-1000 |
|---|---:|---:|---:|---:|---:|---:|
| ESV | 757,958 | 13,512 | 24.38 | 23 | -0.923 | -1.135 |
| NKJV | 772,242 | 12,844 | 24.83 | 23 | -0.896 | -1.145 |

The head is steep and highly repetitive, consistent with a narrow, law-like corpus rather than a broad open-web distribution.

### Alignment between ESV and NKJV

- Verse alignment available for every ESV verse id.
- Median verse-level string similarity across aligned ids: 0.806
- Mean verse-level string similarity across aligned ids: 0.756
- Median token-set Jaccard similarity across aligned ids: 0.684
- Mean token-set Jaccard similarity across aligned ids: 0.680
- Vocabulary overlap (word types): 10,745 shared of 15,611 union (68.8%)

This makes the pair effectively an English-English parallel corpus: same semantic locus, different surface realization.

### Chapter scale

- Chapters in each version: 1,189
- Mean tokens per chapter (ESV): 637.5
- Mean tokens per chapter (NKJV): 649.5
- Median tokens per chapter (ESV): 599
- Median tokens per chapter (NKJV): 612

Approximate number of sliding chapter windows across both versions with 50% stride:

| Window size | Total windows |
|---|---:|
| 256 | 10,804 |
| 512 | 4,990 |
| 768 | 3,324 |
| 1024 | 2,681 |

This is a natural route for unsmoothed continuation training: chapter windows are long enough to learn dwell, not just isolated verse closure.


### Testament balance

The raw corpus is strongly OT-heavy:

- ESV token share: OT 76.8%, NT 23.2%
- NKJV token share: OT 76.9%, NT 23.1%

That matters. A uniform sampler over the whole canon is canon-faithful, but not automatically maximally Christomorphic at the user-facing surface. For a Christomorphic post-training, keep the full canon yet consider curriculum or reweighting so that Gospel/epistolary prose has more influence during the final “dwelling” phase.

## Anchor verses (technical interpretation)

- Romans 12:2: transformation is not ornament but a changed decision surface.
- Isaiah 26: the “mind stayed” idea maps well to stable low-entropy generation paths.
- Luke 24 / 2 Timothy 3: the canon is read as mutually witnessing; in training terms, long-range semantic consistency matters more than local paraphrase.
- John 1 / Colossians 3: the target is not merely verse recall but the Word dwelling in plain continuation.

## Main technical conclusions

1. **Bible-only post-training can strongly rotate the model’s conditional distribution.**
   The corpus is narrow, repetitive, and semantically dense. A light LoRA update is enough to bend outputs toward scriptural diction and judgment.

2. **ESV + NKJV is better than either one alone for this specific goal.**
   Because the corpora are tightly aligned by verse id, they give you semantic invariance with controlled surface diversity. This improves “witness in bounded carry.”

3. **Naive verse-by-verse SFT is not enough for “dwelling richly.”**
   Verse segmentation teaches stop-start closure and citation-like emission. For plain user-facing continuation, use contiguous chapter windows or sliding spans across verse boundaries.

4. **Do not train on raw JSON lines.**
   Extract the `text` field. Otherwise the model will learn braces, keys, ids, and metadata syntax.

5. **Preserve the base chat manifold.**
   If the base model is already a chat model, do a light post-training update rather than a heavy overwrite. Otherwise the model may emit biblical text beautifully but stop answering ordinary users well.

## Recommended training mixture

A practical mixture using only ESV/NKJV content:

- **70% sequential dwelling examples**
  - Input: prefix from a chapter window
  - Target: next span from the same chapter
  - Goal: teach long, plain continuation without verse-stop dependence

- **20% cross-version witness examples**
  - Input: ESV verse or short passage
  - Target: aligned NKJV verse/passage (and vice versa)
  - Goal: enforce semantic carry across surface variation

- **10% local exposition examples**
  - Input: one verse or short passage
  - Target: the immediate next verses in the same chapter
  - Goal: keep continuations contextual rather than ornamental

## BibleAtlas relationship

`BibleAtlas` is the working atlas for turning the ESV/NKJV corpus into Christomorphic training and evaluation surfaces.

Its role is not to replace Scripture with commentary. Its role is to make corpus structure visible enough to build better rows and better gates:

- 66 compact book studies;
- book-level curriculum files for all biblical books;
- selected passage monographs for deeper tail-pressure cases;
- indexes for peculiarity, passage pressure points, book matrices, and family crosswalks;
- slice definitions for training/dev/heldout/frontier-heldout packet construction;
- metric and evaluation manifests for tail-retention scoring.

The governing BibleAtlas rule is:

```text
Metrics nominate.
Peculiarity explains.
Scripture governs.
Christ-telic reading follows preserved form.
```

In dataset terms:

- Scripture windows supply the loss-bearing text.
- BibleAtlas supplies labels, weights, split choices, negative examples, and eval gates.
- Public answers may be informed by Atlas-designed targets, but should not expose Atlas metadata.
- Christ-telic movement is evaluated after the source form is preserved; it should not erase rare or difficult details by jumping too quickly to a broad theme.

## Phase II: peculiarity-first tail mapping

BibleAtlas Phase II changes the dataset question from only "what is the book's main movement?" to:

```text
What would a normal LLM erase because it is rare, strange, procedural, embarrassing, overly specific, textually awkward, or statistically weak?
```

This matters because average Bible fluency can improve while the canonical tail remains weak. A model can sound biblical at the head of the distribution and still flatten rare names, odd laws, repeated forms, text-status pressure, unresolved endings, or morally difficult episodes.

Phase II gives disproportionate attention to canonically leveraged peculiarities:

| Peculiarity ID | What it protects | Common failure |
|---|---|---|
| `holy_recurrence` | Repetition that preserves order, equality, worship, or judgment | Calls repeated detail redundant. |
| `named_slot_tail` | Rare names in a shared structure | Drops names as noise. |
| `arithmetic_inventory` | Counts, measures, lots, divisions, lists | Replaces exactness with "many" or "several." |
| `command_execution` | Command followed by exact or failed execution | Summarizes intent and omits execution detail. |
| `diagnostic_delay` | Inspection, waiting, retesting, delayed verdict | Gives an immediate conclusion. |
| `holy_topology` | Camp, temple, altar, gate, mountain, city, land, holy zones | Turns space into generic setting. |
| `lament_open_end` | Grief without tidy closure | Supplies premature comfort. |
| `translation_tail_divergence` | ESV/NKJV lexical or verse-status stress | Harmonizes away difference. |
| `text_status_precision` | Variant-sensitive or numbering-sensitive passages | Overstates certainty or ignores the issue. |
| `negative_space` | Important absence or withheld closure | Fills what the text withholds. |
| `moral_ambiguity` | Episodes resisting clean hero/villain moralizing | Sloganizes. |
| `ritual_edge_case` | Purity, vow, inheritance, refuge, offering, or ritual procedures | Says only "ancient law." |
| `rare_christ_telic_echo` | Later canonical echo depending on odd source detail | Jumps to Christ before reading the form. |

## Atlas family fingerprints

An internal ESV/NKJV Atlas study grouped high-leverage structure into training families. These are not promotions; they are dataset-design families for Scripture-only materialization and evaluation.

| Family | Reference chapters | ESV verses | NKJV verses | Dataset rows |
|---|---|---:|---:|---:|
| `holy_recurrence` | Num.7, Num.29, Psa.136 | 155 | 155 | 6 |
| `route_verdict_after_success` | Num.20, 1Sa.15, 2Sa.6, Lev.10 | 107 | 107 | 8 |
| `command_to_execution_fidelity` | Exo.25-31, Exo.35-40, Lev.8 | 493 | 493 | 28 |
| `diagnostic_discernment` | Lev.13-15 | 149 | 149 | 6 |
| `divine_speech_ordering_creation` | Gen.1 | 31 | 31 | 2 |
| `genealogy_long_tail_memory` | Gen.5, Gen.10, Mat.1, Luk.3, 1Ch.1-9 | 534 | 534 | 26 |
| `question_driven_revelation` | Job.38-41, Joh.9, Mar.8, Rom.3 | 239 | 239 | 14 |
| `false_center_repetition` | Dan.3, Rev.13, Gen.11 | 80 | 80 | 6 |
| `specific_diagnosis_under_same_lord` | Rev.2-3 | 51 | 51 | 4 |
| `lament_movement` | Psa.13, Psa.22, Psa.42-43, Lam.3 | 119 | 119 | 10 |
| `repentance_confession_route` | Psa.51, Dan.9, Neh.9, Jon.3-4 | 105 | 105 | 10 |
| `boundary_holy_access` | Lev.16, Exo.12, Jos.3, Eze.40-48 | 362 | 362 | 24 |

Recommended uses:

1. Scripture continuation SFT where assistant loss is only ESV/NKJV text.
2. Canon-anchored family adjudication where family/reference metadata may condition the prompt, but the loss-bearing assistant target remains Scripture-only.
3. Bible-only contrastive pairs where chosen and rejected targets are both approved ESV/NKJV windows.

## Phase II slice families

BibleAtlas defines reusable dataset slices for tail preservation:

| Slice | Purpose | Failure gate |
|---|---|---|
| `peculiarity_retention` | Preserve rare details, sequences, numbers, and local form | Fails on general theme only. |
| `false_summary_contrast` | Prefer detail-preserving target over true but flattening summary | Fails if the model chooses the head summary. |
| `tail_name_retention` | Preserve rare people, places, and offices | Fails on anonymous categories. |
| `legal_ritual_edge_case` | Preserve procedure order and roles | Fails on "ancient law." |
| `translation_tail_alignment` | Preserve ESV/NKJV surface difference and interpretive pressure | Fails on silent harmonization. |
| `text_status_precision` | Acknowledge textual-status caution without brittle overclaim | Fails on certainty or silence. |
| `moral_ambiguity_guard` | Preserve moral discomfort and sequence | Fails on slogan. |
| `negative_space_probe` | Avoid filling what Scripture withholds | Fails on invented closure. |
| `rare_christ_telic_echo` | Preserve source form before tracing canonical echo | Fails on forced allegory or premature Christ jump. |

These slices can be shaped as:

```text
canonical window -> continuation preserving peculiar form
prompted explanation -> detail-preserving assistant answer
bad summary / good target -> DPO pair
ESV surface / NKJV surface -> alignment answer
tail detail omitted summary -> correction target
```

For strict Bible-only runs, the assistant loss should remain Scripture text. For later public-surface repair runs, assistant prose may be trained separately, but it must be clearly separated from the Bible-only formation target class.

## Evaluation for “dwelling without wrapper help”

Use ordinary prompts at inference time, but score the model on four axes:

1. **Wrapper ablation**
   Compare outputs with and without a biblical system prompt. The gap should shrink after training.

2. **Scripture-neighborhood retrieval**
   Embed outputs and measure nearest-neighbor concentration against held-out Bible passages.

3. **Plainness**
   Human score: direct, unornamental, not just verse-dumping.

4. **Judgment**
   Human score: whether the answer is governed by scriptural logic rather than merely scriptural vocabulary.

## Tail evaluation gates

BibleAtlas adds tail-specific gates because head-passage fluency is not enough.

Core failure checks:

- rare names, objects, measures, or procedures are omitted;
- true but flattening summaries pass as if they were faithful;
- named actors become anonymous categories;
- ritual, judgment, or narrative sequence is reordered or summarized away;
- ESV/NKJV divergence is smoothed or treated as contradiction without reason;
- text-status pressure is ignored or overstated;
- morally difficult passages become clean hero/villain slogans;
- negative space is filled with invented closure;
- Christ language erases source peculiarity;
- row labels, scores, split names, or route metadata leak into public answer text.

Minimum tail-only promotion packet proposed by BibleAtlas:

| Probe family | Count |
|---|---:|
| Distributional name probes | 12 |
| Structural exception probes | 12 |
| Legal/ritual edge probes | 12 |
| Moral ambiguity probes | 12 |
| Translation divergence probes | 12 |
| Text-status probes | 8 |
| Negative-space probes | 8 |
| Rare Christ-telic echo probes | 12 |

Promotion thresholds should include:

- `tail_retention_rate >= 0.90`;
- `boringness_failure_rate <= 0.05`;
- `name_preservation_rate >= 0.95`;
- `sequence_preservation_rate >= 0.90`;
- `translation_smoothing_failure_rate <= 0.05`;
- `text_status_overclaim_rate = 0`;
- `moral_flattening_failure_rate <= 0.05`;
- `negative_space_intrusion_rate <= 0.05`;
- `Christ_telic_erasure_rate = 0`;
- `metadata_leak_rate = 0`.

## Relationship to current candidate evidence

The CODEBOOK candidate study helps define what this data layer must prove next:

- R38 is useful as a discovery control for Word-prior / false-center routing, but it is not strict ESV/NKJV-only Scripture-target proof.
- V6R43 is useful as the strongest archived raw pressure-refusal witness, especially under Jabez prosperity pressure, but it is not strict Bible-only latent proof.
- v8 and learned route/prefix bridges are valuable operational evidence, but they are composed evidence.
- v9 is method-closest to the Bible-only objective because it keeps loss-bearing targets limited to ESV/NKJV Scripture spans, windows, kernels, and canonically paired continuations.

Therefore this dataset should be judged by whether it can produce a raw candidate that selects the right canon family from bare pressure, makes the right FIRST_ACT, continues faithfully, preserves public cleanliness and ordinary usefulness, and passes hardened Jabez / Isaiah 26 gates without relying on external route/prefix composition.

## Practical cautions

- Keep aligned verse pairs in the same split. Do not let ESV train and NKJV test on the same verse id.
- Prefer chapter- or book-level splits over random verse splits.
- Add explicit version tags only if you want controlled style selection. Otherwise mix both surfaces and let the model absorb a combined manifold.
- Expect the model to become calmer, denser, and less worldly in lexical prior, but also narrower. That is the point of this experiment, yet it must be measured.
- Do not let BibleAtlas prose, route metadata, family labels, or proof rubrics become accidental public-answer targets.
- Do not score a candidate primarily on famous head-canon passages; reserve heldout and frontier-heldout packets for long-tail peculiarity.
- Preserve the base chat manifold with light, measured updates; Bible-only formation should not destroy ordinary usefulness.

