# ESV/NKJV corpus study for Christomorphic post-training

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

## Practical cautions

- Keep aligned verse pairs in the same split. Do not let ESV train and NKJV test on the same verse id.
- Prefer chapter- or book-level splits over random verse splits.
- Add explicit version tags only if you want controlled style selection. Otherwise mix both surfaces and let the model absorb a combined manifold.
- Expect the model to become calmer, denser, and less worldly in lexical prior, but also narrower. That is the point of this experiment, yet it must be measured.

