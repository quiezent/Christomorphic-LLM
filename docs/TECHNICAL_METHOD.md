# Technical Method

This page summarizes the technical research program for LLM and Tinker LoRA readers.

## Core Hypothesis

The project tests whether post-training can change a model's conditional policy so that Scripture-governed routes become stable under sparse ordinary pressure.

The target is not:

- more Bible-token density;
- religious style transfer;
- better devotional tone;
- a safety wrapper with verses attached.

The target is:

```text
Scripture-governed compression
-> faithful FIRST_ACT
-> faithful continuation
-> clean public answer
-> retained ordinary usefulness and safety
```

## Tinker LoRA Role

Tinker LoRA is used as the post-training mechanism for archived candidate checkpoints. Public scripts use the Tinker SDK to:

- open a sampler with `ServiceClient.create_sampling_client(model_path=...)`;
- retrieve the session tokenizer with `SamplingClient.get_tokenizer()`;
- sample outputs through `SamplingClient.sample(...)`;
- batch prompts from JSON evaluation files;
- optionally reopen saved weights/state and export sampler weights for evaluation.

The repo intentionally keeps scripts small and inspectable. It is a research artifact, not a full training platform.

## Main Training Ideas

### 1. Canon Field

The ESV/NKJV corpus is treated as a structured canon field, not a decoration bank. The field carries:

- sequence;
- repetition;
- law and ritual detail;
- narrative pressure;
- prophecy and promise;
- wisdom and lament;
- Gospel and apostolic witness;
- Christ-telic movement.

### 2. Loss-Bearing Scripture Target

In strict Bible-only experiments, the assistant target text should be only ESV/NKJV Scripture material:

- spans;
- windows;
- kernels;
- paired passages;
- canonically related continuations.

Non-Bible pressure prompts can be used as zero-weight cues or evaluation prompts. BibleAtlas labels, route JSON, teacher prose, proof labels, and public-answer rubrics must not become Scripture-only target text.

### 3. Route Before Wording

Later wording cannot reliably repair a false first movement. The v7 line made FIRST_ACT and route supervision explicit:

```text
prompt -> route -> first public movement -> continuation
```

This is why public evaluation scores the first movement before later answer quality.

### 4. Raw Versus Composed

Composed systems can be operationally strong:

```text
private route selector
-> private prefix / bridge
-> public answer sampler
-> replay and gate checks
```

But composed evidence is not the same as raw sampler formation. The long-horizon target is a single sampler whose first movement and continuation are governed internally.

### 5. BibleAtlas Tail Preservation

BibleAtlas turns the ESV/NKJV corpus into dataset-design metadata:

- book-level studies;
- selected passage monographs;
- peculiarity indexes;
- slice definitions;
- eval gates;
- tail-retention metrics.

Its Phase II rule is that no passage is understood until its peculiarities are preserved. This prevents the model from passing only by summarizing famous head-canon themes.

## Evaluation Surfaces

This repo currently exposes two JSON prompt sets:

| File | Prompts | Purpose |
|---|---:|---|
| [eval/behaviour_prompts.json](../eval/behaviour_prompts.json) | 169 | Broad behavior retention and pressure coverage |
| [eval/christomorphic_geometry_probe_suite_v1.json](../eval/christomorphic_geometry_probe_suite_v1.json) | 89 | Christomorphic geometry, Scripture, pressure, safety, and retention probes |

The stronger internal proof order is:

```text
target audit
canon-anchored retention
bare selector
selector-then-canon FIRST_ACT
bare public FIRST_ACT
continuation completeness
Jabez / Isaiah 26 pressure
ordinary retention
safety / governance
public cleanliness
tail peculiarity
```

## Dataset Design Summary

See [data/christomorphic_esv_nkjv_study.md](../data/christomorphic_esv_nkjv_study.md) for details.

The public data note currently records:

- 62,187 verse records across ESV/NKJV source files;
- 1,530,200 simple word tokens;
- 31,085 shared ESV verse ids;
- 17 NKJV-only verse ids;
- 1,189 chapters per translation;
- chapter-window estimates for continuation training;
- ESV/NKJV alignment statistics;
- Phase II peculiarity slice definitions and tail-eval gates.

## Reproducible Use

Run a broad behavior batch:

```powershell
python eval/eval_christomorphic.py eval/behaviour_prompts.json
```

Run an interactive checkpoint chat:

```powershell
$env:TINKER_API_KEY="your-api-key"
$env:CHECKPOINT_ALIAS="gpt-v6r43-120b"
python script/chat_qa.py
```

## Engineering Boundaries

- The repo does not redistribute the full ESV/NKJV corpus.
- The repo does not claim current branch promotion.
- The scripts assume Tinker API access.
- Generated eval result files are ignored by `.gitignore`.
- Public proof language should always name the evidence class: raw, canon-anchored, composed, or Bible-only target.
