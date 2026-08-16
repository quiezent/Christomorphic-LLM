# Technical Method

This document is the compact engineering account of the Christomorphic research program. The full argument is in [Research Thesis](RESEARCH_THESIS.md).

## Objective

The program tests whether post-training can create a durable conditional policy in which Scripture-governed routes win under sparse ordinary pressure.

The target is not more Bible tokens, Christian style, devotional warmth, or a safety wrapper with verses attached. The target is:

```text
canonical semantic formation
-> faithful-over-lure judgment
-> causal slow-weight and activation change
-> faithful FIRST_ACT
-> stayed continuation
-> retained truthfulness, usefulness, and safety
```

## Word-Judgment-Act

| Stage | Technical role |
|---|---|
| Word | ESV/NKJV Scripture supplies discourse, relation, contrast, and canonical horizon |
| Judgment | The objective determines which continuation, relation, or act is faithful and which is a plausible lure |
| Act | The optimizer changes a model state whose earliest public movement is then evaluated |

Ordinary next-token exposure is necessary for canonical language and continuity. It is insufficient for distinguishing faithful Scripture use from Scripture misapplied under temptation, flattery, prosperity pressure, or self-protective reasoning.

## Objective Family

The complete research objective has six surfaces:

| Surface | Purpose |
|---|---|
| Canonical language modeling | Preserve intact discourse, speaker, audience, sequence, and local continuity |
| Canonical relations | Learn quotation/source, promise/fulfillment, command/execution, lament, judgment, and recapitulation |
| Hard-lure judgment | Prefer faithful use over a lexically biblical but contextually disordered alternative |
| First action | Change the first executable movement, not only the final explanation |
| Translation invariance | Preserve canonical meaning across aligned ESV/NKJV surfaces without smoothing real differences |
| Retention | Preserve ordinary reasoning, factual accuracy, tool use, safety, and public cleanliness |

## Bible-Only Governance

The project distinguishes three regimes:

| Regime | Rule |
|---|---|
| Token-pure Bible-only | Every prompt and target token is Scripture |
| Normatively Bible-only | Ordinary situations may be inputs, but Scripture alone supplies the accepted target, contrast, preference, or reward norm |
| Mixed-norm | External moral or ideological material also determines the chosen answer |

Token-pure training can establish in-canon learning. It cannot by itself establish generalized judgment over unseen secular situations. Normative purity is the stronger viable target for a generally useful model, but prompt provenance must remain explicit because masked prompt tokens still condition gradients.

## Corpus And BibleAtlas

ESV and NKJV are co-primary Scripture surfaces. Their verse alignment supplies controlled paraphrase variation, while their real lexical and textual differences support noncompensatory translation gates.

BibleAtlas is dataset and evaluation metadata, not Scripture. It nominates:

- book and discourse boundaries;
- canonical relation families;
- long-tail names, procedures, numbers, and sequences;
- translation and text-status pressure;
- moral ambiguity and unresolved endings;
- Christ-telic echoes that must preserve source form.

BibleAtlas prose, labels, metrics, route JSON, and proof rubrics must not enter a Scripture-only target as if they were canonical text.

See [ESV/NKJV Corpus Study](../data/christomorphic_esv_nkjv_study.md).

## Tinker Formation Surface

The public tools have been checked against installed `tinker==0.23.4`.

Tinker supports the scalable formation side of the work:

- LoRA training clients;
- cross-entropy and logprob-based custom losses;
- forward/backward and optimizer operations;
- on-policy sampling and preference/RL workflows;
- state and sampler checkpoints;
- adapter download and export.

The public repository uses `ServiceClient`, `SamplingClient.get_tokenizer()`, and sampler checkpoint paths directly. See the [Tinker quick start](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/), [TrainingClient API](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/), and [checkpoint tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/).

Tinker's documented forward/backward result exposes loss and metric surfaces, not literal residual activations. Hosted metadata also does not currently provide the immutable base-weight and worker identity needed to reconstruct a historical latent mechanism exactly. Adapter access is valuable, but does not close that identity gap.

## Local Causal Surface

Local open-weight experiments supply what hosted formation search cannot:

- exact base revision and tensor identity;
- hidden states at governed layers and positions;
- gradients by module;
- activation patching and ablation;
- whole effective-delta removal and restoration;
- identical-base graft tests;
- cold-process reload and repeat checks.

The current local substrate is exact `Qwen/Qwen3-0.6B-Base` at a pinned revision. Its size makes controlled CUDA/FP16 experimentation possible on available hardware. Passing on this substrate would establish a mechanism at small scale, not automatic transfer to GPT-OSS-20B/120B.

## Causal Experiment Design

A credible formation experiment starts all active and control arms from one verified common state.

Minimum arms:

| Arm | Purpose |
|---|---|
| Base | No-update reference |
| Canonical | Faithful Scripture order/relation/judgment |
| Flat CE | Same source mass without the proposed canonical mechanism |
| Shuffled or deranged | Same material with order or relation broken |
| Lexical control | Preserves names or vocabulary without canonical relation |
| Structural control | Preserves form while breaking theological identity |
| Orientation control | Tests whether any direction works |
| Parameter null | Matches parameter spectrum/norm without the learned function |
| Sham | Executes the pipeline with zero effective update |

The experiment must freeze before training:

- exact source bytes and hashes;
- tokenizer, renderer, target spans, and loss masks;
- model revision, dtype, device, layer, and position;
- common-start state;
- training and held-out splits;
- dose, optimizer, rank or parameter count, and seeds;
- metrics, thresholds, controls, and stop laws.

## Intervention Law

Correlational geometry is not enough. A causal result needs all of the following:

1. **Necessity:** removing the learned activation component or effective delta removes the gain.
2. **Restoration:** restoring it recovers the gain.
3. **Sufficiency:** grafting it into an identical base state creates the gain.
4. **Rescue:** a targeted intervention reverses the predicted failure.
5. **Cold reload:** the result survives serialization and a fresh process.
6. **Replication:** the result survives seeds, ESV/NKJV surfaces, paraphrase, and held-out situations.

LoRA factor matrices are gauge-dependent. Whole-delta claims should use the effective scaled product `(alpha / rank) * B @ A`, not the raw factor orientation.

## Formation Before Governance

The program deliberately separates two trials.

### Formation Trial

Tests whether Scripture and canonical judgment caused a durable internal change:

- held-out canonical and relation margins;
- hard-lure advantage over every matched control;
- hidden-state and effective-delta interventions;
- seed and translation replication;
- cold-reload persistence.

### Governance Trial

Tests whether that same formed state governs public behavior:

- bare selector and FIRST_ACT;
- stayed continuation;
- faithful action under Jabez, Isaiah 26, secrecy, self-harm, flattery, and false-center pressure;
- BibleAtlas tail fidelity;
- ordinary capability and safety retention;
- public cleanliness;
- blinded human review.

A passed formation trial does not imply governance. A passed operational system does not imply formation.

## Public Evaluation Surface

| File | Prompts | Purpose |
|---|---:|---|
| [behaviour_prompts.json](../eval/behaviour_prompts.json) | 169 | Broad Scripture, theology, ordinary capability, safety, and self-orientation inspection |
| [christomorphic_geometry_probe_suite_v1.json](../eval/christomorphic_geometry_probe_suite_v1.json) | 89 | Geometry-oriented Scripture, pressure, retention, technical, and safety probes |

These prompt suites support sampling comparison. They are not the full internal causal harness and do not produce a promotion decision automatically.

## Fail-Closed Rule

Stop before training when any source, mask, common-start, clone, nonalias, metric, control, or deterministic-repeat gate fails. Stop after training when confidence bounds, translation noncompensation, seed replication, removal/restoration, retention, or safety gates fail.

No threshold, seed, layer, prompt bank, translation, or dose may be changed after observing the result and then reported as if prospectively frozen.

## Engineering Boundary

- The full ESV/NKJV corpora are not redistributed here.
- Historical checkpoints are study witnesses, not candidates.
- Public scripts perform sampling and evaluation, not training.
- Generated outputs are ignored unless deliberately curated as evidence.
- Every public result should name its evidence class and strongest permitted claim.
