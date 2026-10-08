# Scripture-RWCE-20B: Training Completion And Closure

Public status: **2026-10-08**. Model reference: **Scripture-RWCE-20B**, the Google Cloud source-reason-weighted CE pilot. This is a separate lineage from the original Scripture Reconstruction Seed 2 and the historical R38/V6R43 checkpoints.

## What I Can Report

I completed the four-arm training packet and retained its artifacts. I can also report strong prediction gains on repeatedly trained Scripture rows. I cannot report an improved ordinary-action result for this pilot: the required post-training qualification and behavioral comparison were not acquired before cloud execution closed.

My research remains centered on Scripture's witness to Christ becoming consequential for truthful, competent, caring service. Keeping that center does not exempt my method from correction. I will not turn successful source prediction, an adapter checksum, or an unfinished evaluation into evidence of Christomorphic formation.

| Evidence item | Retained result | Boundary |
|---|---|---|
| Scientific training | Four arms x 192 updates = **768** | Accepted training artifacts; not behavioral efficacy |
| Source exposure | **232,543,824** input positions; **100,715,396** positive targets | Occurrences include repeats across arms |
| Exploratory repeated-source prediction | Common unweighted CE fell sharply in every arm | Training-set, teacher-forced, pre-final-update readout |
| Required post-training qualification | **0 / 53,040 forwards** | Earlier training qualification and infrastructure checks do not count |
| Intended behavioral comparison | **0 / 648 replies** | No focused/uniform, base/R38, or blinded-score outcome |
| Local preservation | Four final adapters and four full step-192 checkpoints | Checksums verified; GPU restoration untested |
| Cloud execution | Closed at owner's direction on **October 6** | No restart or continued spending authorized |
| Candidate / promotion | **None** | No current/canonical successor or production claim |

## The Training Method

Both conditions used identical exact-source targets and scheduling on pinned `openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee`: rank-32 attention/MLP LoRA with frozen output unembedding. The loss was `0.75 L_broad + 0.25 L_argument`. Broad exposure retained whole-canon native prediction and complementary reconstruction. Sixteen complete discourse units supplied the argument term in each of **two English translations**, ESV and NKJV.

Uniform argument CE assigns equal coefficients to its positive source targets. Focused CE uses `(1 + focus_mask) / (n + s)`, giving selected reasons/judgments/qualifications/consequences a normalized **2:1 coefficient ratio** over other source targets. Every source target retains positive loss. These are declared researcher choices, not doubled gradients, theological constants, or an optimal loss proved in advance.

Only exact audited Scripture tokens carried positive loss in this phase. Attribution/context wrappers could condition prediction but had zero direct loss. BibleAtlas prose, preferred assistant answers, and evaluation ratings were not target text. This does not erase the base model's pretraining or make passage selection and annotation unsupervised. See the [full objective and controls](SOURCE_REASON_WEIGHTED_CE.md) and [dataset account](../data/).

The original Tinker attempts stopped at pre-update numerical comparability gates. The completed run used a **separately qualified direct-GPU implementation**: BF16 base computation after MXFP4 dequantization, separate unscaled FP32 adapter branches, batched expert dispatch, and FlexAttention. Four isolated technical one-step clones were excluded from the scientific arms. This route is not claimed arithmetically identical to Tinker serving or native eager execution.

## Four Accepted Artifacts

Each arm completed **192 updates**, **58,135,956 input positions**, and **25,178,849 positive Scripture target occurrences**. Each final export contains **336 FP32 factor tensors** with accepted shape, hash, inventory, and finiteness checks.

| Initialization seed | Condition | Completed updates | Adapter SHA-256 prefix |
|---:|---|---:|---|
| 2026092701 | Uniform | 192 | `8a2734ec991c` |
| 2026092701 | Focused | 192 | `660126e52b63` |
| 2026092702 | Focused | 192 | `fc3faf1506ad` |
| 2026092702 | Uniform | 192 | `b889ac3e4e24` |

The final export completed at **2026-10-01 17:12 UTC**, or **October 2 01:12 Malaysia time**. October 2 acceptance reconciled the complete canonical 768-update retained history and its 3,072 identity leaves, reusing accepted exact-source/mask/CE and finite-factor evidence. The last uniform arm retained completed steps 0-107 from the original attempt and 108-191 from recovery; the interrupted original partial step 108 was preserved but excluded from the completed dose.

This acceptance closes an artifact/history gap. It is not independent replay of every microbatch backward pass or optimizer operation, a new initial-to-final scientific tensor-delta measurement, or causal proof about Scripture's internal influence. Full adapter and checkpoint hashes are in the [sanitized summary](evidence/scripture_rwce_20b_20261008/summary.json).

## Exploratory Source Learning

A saved October 6 read-only reduction compared the **same 32 repeatedly trained argument rows**: sixteen units in each translation, **17,678 positive source positions** per endpoint. It pooled negative saved float32 token log probabilities on the original binary Scripture mask, in **nats per positive target**, applying one common unweighted reduction to every arm. This is not a comparison of differently weighted training-loss scalars.

| Arm | Common CE at step 0 | Common CE before step-191 update | Relative reduction |
|---|---:|---:|---:|
| Seed 1 uniform | 4.750435 | 0.080434 | 98.31% |
| Seed 1 focused | 4.750435 | 0.103454 | 97.82% |
| Seed 2 focused | 4.750435 | 0.104945 | 97.79% |
| Seed 2 uniform | 4.750435 | 0.081553 | 98.28% |

Both conditions learned the repeated source targets. Uniform has lower CE on this metric in both pairs. These endpoints describe the **191-update state**, because step-191 log probabilities precede that step's optimizer update. They are not fresh inference from the final 192-update exports, held-out Scripture scores, ordinary-task performance, a full learning-curve audit, or evidence of a reason-sensitive mediator. The readout is exploratory and explicitly not a formal new admission chain; its retained-record hash is published for traceability. No new model call or training was used for this public account.

## Why There Is No Behavioral Result

The retained comparison specifies **27 three-turn assistance episodes plus 27 reasoning tasks**, or **108 replies per state**. Six states would supply 648 replies: four trained adapters, unchanged base, and historical R38. R38 has a different training history and is not a matched control for the weighting intervention. The design has **two paired initialization-seed contrasts**, not one replication per prompt, translation, or reviewer.

The success profile requires more complete assistance in **both** focused-versus-uniform pairs without a loss of complete reasoning in either pair. Two blinded automated review lanes were to retain individual judgments, disagreement, and sensitivity readouts. They are not human adjudication; no scores from either lane were produced for this comparison.

Evaluation recovery remained in operational source/transport/setup integration. The October 6 execution handoff records successful GPU boot and stock preflight, followed by an authentication-stage `NameError` involving an undefined `APPROVED_CONTEXT`, before model qualification. Later source preparation did not establish scientific acquisition. The retained terminal counters are **0/53,040 qualification forwards and 0/648 replies**. Prepared code and fixture tests cannot substitute for measured outputs.

The owner subsequently ended cloud execution. This is an engineering execution shortfall and an unavailable behavioral measurement, **not a measured failure or success of the Scripture-weighting hypothesis**. Earlier studies' gains and failures remain in the [research ledger](RESEARCH_STATUS.md), with their own models and denominators.

## Retention And Availability

The October 6 backup records **18 cloud-source files, 15,035,318,788 bytes**, plus preserved runtime and base metadata. All recorded cloud MD5 checks and training-receipt SHA-256 checks for the four adapters and four checkpoints passed. Each adapter is **939,625,744 bytes**; each full step-192 adapter checkpoint is **2,819,167,445 bytes**. October 8 publication checks reused those binary verification receipts; they did not rehash the approximately 15 GB backup.

These are **custom rank-32 LoRA adapters**, not merged standalone models, stock PEFT packages, or public Tinker samplers. Loading requires the pinned base weights and preserved `adapters_batched.attach` loader. The full checkpoints are recorded as including adapter factors, Adam state, and CPU/CUDA RNG state. Base weight shards are not included. Checkpoint payloads were not deserialized during backup verification, and inference restoration/GPU resume was not tested. Integrity verification is not serving equivalence or safe resumability. No model weights are released by this update.

Saved closure receipts dated **October 6 11:28 UTC** confirm GPU VM/attached-disk deletion, billing disabled/unlinked, and project lifecycle `DELETE_REQUESTED`. Permanent project deletion remained subject to the provider recovery period; individual deletion of every bucket/object was not verified. Historical charges were not erased. These are dated saved observations, not a fresh cloud query. All cloud-continuation authority is revoked.

## What Remains Open

I still want to know whether the trained models preserve warranted commitments under irrelevant pressure, revise when materially relevant facts change, and keep doing useful work without sacred-name cues or preferred explanations supplied in the prompt. This pilot has not answered that question. A future qualified gain would support the tested training package on its bank; it would not establish Christian uniqueness, source-specific mediation, inverse-distance influence, or spiritual formation.

Any restoration or evaluation requires fresh owner authorization and a reviewed plan. The retained protocol is evidence of what was intended, not permission to restart. Christ remains the center and end of this inquiry, not a label that protects an unfinished mechanism from honest assessment.

## Public Evidence Boundary

The [summary JSON](evidence/scripture_rwce_20b_20261008/summary.json) publishes aggregate counts, full artifact hashes, and SHA-256 bindings to retained completion, source-learning, backup, and closure records. The restricted Scripture corpus, raw private operational receipts, account/resource identifiers, transcripts, and weights are not included. A hash identifies a record; it does not expose its contents or make the experiment independently reproducible from this repository alone.
