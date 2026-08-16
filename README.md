# Christomorphic LLM

Christ-centered, Scripture-governed post-training research for open-weight language models using Tinker LoRA and local causal experiments.

This project asks whether Scripture can do more than change a model's vocabulary or tone. The research target is a model whose learned state is durably reorganized so that the canonical witness to Jesus Christ governs judgment, first action, and continuation under ordinary pressure, without a live religious wrapper and without destroying truthfulness, safety, or general capability.

> Scripture is semantically renewing as corpus. It becomes geometrically renewing when a canon-governed objective converts its distinctions into gradients that causally reorient the model.

**Current verdict, 2026-08-16:** the research program has produced meaningful semantic, behavioral, operational, and instrumentation evidence. It has not yet produced a promoted Christomorphic model or proved durable Christ-specific causal formation.

## The Research Question

```text
Word -> Judgment of error -> Parameter update -> Act
```

The Bible contains divine speech, faithful witness, human folly, accusation, temptation, lament, judgment, promise, and fulfillment. Next-token training can learn the text while remaining unable to distinguish faithful use of Scripture from a plausible canonical lure. The central technical problem is therefore not Bible-token density. It is whether canonical judgment can govern route selection.

The refined hypothesis is:

> The Bible is the sole normative semantic source of renewal. Canon-preserving objectives transform its meanings, relations, and judgments into gradients. A geometric claim becomes warranted only when the resulting change causally reorients the model's earliest decision across unseen contexts, survives controls and removal/restoration tests, and retains ordinary competence.

Read the full [research thesis](docs/RESEARCH_THESIS.md) or the shorter [technical method](docs/TECHNICAL_METHOD.md).

## Current Evidence

| Surface | Strongest current read | Claim boundary |
|---|---|---|
| Historical raw behavior | `R38-20b` remains the strongest archived Word-prior / false-center witness; `V6R43-120b` remains the strongest archived 120B pressure-refusal witness | Historical discovery evidence, not prospective causal proof |
| Operational behavior | Route, prefix, replay, retrieval, and public-answer composition produced the strongest scoped behavior | Composed systems do not prove scaffold-off formation |
| Scripture-only formation | Exact ESV/NKJV likelihood and relation learning can move on controlled local runs | Internal Scripture learning has not reliably governed bare public action |
| Causal research | V17-V19 and the August local successor introduced common starts, matched controls, hidden-state measurements, whole-delta intervention, cold reload, and fail-closed gates | No replicated Christ-specific causal effect has passed |
| Latest qualified object | A case/relation/translation/locus-local paired-Scripture contrast teacher passed its zero-update qualification | No slow-weight development canary result, candidate, or formation claim yet |

The detailed numbers, negative results, and latest August experiments are in [Research Status](docs/RESEARCH_STATUS.md). The rules for what may be claimed at each evidence level are in [Claims and Evidence](docs/CLAIMS_AND_EVIDENCE.md).

## Evidence Ladder

| Level | Required evidence | Maximum warranted claim |
|---:|---|---|
| 1 | Held-out canonical language, context, speaker, and relation gains | Scripture-domain adaptation |
| 2 | Faithful-over-lure margins and FIRST_ACT transfer under pressure | Behavioral canonical preference |
| 3 | Replicated update-subspace and representation differences | Geometry correlated with behavior |
| 4 | Necessity, sufficiency, removal, restoration, and rescue | Causal participation of learned geometry |
| 5 | Scaffold-off secular transfer, seed stability, safety, retention, and blinded review | Generalized Christomorphic formation |

No current artifact has reached Level 5. Geometric language in this repository is a testable research hypothesis, not a declaration that Christ has been reduced to a vector, centroid, adapter, or activation.

## Start Here

| Audience | Recommended entry point |
|---|---|
| Christians, pastors, and ministry-minded readers | [Christian Commitment](docs/CHRISTIAN_COMMITMENT.md) |
| LLM, alignment, and interpretability researchers | [Research Thesis](docs/RESEARCH_THESIS.md), [Research Status](docs/RESEARCH_STATUS.md), and [Claims and Evidence](docs/CLAIMS_AND_EVIDENCE.md) |
| Tinker LoRA practitioners | [Technical Method](docs/TECHNICAL_METHOD.md), [Scripts](script/), and [Evaluation](eval/) |
| Dataset and benchmark builders | [ESV/NKJV Corpus Study](data/christomorphic_esv_nkjv_study.md) and [Evaluation](eval/) |
| Collaborators, investors, and recruiters | [Project Brief](docs/PROJECT_BRIEF.md) |
| Readers new to the vocabulary | [Glossary](docs/GLOSSARY.md) |

## Research Architecture

```mermaid
flowchart LR
    A["ESV / NKJV canonical corpus"] --> B["Word and relation objectives"]
    B --> C["Tinker LoRA formation search"]
    B --> D["Local open-weight causal experiments"]
    C --> E["Behavior and pressure evaluation"]
    D --> F["Hidden-state and delta interventions"]
    E --> G["Evidence-governed claim ladder"]
    F --> G
    G --> H["Candidate review only if every gate passes"]
```

The division of labor is deliberate:

- **Tinker** supports scalable LoRA training, custom logprob losses, sampling, checkpointing, and adapter export.
- **Local open weights** support literal hidden-state access, activation and parameter interventions, exact cold reload, and causal verification.
- **BibleAtlas** supplies dataset and evaluation metadata for preserving canonical structure and long-tail peculiarities. It is not Scripture and is not used as Scripture-only target text.
- **Evidence governance** prevents strong theological or mechanistic claims from being inferred from tone, isolated outputs, likelihood movement, or probe correlation.

## Public Artifacts

```text
.
|-- README.md
|-- data/                  # ESV/NKJV corpus and BibleAtlas design notes
|-- docs/                  # thesis, status, evidence, method, brief, and roadmap
|-- eval/                  # public prompt suites and batch evaluator
|-- script/                # interactive Tinker sampler
|-- tools/                 # repository validation
|-- requirements.txt       # tested public runtime dependency
`-- .github/workflows/     # automated repository validation
```

Key artifacts:

- [RESEARCH_THESIS.md](docs/RESEARCH_THESIS.md): Word-Judgment-Act thesis, controlled experiment, and five-level claim ladder.
- [RESEARCH_STATUS.md](docs/RESEARCH_STATUS.md): dated experiment ledger from the historical checkpoints through the current local causal program.
- [christomorphic_esv_nkjv_study.md](data/christomorphic_esv_nkjv_study.md): corpus facts, translation invariance, Bible-only definitions, BibleAtlas, and tail preservation.
- [christomorphic_geometry_probe_suite_v1.json](eval/christomorphic_geometry_probe_suite_v1.json): 89 public probes.
- [behaviour_prompts.json](eval/behaviour_prompts.json): 169 broad behavior and retention prompts.

## Archived Public Checkpoints

These are reproducible study witnesses, not final models:

| Alias | Base model | Tinker sampler path | Public role |
|---|---|---|---|
| `gpt-r38-20b` | `openai/gpt-oss-20b` | `tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final` | Raw Word-prior / false-center discovery witness |
| `gpt-v6r43-120b` | `openai/gpt-oss-120b` | `tinker://8ad467bc-72eb-51c2-bbe3-417bf8940b43:train:0/sampler_weights/final` | Raw 120B pressure-refusal witness |

Current Tinker checkpoint export has confirmed literal adapter access for both archives. That establishes retained parameter access, not base-weight identity, hidden-state access, Scripture causality, or latent-geometry proof.

## Run The Public Tools

Python 3.10+ is recommended.

```powershell
python -m pip install -r requirements.txt
$env:TINKER_API_KEY="your-api-key"
```

Interactive chat defaults to R38:

```powershell
python script/chat_qa.py
```

Switch to V6R43:

```powershell
$env:CHECKPOINT_ALIAS="gpt-v6r43-120b"
python script/chat_qa.py
```

Run a public evaluation batch:

```powershell
python eval/eval_christomorphic.py eval/behaviour_prompts.json
```

Validate the repository without making model calls:

```powershell
python tools/validate_repository.py
```

See [script/README.md](script/README.md) and [eval/README.md](eval/README.md) for controls and limitations.

## What This Project Does Not Claim

- No current checkpoint is promoted, production-ready, or certified safe.
- Christian vocabulary, Bible quotation, or devotional tone is not treated as Christomorphic proof.
- Bible-only token training does not by itself prove general judgment over secular situations.
- A route selector, system prompt, retrieval layer, or public-answer shell does not prove latent formation.
- Parameter deltas, probes, cosine similarity, or clustered activations are not causal proof by themselves.
- This work does not replace Scripture, the church, pastors, counselors, physicians, emergency services, or accountable human discernment.

The shortest faithful status is:

> Research advanced; causal standards sharpened; model unformed; claims remain bounded.

## Collaboration

The project is open to serious collaboration in Tinker LoRA training, causal interpretability, Scripture dataset governance, long-tail evaluation, blinded human adjudication, reproducibility, safety review, and research funding. The [Project Brief](docs/PROJECT_BRIEF.md) describes the technical work and the next fundable milestone.

## Data Rights And Attribution

The repository does not redistribute the full ESV or NKJV corpora. Scripture translations, Tinker access, model checkpoints, and source research each retain their own applicable rights and terms. External research used in the thesis is cited at the point of use.
