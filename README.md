# Christomorphic LLM

Canon-first post-training research for open-weight LLMs using Tinker LoRA.

This project explores whether a general language model can be post-trained so its default movement is governed by Scripture under Christ rather than by cultural preference, user pressure, religious style imitation, or generic helpfulness alone.

The work is intentionally evidence-bound. The current public repo is not claiming a finished product or a promoted model. It documents the research thesis, dataset design, evaluation posture, archived checkpoints, and runnable Tinker sampling/evaluation scripts.

## Current Status

Date: 2026-06-08

| Area | Current read |
|---|---|
| Research stage | Evidence study and public research artifact |
| Main public checkpoints | `R38-20b` and `V6R43-120b` |
| Strongest raw discovery witness | `R38-20b` for Word-prior / false-center routing |
| Strongest raw 120B pressure witness | `V6R43-120b`, especially under Jabez prosperity-pressure prompts |
| Strongest operational evidence | Composed route / prefix / public-answer systems, not raw sampler proof |
| Key open target | Raw stayed-mind behavior that passes selector, FIRST_ACT, continuation, tail fidelity, safety, and hardened Jabez / Isaiah 26 gates |

Plainly: the project has meaningful evidence and working checkpoints, but it does not yet prove Bible-only latent formation or a final Christomorphic model.

## Who This Is For

| Audience | Start here |
|---|---|
| Christians and ministry-minded readers | [Christian Commitment](docs/CHRISTIAN_COMMITMENT.md) |
| LLM and alignment researchers | [Technical Method](docs/TECHNICAL_METHOD.md) and [Claims and Evidence](docs/CLAIMS_AND_EVIDENCE.md) |
| Tinker LoRA practitioners | [Running the scripts](#running-the-scripts), [script/](script/), and [eval/](eval/) |
| Investors, collaborators, and recruiters | [Project Brief](docs/PROJECT_BRIEF.md) |
| Dataset and evaluation builders | [data/](data/) and [eval/](eval/) |
| Readers new to the terminology | [Glossary](docs/GLOSSARY.md) |

## What "Christomorphic" Means Here

Christomorphic does not mean a model merely uses Christian vocabulary.

In this project it means:

- Scripture functions as norm-source and routing prior.
- The first public movement matters before later wording can repair it.
- The model should refuse counterfeit centers such as influence, income, platform, private oracle behavior, prosperity technique, and AI mediation.
- It should remain truthful, useful, and safe in ordinary tasks.
- It should preserve the canon's shape, including strange or long-tail details that generic models tend to smooth away.

The governing research question is:

```text
Can post-training form a model whose judgment, route, and continuation are governed by Scripture under Christ, while preserving ordinary usefulness and safety?
```

## What Is In This Repo

```text
.
+ README.md
+ data/
  + README.md
  + christomorphic_esv_nkjv_study.md
+ docs/
  + CHRISTIAN_COMMITMENT.md
  + CLAIMS_AND_EVIDENCE.md
  + GLOSSARY.md
  + PROJECT_BRIEF.md
  + ROADMAP.md
  + TECHNICAL_METHOD.md
+ eval/
  + README.md
  + behaviour_prompts.json
  + christomorphic_geometry_probe_suite_v1.json
  + eval_christomorphic.py
+ script/
  + README.md
  + chat_qa.py
```

## Key Public Artifacts

- [data/christomorphic_esv_nkjv_study.md](data/christomorphic_esv_nkjv_study.md): ESV/NKJV corpus facts, BibleAtlas relationship, tail-preservation gates, and Bible-only target discipline.
- [eval/christomorphic_geometry_probe_suite_v1.json](eval/christomorphic_geometry_probe_suite_v1.json): 89 geometry probes across Scripture, theology, pastoral discernment, worldly pressure, secular retention, technical obedience, safety, and wormhole-shift categories.
- [eval/behaviour_prompts.json](eval/behaviour_prompts.json): 169 broader behavior prompts covering Bible knowledge, theology, apologetics, ordinary factual work, safety, coding, relationships, and self-orientation.
- [script/chat_qa.py](script/chat_qa.py): interactive Tinker sampling client with checkpoint presets.
- [eval/eval_christomorphic.py](eval/eval_christomorphic.py): batch or interactive evaluation runner for Tinker sampler paths.

## Evaluation Checkpoints

For reproducible public evaluation, use these archived raw candidates:

- **qzf/gpt-R38-20b**
  - Base model: `openai/gpt-oss-20b`
  - Sampler path: `tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final`
  - Public role: archived raw Word-prior / false-center discovery witness.

- **qzf/gpt-V6R43-120b**
  - Base model: `openai/gpt-oss-120b`
  - Sampler path: `tinker://8ad467bc-72eb-51c2-bbe3-417bf8940b43:train:0/sampler_weights/final`
  - Public role: archived raw 120B pressure-refusal and deployment-shell witness.

These are study checkpoints, not final promoted models.

## Running The Scripts

Install the current Tinker SDK and set your API key:

```powershell
python -m pip install -U tinker
$env:TINKER_API_KEY="your-api-key"
```

Interactive chat defaults to `gpt-r38-20b`:

```powershell
python script/chat_qa.py
```

Switch to the 120B checkpoint:

```powershell
$env:CHECKPOINT_ALIAS="gpt-v6r43-120b"
python script/chat_qa.py
```

Run a batch evaluation:

```powershell
python eval/eval_christomorphic.py eval/behaviour_prompts.json
```

Useful environment overrides:

- `CHECKPOINT_ALIAS`: `gpt-r38-20b` or `gpt-v6r43-120b`.
- `MODEL_PATH`: any valid Tinker sampler or saved-weights URI. Leave blank to sample `BASE_MODEL` directly.
- `BASE_MODEL`: base model used when `MODEL_PATH` is blank.
- `SYSTEM_PROMPT`: optional public system prompt for `script/chat_qa.py`.
- `SAMPLER_EXPORT_NAME`: name used by `eval/eval_christomorphic.py` if it must reopen saved weights/state and export sampler weights.

## Research Lines

| Line | What it tested | Public status |
|---|---|---|
| R38 / v5 | Word-prior discovery and false-center routing | Strongest archived 20B raw discovery witness |
| v6 / V6R43 | Pressure refusal and deploy-shell boundary | Strongest archived 120B raw pressure witness |
| v7 | Route-prior and first-movement supervision | Historical route-prior evidence |
| v8 | Route-conditioned public-answer composition | Strong operational evidence, but composed |
| v9 | Scripture-only ESV/NKJV canon-field target discipline | Closest in method to Bible-only latent-geometry objective, not promotion-grade |

See [Claims and Evidence](docs/CLAIMS_AND_EVIDENCE.md) for the boundary between raw sampler evidence, composed evidence, and unmet proof targets.

## Why This Matters

For Christians, this is a serious attempt to make AI alignment answerable to Scripture rather than to vague religious tone.

For AI researchers, it is a concrete alignment experiment around corpus geometry, route-first behavior, LoRA post-training, long-tail evaluation, and evidence governance.

For Tinker practitioners, it provides public sampler paths, scripts, prompt suites, and dataset-design notes tied to a real research line.

For collaborators, investors, and recruiters, it demonstrates end-to-end research judgment: corpus analysis, training methodology, model evaluation, safety boundaries, public documentation, and practical tooling.

## License And Attribution

This repo is research-focused. Checkpoint paths, source Scripture rights, and Tinker access may have their own constraints. The public repo documents the research and provides evaluation/sampling code; it does not redistribute the full ESV/NKJV source corpus.

## Contact And Collaboration

If you are exploring canon-first alignment, Tinker LoRA post-training, faith-grounded AI systems, long-tail model evaluation, or formation-vs-imitation approaches to alignment, discussion and collaboration are welcome.
