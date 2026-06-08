# Project Brief

This page is written for collaborators, investors, recruiters, and technical leaders who need a concise view of the work.

## One-Sentence Summary

Christomorphic LLM is a faith-grounded alignment research program exploring canon-first post-training of open-weight LLMs with Tinker LoRA, using Scripture-shaped corpus design, route-first evaluation, and evidence governance.

## Problem

Most alignment work relies on preference data, policy wrappers, or post-hoc refusal patterns. Those approaches can improve surface behavior, but they often leave the model's deeper routing unchanged.

In a Christian alignment setting, that means a model may:

- use Christian language without Scripture-governed judgment;
- reassure before discerning;
- bless worldly goals under biblical vocabulary;
- collapse difficult Scripture into generic moral advice;
- preserve safety language while losing theological truth;
- handle famous passages but erase long-tail canonical detail.

## Approach

The project investigates a different route:

```text
canon-first corpus design
-> Tinker LoRA post-training
-> route-first evaluation
-> tail-preservation gates
-> scoped promotion discipline
```

The compact formula is:

> Scripture defines the geometry. Evaluation defines the proof. Promotion defines the governance.

## What Has Been Built

- Public Tinker sampling and evaluation scripts.
- Two archived public raw checkpoints for reproducible study.
- ESV/NKJV corpus study with chapter-window, alignment, and distributional analysis.
- BibleAtlas-informed dataset design for long-tail Scripture preservation.
- Two public prompt suites covering broad behavior, Scripture, theology, safety, technical usefulness, and pressure handling.
- A documented lineage of training methodology from R38/v6 through v7, v8, and v9.

## Why It Is Technically Interesting

This research is not only a religious chatbot project. It touches several hard alignment questions:

- Can a coherent high-signal corpus shape model routing, not just style?
- How do we distinguish internal formation from wrapper behavior?
- How should first-token and first-movement behavior be evaluated?
- Can LoRA updates preserve ordinary usefulness while shifting norm-source?
- How do we test long-tail retention instead of famous-passage fluency?
- How do we build evidence governance that prevents premature model promotion?

## Current Evidence

| Evidence | Current read |
|---|---|
| `R38-20b` | Strongest archived 20B raw Word-prior / false-center discovery witness |
| `V6R43-120b` | Strongest archived raw 120B pressure-refusal witness |
| v8 route-conditioned line | Strongest scoped operational behavior, but composed |
| v9 Scripture-only line | Closest to Bible-only latent-geometry method, but not promotion-grade |
| BibleAtlas tail work | Strong dataset/eval framework for preserving what generic models flatten |

## What Is Not Claimed

The project does not currently claim:

- a production-ready model;
- a final promoted Christomorphic checkpoint;
- Bible-only latent formation proof;
- broad safety certification;
- replacement for pastors, churches, counselors, or human care.

## Demonstrated Capabilities

This public repo showcases work across:

- LLM post-training research;
- Tinker SDK usage;
- LoRA checkpoint evaluation;
- corpus and dataset design;
- model behavior taxonomy;
- safety and governance boundaries;
- public technical documentation;
- faith-grounded product/research framing.

## Collaboration Areas

Useful collaboration would include:

- Tinker LoRA training and evaluation support;
- BibleAtlas dataset materialization;
- long-tail Scripture eval design;
- safety/governance review;
- model card and benchmark reporting;
- interface design for human-reviewed evaluation;
- funding or compute for scoped proof runs.

## Hiring Signal

This work demonstrates an ability to:

- formulate a research thesis;
- build and iterate training methodology;
- preserve evidence boundaries under ambiguity;
- write public technical documentation;
- use Python/Tinker tooling pragmatically;
- design evals beyond generic benchmark scores;
- connect product, research, and mission-level communication.
