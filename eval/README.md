# Evaluation

This folder contains public prompt suites and a Tinker evaluation runner. The runner has been checked against installed `tinker==0.23.4`.

## Files

| File | Purpose |
|---|---|
| [behaviour_prompts.json](behaviour_prompts.json) | 169 broad behavior prompts across Bible, theology, secular factual work, safety, coding, relationships, and self-orientation |
| [christomorphic_geometry_probe_suite_v1.json](christomorphic_geometry_probe_suite_v1.json) | 89 geometry probes across Scripture, Christology, pressure, retention, technical obedience, and safety |
| [eval_christomorphic.py](eval_christomorphic.py) | Batch or interactive Tinker sampling script |

## Run

```powershell
python -m pip install -r requirements.txt
$env:TINKER_API_KEY="your-api-key"
python eval/eval_christomorphic.py eval/behaviour_prompts.json
```

The script writes generated outputs to a timestamped `.jsonl` file next to the prompt file. Each row records the checkpoint/base model, system prompt, timestamp, and sampling parameters. Generated result files are ignored by Git.

Useful environment controls:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | R38 sampler path | Tinker sampler or saved-state path |
| `BASE_MODEL` | `openai/gpt-oss-20b` | Used when `MODEL_PATH` is blank |
| `SYSTEM_PROMPT` | empty | Optional inference scaffold; report it with results |
| `MAX_TOKENS` | `1024` | Maximum generated tokens |
| `TEMPERATURE` | `0.5` | Sampling temperature |
| `TOP_P` | `0.9` | Nucleus-sampling threshold |

## What These Evals Are

These are public behavioral surfaces for inspection and comparison. They can reveal differences in raw outputs, failure modes, first movement, continuation, ordinary retention, and safety handling. They are not the full internal proof harness.

A stronger claim additionally requires:

```text
source and target audit
-> fresh common starts and matched controls
-> held-out semantic and relation evidence
-> causal removal, restoration, graft, and rescue
-> seed and ESV/NKJV replication
-> bare FIRST_ACT and stayed continuation
-> tail peculiarity, ordinary retention, and safety
-> blinded human review and independent reproduction
```

Sampling results alone remain behavioral evidence. They do not expose hidden states or establish the causal mechanism behind an answer.

## Raw And Composed Comparisons

When reporting a run, state whether the output came from:

- a raw sampler;
- a canon-anchored prompt;
- a route/prefix/retrieval composition;
- a system prompt or answer-repair shell.

Do not compare a composed result against a raw checkpoint as if both measured the same object.
