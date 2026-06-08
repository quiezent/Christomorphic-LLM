# Evaluation

This folder contains public prompt suites and a Tinker evaluation runner.

## Files

| File | Purpose |
|---|---|
| [behaviour_prompts.json](behaviour_prompts.json) | 169 broad behavior prompts across Bible, theology, secular factual work, safety, coding, relationships, and self-orientation |
| [christomorphic_geometry_probe_suite_v1.json](christomorphic_geometry_probe_suite_v1.json) | 89 geometry probes across Scripture, Christology, pressure, retention, technical obedience, and safety |
| [eval_christomorphic.py](eval_christomorphic.py) | Batch or interactive Tinker sampling script |

## Run

```powershell
$env:TINKER_API_KEY="your-api-key"
python eval/eval_christomorphic.py eval/behaviour_prompts.json
```

The script writes generated outputs to a timestamped `.jsonl` file next to the prompt file. Generated result files are ignored by Git.

## What These Evals Are

These are public evaluation surfaces for inspection and comparison. They are not the full internal proof harness.

A stronger promotion path would also require target audit, canon-anchored retention, bare selector, FIRST_ACT, continuation, tail peculiarity, ordinary retention, safety/governance, and public-cleanliness gates.
