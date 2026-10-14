# Code

The scripts behind the reports in this folder. They run against `tinker==0.33.1` and `tinker-cookbook==0.5.7`. Two things are not here:
- The ESV and NKJV verse files they read from `data/`, which are licensed for research only.
- The GSM8K test file. It comes from [openai/grade-school-math](https://github.com/openai/grade-school-math) and goes in `bench/gsm8k/test.jsonl`.

| File | Purpose |
|---|---|
| `evaluate.py` | Samples base and fine-tuned gpt-oss models on the geometry probes or GSM8K. It has bare and native Harmony rendering, per-model temperature and no prefill, and writes raw replies plus a blinded packet with its key kept in a separate file |
| `review.py` | Builds blinded review groups and aggregates reviewer scores. Also includes a quoted-span overlap check against ESV/NKJV, and the recall-exactness readout used for training run 1 |
| `eval_probes.json` | The 28-probe subset Clayton approved for reading packets. It records which probes R38 trained on and which were swapped out |
| `build_data.py` | Builds training run 1's data: ask-and-reply scenes from the Gospels and Acts, exact ESV and NKJV text |
| `refs.json` | The 99 ask-and-reply pairs kept by two rule-applying agents. References only, no Scripture text |
| `train.py` | LoRA training on Tinker. Loss falls only on the recalled scene in the analysis channel, and every save has a TTL |
| `bible.py` | Exact ESV/NKJV verse text by reference, from the audited files |
| `sw.py` | Runs 2 and 4 data pipeline: prompts (oasst1, HH-RLHF), passage picking and confirmation, candidates, objective faithfulness checks, judge rankings, selection, training |
| `sw_eval.py` | Held-out witness-gap test, four-model held-out comparison with the judge, judge-decision packet |
| `stage1.py` | Run 3: whole ESV/NKJV as plain chapter documents, plus a verse-recall test |
| `stage3.py` | Runs 4 and 5: GRPO-style RL with a Scripture-reading judge, both label orders. `--reward rank` (run 4: faults and violations −1) or `--reward servant` (run 5: violations dominate, length neutral); `--pilot` judges the base model once with no update |
| `servant.py` | Run 5 prompts: the exact words of the person speaking in each Scripture scene (ESV and NKJV), the scene and Philippians 2:3-11 for the judge, bridge and ordinary prompts, and a scene-level hold-out |
| `scenes.json` | Run 5's scene list: ids, families, the verses spoken and the verses the judge reads. References only, no Scripture text |
| `pilot_report.py` | Run 5 pilot readout by scene family: reward spread, violations, faults, Bible-study mode, length |
| `judges.py` | Judge bake-off: order consistency, agreement with blinded review, violation recall, cost |
