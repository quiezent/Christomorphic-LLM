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
| `train.py` | LoRA training on Tinker (`fit`): AdamW or Dimuon, cosine or linear-to-zero schedule, per-document loss logging, a TTL on every save. Also run 1's data, where loss falls only on the recalled scene in the analysis channel |
| `bible.py` | Exact ESV/NKJV verse text by reference, from the audited files |
| `sw.py` | Runs 2 and 4 data pipeline: prompts (oasst1, HH-RLHF), passage picking and confirmation, candidates, objective faithfulness checks, judge rankings, selection, training |
| `sw_eval.py` | Held-out witness-gap test, four-model held-out comparison with the judge, judge-decision packet |
| `stage1.py` | Run 3 and the Stage 1 optimizer test: whole ESV/NKJV as plain chapter documents (optimizer, schedule and hold-out options), in-layout scoring of the text (`layout`), forgetting on base's own replies (`drift`), and a verse-recall test |
| `stage3.py` | Runs 4 and 5: GRPO-style RL with a Scripture-reading judge, both label orders. `--reward rank` (run 4: faults and violations −1) or `--reward servant` (run 5: violations dominate, length neutral); `--pilot` judges the base model once with no update |
| `servant.py` | Run 5 prompts: the exact words of the person speaking in each Scripture scene (ESV and NKJV), the scene and Philippians 2:3-11 for the judge, bridge and ordinary prompts, and a scene-level hold-out |
| `scenes.json` | Run 5's scene list: ids, families, the verses spoken and the verses the judge reads. References only, no Scripture text |
| `pilot_report.py` | Run 5 pilot readout by scene family: reward spread, violations, faults, Bible-study mode, length |
| `side_effects.py` | Mechanical side-effect counts on the 89 probes, the same for every model |
| `s1ab_report.py` | Readout of the Stage 1 optimizer test, with paired intervals |
| `dimuon_smoke.py` | Three-step check that Dimuon trains a gpt-oss-20b LoRA on Tinker |
| `judges.py` | Judge bake-off: order consistency, agreement with blinded review, violation recall, cost |

## Added 23 November 2026: the words in the original tongues

These run locally (PyTorch, transformers and PEFT; Ministral 3 8B Instruct in 4-bit on an 8 GB card), not on Tinker. They read the Berean Standard Bible, the Hebrew and Greek texts and the publisher's word-by-word table from `data/`, none of which is here.

| File | Purpose |
|---|---|
| `tongues.py` | Training and reading 1: English words drawn to the original words they render at layer 17, true and false pairing, the place-only and exact-words recipes, and the share of words whose own original is nearest on unread chapters |
| `tongues_words.py` | The join of each English phrase to the original word it renders, as character spans in the two texts |
| `tongues_harm.py` | Speech: unread Hebrew, Greek and English verses, ordinary prose, and the letter sets, each arm against base |
| `words_probe.py` | The English read alone: drawn together and held apart, with the baselines that need no model, the across-books and single-word cuts |
| `speakers_probe.py`, `speakers_scene.py` | Who is speaking: the table of scenes, the capped score, and the scene-by-scene score |
| `chat.py` | Replies from base and adapters on the same seeds, at a chosen strength; `--first-draw` draws on seeds no earlier run used |
| `pairs_probe.py`, `probe.py`, `rd0.py`, `measure_base.py` | Loading, scaling and scoring helpers the scripts above import |
| `bible.py` | Updated: reference lookup across the translations held locally |

