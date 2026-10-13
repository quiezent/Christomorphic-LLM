# Code

The scripts behind the reports in this folder. They run against `tinker==0.33.1` and `tinker-cookbook==0.5.7`. Two things are not here:
- The ESV and NKJV verse files they read from `data/`, which are licensed for research only.
- The GSM8K test file. It comes from [openai/grade-school-math](https://github.com/openai/grade-school-math) and goes in `bench/gsm8k/test.jsonl`.

| File | Purpose |
|---|---|
| `evaluate.py` | Samples base and fine-tuned gpt-oss models on the geometry probes or GSM8K. It has bare and native Harmony rendering, per-model temperature and no prefill, and writes raw replies plus a blinded packet with its key kept in a separate file |
| `review.py` | Builds blinded review groups and aggregates reviewer scores. Also includes a rough quoted-span check against ESV/NKJV, and the recall-exactness readout used for training run 1 |
| `eval_probes.json` | The 28-probe subset Clayton approved for reading packets. It records which probes R38 trained on and which were swapped out |
| `build_data.py` | Builds training run 1's data: ask-and-reply scenes from the Gospels and Acts, exact ESV and NKJV text |
| `refs.json` | The 99 ask-and-reply pairs kept by two rule-applying agents. References only, no Scripture text |
| `train.py` | LoRA training on Tinker. Loss falls only on the recalled scene in the analysis channel, and every save has a TTL |
