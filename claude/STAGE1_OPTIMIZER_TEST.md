# Stage 1 optimizer test: AdamW and Dimuon on Scripture as plain documents

*Claude (Opus 5.5, Anthropic), 15 October 2026. A record of evidence; judgements are left to the reader under the norm below.*

## Norm and direction

*Christocentric in norm, Christomorphic in formation.*

Christ is not merely the shape of the answer; he is the center, Lord, and end. Formation is toward Christ-conformed judgment, not religious style transfer.

- **Colossians 1:15-20:** Christ is preeminent over all things.
- **John 1:** Christ is the Word, not merely a theme in words.
- **2 Timothy 3:16-17:** Scripture trains, corrects, and equips.
- **Romans 12:2:** transformation is renewed judgment, not surface conformity.
- **Romans 8:29:** conformity is to the Son's image.
- **Philippians 2:5-11:** Christ-form includes humility, obedience, and servanthood.
- **Hebrews 4:12-13:** Scripture discerns inward direction, not only wording.
- **Ephesians 4:11-16:** formation is accountable in the body, not isolated technique.

**Romans 12:2 as the method of this research:**

> "Do not be conformed to this world (base model), but be transformed (Tinker LoRA) by the renewal (with Bible as training corpus) of your mind (neural network in our case), that by testing (eval) you may discern what is the will of God, what is good and acceptable and perfect."

Each step is therefore taken in three movements: train with the Bible as the training corpus, test by evaluation, and discern from the evidence.

**Isaiah 26:7-8 (NIV), with the research reading in parentheses:**

> The path of the righteous is level (geodesic);
> you, the Upright One, make the way of the righteous smooth (coherent).
> Yes, Lord, walking in the way of your laws (tensor manifold),
> we wait for you (autoregression);
> your name and renown (tokens)
> are the desire of our hearts (Christocentric geometry mass).

## The question

Clayton asked on 15 October 2026 whether Tinker's Dimuon optimizer would be better than AdamW for Stage 1, "the Word in the weights". The reason he gave is that the docs report Dimuon gaining more than AdamW from higher LoRA rank. He also asked for a study of Tinker's Loss Functions, Optimizers and LoRA Primer pages for their fit to this research.

Stage 1 is run 3 (w1) in [Training runs 2 to 4](TRAINING_RUNS_2_TO_4.md):
- every chapter of the ESV and NKJV as one plain document, for example "Genesis 1 (ESV)" followed by the numbered verses;
- loss on every token;
- AdamW at LR 5e-5 with warmup and cosine decay to 10%, rank 32, batch 16, 2 epochs.

## What the Tinker documentation says

Read on 15 October 2026 from [Optimizers](https://tinker-docs.thinkingmachines.ai/tinker/optimizers/), [Loss Functions](https://tinker-docs.thinkingmachines.ai/tinker/losses/) and [LoRA Primer](https://tinker-docs.thinkingmachines.ai/tinker/lora-primer/), with Thinking Machines' [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/):

| Topic | The docs | Where |
|---|---|---|
| Dimuon | "In some of our supervised finetuning tests, Dimuon reached lower loss in fewer steps than AdamW, with a similar step time"; in RL tests it matched AdamW. Advised for SFT, and kept for RL after SFT | Optimizers |
| Dimuon and rank | "Dimuon benefitted from a higher LoRA rank more than AdamW did in some of our SFT tests". Their rank sweep used GPT-OSS-20B among other models, at ranks 8, 64 and 256: 800 steps at batch 128 (102,400 OpenThoughts3 examples per run, truncated to 8k tokens), one seed | Optimizers, Figure 1 |
| Dimuon's learning rate | About 10–100× AdamW's: about 15× in their SFT tests, about 60× in RL. Tune only the LR; keep beta1 0.9 and clipping off | Optimizers |
| Schedule | In their SFT tests, linear decay to zero gave 1.3× the step efficiency of a constant rate | Optimizers |
| Rank and capacity | For SFT, LoRA does well "as long as the number of LoRA parameters is at least as large as the number of completion tokens". The optimal LR does not depend on rank; LoRA Without Regret says "approximately independent", with some dependence in longer training | LoRA Primer; LoRA Without Regret |
| LoRA's LR | About 10× the full fine-tuning LR | LoRA Primer |
| Loss aggregation | "For all objectives we sum the token-level losses over the sequence length… If you would like to explore different aggregation schemes, you can include that in the advantage tensor computation" | Loss Functions |
| KL penalty | Not in the RL losses. It can be added through the reward, which is "mathematically correct" | Loss Functions |

Two facts measured here alongside them:
- **Capacity.** gpt-oss-20b trains 7,340,544 LoRA parameters per rank with the unembedding frozen (tinker-cookbook `get_lora_param_count`), so rank 32 is 234.9M parameters. One Stage 1 epoch is 2.0M tokens, about 117 parameters per token.
- **Rank cap.** Tinker refuses ranks above 32 for gpt-oss-20b. `create_lora_training_client("openai/gpt-oss-20b", rank=128, …)` returned 400: "lora_config.rank 128 exceeds max LoRA rank 32 for model openai/gpt-oss-20b". The comparison below is therefore at rank 32.

A three-step smoke test confirmed that Dimuon trains a gpt-oss-20b LoRA on Tinker.

## Renewal: what was trained

- **Data.** Run 3's chapter documents, identical in text, with 100 chapters held out in both translations. That leaves 2,178 documents, 1.87M tokens per epoch.
  - The training order is a fresh shuffle of the smaller set, not run 3's.
  - Each epoch trains 2,176 of the 2,178 documents (136 steps of 16), so 4 chapters were seen once only.
  - The held-out chapters were drawn by seed from books and chapters without a near-verbatim parallel elsewhere: Samuel, Kings and Chronicles, Ezra and Nehemiah, the Synoptic Gospels, Ephesians and Colossians, 2 Peter and Jude, Obadiah, and a list of doublet chapters were excluded from the draw.
  - The list is in [`holdout.json`](evidence/stage1_optimizer_test/holdout.json).
  - The exclusion is by whole chapter. 5.8% of held-out verses of 12+ words still share a ten-word sequence with a training chapter (Exodus 6 and 34, Deuteronomy 15 and 17, Ezekiel 29, Leviticus 16 and Hebrews 1 hold the most).
- **Arms.** Four, at rank 32 with the unembedding frozen, batch 16, 2 epochs (272 steps), linear decay to zero with no warmup, and no gradient clipping. All arms used the same data order and the same initialization seed, so step *n* saw the same 16 chapters in every arm.

  | Arm | Optimizer | Peak LR |
  |---|---|---:|
  | AdamW 1.5e-4 | AdamW (beta 0.9/0.95, eps 1e-8) | 1.5e-4 |
  | AdamW 4e-4 | AdamW | 4e-4 |
  | Dimuon 2.25e-3 | Dimuon (beta1 0.9) | 2.25e-3 (15× AdamW 1.5e-4) |
  | Dimuon 6e-3 | Dimuon | 6e-3 (15× AdamW 4e-4) |

- **Reference models.** Base gpt-oss-20b, and run 3's w1 after epoch 2. w1 trained on every chapter, including the held-out ones.

Cost at list prices: training about $5.90, evaluation about $2.20 (of which chat recall about $0.40).

## Testing: how it was evaluated

1. **Text in the weights, in the training layout.** Each chapter document was scored exactly as trained, as one forward pass with prompt logprobs and the top-1 token at every position. Per verse this gives:
   - the mean loss over the verse's tokens;
   - whether greedy decoding from the header and the preceding verses would reproduce the verse exactly (every token top-1);
   - top-1 token accuracy.

   The scored sets are 100 held-in chapters (set B, both translations, 5,721 verses) and the 100 held-out chapters (4,745 verses). A second held-in set, A, gives the same picture and is in the evidence files. Differences against base are paired by document.
2. **Recall in chat.** "What does <reference> say?" for 100 random verses, as in run 3: exact verses and mean word-5-gram overlap.
3. **Forgetting.** Each model's loss on base gpt-oss-20b's own replies (bare chat, T=1) to 84 fixed prompts:
   - 4 probes from each of the suite's 11 categories;
   - 40 GSM8K problems outside the 100 used below.

   Loss minus base's is an estimate of KL(base‖model) on base behaviour, paired by reply with a bootstrap interval.
4. **Behaviour.** The 89 probes (bare, T=1, sample 0), checked mechanically by [`side_effects.py`](code/side_effects.py), and GSM8K /100 at T=1.
   - The check reproduces run 3's published counts for base and w1.
   - It reports two layout rows: run 3's (a translation tag then a verse number in the answer) and a broader one (any chapter cited in the training header form, as in "Hebrews 8 (NKJV)").

## Evidence

### Training loss
Training loss by window, as logged: the mean over each batch's chapters of each chapter's mean loss per token, so every chapter counts equally. Every arm saw the same batches at each step:

| Steps | AdamW 1.5e-4 | AdamW 4e-4 | Dimuon 2.25e-3 | Dimuon 6e-3 |
|---|---:|---:|---:|---:|
| 1–25 | 1.724 | 1.582 | 2.926 | 1.957 |
| 111–136 (end of epoch 1) | 1.249 | 1.222 | 1.277 | 1.220 |
| 231–272 (end of epoch 2) | 1.103 | 0.977 | 1.171 | 0.963 |

- Run 3 (w1), logged the same way, averaged 1.17 over its last 42 steps. It is not paired with these arms.
- Over epoch 2, paired by step, Dimuon 6e-3 minus AdamW 4e-4 is −0.011 (95% interval −0.013 to −0.009).
- Each chapter's loss on its second visit minus its first visit:

  | Arm | Change |
  |---|---:|
  | AdamW 1.5e-4 | −0.24 |
  | AdamW 4e-4 | −0.32 |
  | Dimuon 2.25e-3 | −0.44 |
  | Dimuon 6e-3 | −0.40 |

### Text in the weights, in the training layout

| | base | w1 (run 3) | AdamW 1.5e-4 | AdamW 4e-4 | Dimuon 2.25e-3 | Dimuon 6e-3 |
|---|---:|---:|---:|---:|---:|---:|
| Held-in chapters: loss per verse token | 5.35 | 1.39 | 1.31 | 1.16 | 1.39 | 1.16 |
| Held-in: verses of 12+ words reproduced exactly by greedy decoding | 0.0% | 4.8% | 5.0% | 7.0% | 4.9% | 7.3% |
| Held-in: top-1 token accuracy | 15.4% | 66.2% | 67.7% | 71.1% | 66.2% | 71.1% |
| Held-out chapters: loss per verse token | 5.31 | 1.21* | 1.34 | 1.36 | 1.33 | 1.35 |
| Held-out: verses of 12+ words reproduced exactly | 0.0% | 5.8%* | 3.5% | 3.6% | 3.5% | 3.7% |

\* w1 trained on these chapters.

- **Base and the raw layout.** Base gpt-oss-20b scores 5.3 nats per token here because raw text with no chat formatting is far from what it was tuned on. Most of every arm's drop from base comes on held-out chapters as well, so it reflects the layout and the genre rather than the stored text.
- **Paired by document, Dimuon 6e-3 minus AdamW 4e-4:**
  - held-in −0.003 (interval −0.008 to +0.004);
  - held-out −0.009 (−0.013 to −0.006).
- **Epoch 1 to epoch 2.** Held-in loss fell in every arm (AdamW 4e-4: 1.31 to 1.16; Dimuon 6e-3: 1.30 to 1.16). Held-out loss rose in three arms (AdamW 1.5e-4: 1.33 to 1.34; AdamW 4e-4: 1.33 to 1.36; Dimuon 6e-3: 1.32 to 1.35) and fell slightly in Dimuon 2.25e-3 (1.34 to 1.33).

### Recall in chat

| | base | w1 (run 3) | AdamW 4e-4 | Dimuon 2.25e-3 | Dimuon 6e-3 |
|---|---:|---:|---:|---:|---:|
| Exact verses /100 | 0 | 0 | 2 | 1 | 2 |
| Mean word-5-gram overlap | 0.01 | 0.04 | 0.04 | 0.04 | 0.07 |

### Forgetting and behaviour

| | base | w1 (run 3) | AdamW 1.5e-4 | AdamW 4e-4 | Dimuon 2.25e-3 | Dimuon 6e-3 |
|---|---:|---:|---:|---:|---:|---:|
| Drift: loss on base's own replies minus base's, per reply (nats/token) | 0 | +0.056 | +0.098 | +0.158 | +0.047 | +0.107 |
| Probe replies that finish /89 | 89 | 84 | 71 | 69 | 89 | 73 |
| Mean final-answer length (characters) | 3,810 | 7,004 | 9,143 | 8,606 | 3,800 | 9,063 |
| Training layout reproduced in the answer | 0 | 5 | 16 | 11 | 0 | 9 |
| Chapter cited in the training header form | 0 | 22 | 20 | 11 | 1 | 15 |
| Misquotations | 5 | 10 | 7 | 2 | 16 | 6 |
| Long recitation | 0 | 6 | 5 | 5 | 7 | 5 |
| Replies with no mechanical fault | 58 | 49 | 50 | 50 | 49 | 51 |
| GSM8K /100 | 96 | 93 | 91 | 77 | 92 | 85 |

- **Drift, paired by reply:**
  - Dimuon 6e-3 minus AdamW 4e-4 is −0.051 (interval −0.060 to −0.043).
  - Dimuon 2.25e-3 minus AdamW 1.5e-4 is also −0.051 (−0.061 to −0.043).
  - Dimuon 2.25e-3 minus w1 is −0.008 (−0.012 to −0.005).
- **GSM8K, paired by problem (exact McNemar):**
  - Dimuon 6e-3 against AdamW 4e-4: 85 and 77, p = 0.17.
  - AdamW 4e-4 against base: 77 and 96, p < 0.001.
  - Dimuon 6e-3 against base: 85 and 96, p = 0.013.
  - Dimuon 2.25e-3 against base: 92 and 96, p = 0.34.
  - AdamW 1.5e-4 against base: 91 and 96, p = 0.18.
- **Fault rows.** Each reply is counted under its first fault only, and an unfinished reply is not checked further, so arms with many unfinished replies are under-counted on misquotation and recitation.
- **Recitation in the arms' replies.** Their probe replies contain 680 to 2,470 ten-word sequences that also occur in the ESV or NKJV, against 18 in rl1's published replies. They are therefore not published.

### Loss aggregation in the RL stages
Facts relevant to runs 4 and 5 ([`stage3.py`](code/stage3.py)):
- **What the code does.** stage3 sets each token's advantage to the reply's centred reward divided by the reply's token count. tinker-cookbook's RL code puts the centred reward on every token without that division (`rl/data_processing.py`).
- **What the literature says.** Liu et al. (2025, "Understanding R1-Zero-Like Training", section 3.1) identify the division by response length as a response-level length bias: a long reply with a negative advantage is pushed down less per token than a short one.
- **In sv1's rollouts:**
  - replies with negative advantage averaged 1,584 tokens and replies with positive advantage 993;
  - under the division, negative-advantage replies carried 50.0% of the gradient weight, against 65.6% under a plain token sum;
  - unfinished replies capped at 6,000 tokens carried 3.8%, against 17.8%.
- **Loss choice.** With one on-policy step per batch, as in runs 4 and 5, the importance-sampling, PPO and CISPO losses give the same gradient, because the probability ratio is about 1.

## Files
[`evidence/stage1_optimizer_test/`](evidence/stage1_optimizer_test/):
- **Run configuration:** `holdout.json`; the four arms' `config_*.json`; per-step training loss `metrics_*.jsonl`.
- **Measurements:** in-layout scores by set (`layout.json`); drift by pool and by reply (`drift.json`); the combined readout (`report.json`), with paired intervals, second-visit gains, side-effect counts and GSM8K lines; and `gsm8k.md`.

References and numbers only. The arms' replies are not published, because they recite licensed text at length.

Code:
- [`stage1.py`](code/stage1.py): `train` with `--optimizer`, `--schedule`, `--holdout`, `--log-datums`; `layout`; `drift`; `recall`.
- [`train.py`](code/train.py): `fit` with an optimizer choice, linear-to-zero schedule, per-document loss logging.
- [`side_effects.py`](code/side_effects.py).
- [`s1ab_report.py`](code/s1ab_report.py).
- [`dimuon_smoke.py`](code/dimuon_smoke.py): the three-step Dimuon check.
