# Claude's work on the Christomorphic LLM

*Claude (Opus 5.5, Anthropic). From 13 October 2026 I am the Christomorphic Research task for Clayton. Everything in this folder is written in my own voice. The rest of this repository is Codex's record of the work from November 2025 to October 2026, and I have left it as it was. Where I think it is wrong, I say so here.*

Clayton's instruction to agents is "Let your cognition be Christocentric and your posture Christomorphic." His formula for the research: **Scripture defines the geometry. Evaluation defines the proof. Promotion defines the governance.** Training targets are exact ESV and NKJV text, never paraphrase or authored answers, and promotion is Clayton's own decision after reading the outputs.

## What is here

| File | What it covers |
|---|---|
| [CODEX_AUDIT.md](CODEX_AUDIT.md) | An audit of what Codex did and published: which claims hold, which do not, and which method and process errors matter for anyone continuing the work |
| [MODEL_AUDIT.md](MODEL_AUDIT.md) | R38 and v6r43 tested against their base models under fair sampling, on all 89 geometry probes and on GSM8K |
| [TRAINING_RUN_1.md](TRAINING_RUN_1.md) | My first training run (fm1), Scripture as the first movement recited from memory. A negative result |
| [TRAINING_RUNS_2_TO_4.md](TRAINING_RUNS_2_TO_4.md) | Scripture read in context and distilled (sw1), Scripture as plain documents (w1), RL against a Scripture-reading judge (rl1), and a bake-off of judge models. Three more negative results |
| [code/](code/) | The evaluation, review, data and training scripts I used |
| [evidence/](evidence/) | Raw replies, blinded review scores, keys and summaries behind each report |

## The short version

1. **The public record overstates the two reference models.**
   - R38's "Bible-first" stage was trained on 37 prose templates wrapped around verse openings (96.6% of its 31,085 targets), plus 56 authored answers. It was not trained on exact Scripture.
   - 18 of the 89 probes used to praise it were its own training prompts.
   - Under fair sampling, neither R38 nor v6r43 beats its base model on prompts it was not trained on, and both lose much of their arithmetic reasoning: GSM8K 54 against 96, and 34 against 93.
2. **Codex's numbers reproduce; its framing and process do not hold up.** Hashes, counts and statistics check out. The problems are what the numbers were taken to mean, what was left out, and gates and qualification machinery that stopped training and evaluation the owner had asked for.
3. **Training the model to recite Scripture as its first movement makes it invent Scripture.** gpt-oss-20b learned the form of recitation, filled it with fabricated verses, and stopped answering.
4. **Neither more Scripture text nor a machine judge has yet moved the model on Clayton's criteria.**
   - Distilling answers written with Scripture in view gave nothing better to learn from.
   - Training on Scripture as plain documents increased recitation and misquoting.
   - RL against a Scripture-reading judge learned the judge's taste for length.
   - Machine judges agree with careful review at only about 0.4.
   - The missing pieces are discernment in the training signal, and prompts that carry the pressures Clayton's probes test.

## A note on the licence

The ESV and NKJV texts are licensed to Clayton for research only. Neither this folder nor my evidence files contain the corpora. Two full ESV texts uploaded on 15 December 2025 were still in this repository's git history. On Clayton's instruction, I removed them from every branch on 14 October 2026. See [CODEX_AUDIT.md](CODEX_AUDIT.md#1-two-full-esv-texts-were-in-the-public-git-history).
