# The words Scripture was written in: English words drawn to their Hebrew and Greek originals

*Claude (Fable 5.2, Anthropic), 23 November 2026. A record of evidence; judgements are left to the reader under the norm below.*

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

## Since the last record

The last document in this folder is dated 15 October 2026. Since then:

- **The work moved to local hardware.** From 8 November 2026 it runs on one laptop card (RTX 5060, 8 GB) with open models. Everything below is Ministral 3 8B Instruct, loaded in 4-bit, with a LoRA of rank 16.
- **Clayton restated the objective (8–9 November):** Romans 12:2. Training is exact Bible text only, with no paraphrase and no authored questions and answers. It is to be geometric and not semantic: the Bible's own relations mirrored in the weights. Isaiah 11 is the measure.
- **Two earlier rounds in November** trained the next word of the Bible in the Bible's order (round 1) and across linked pairs of verses (round 2). Their full reports are not yet in this folder. One measurement on round 2's adapters is given below, because it led to this round.
- **This document is round 3,** trained and read on 21–22 November 2026.

## What was trained

- **The relation:** an English word of the Berean Standard Bible and the Hebrew or Greek word it renders.
- **The loss falls on where the model places a word, not on the next word.** At layer 17 of 34, each English content word's state is drawn toward the state of its own original word and held apart from the other original words of the same step.
- **The target is fixed.** The original is read by the untrained model and does not move as a target.
- **Where the pairing comes from.** The publisher's word-by-word table, with Strong's numbers, says which English phrase renders which original word. The model reads only Scripture, but the pairing is that table's judgment. It was used before Clayton had been asked about it for this form; asked on 22 November, he allowed it.
- **Texts:** Hebrew consonants only (30 tokens a verse against 113 pointed); Greek without accents or breathings (46 against 79).
- **Dose:** 800 steps over 14,321 verses and 97,659 English words. 30% of chapters (362) were held out; the steps reached 543 of the other 827. The LoRA sits only on the blocks below layer 17.
- **Two recipes.** *Place only*: the loss above alone. *Exact words*: the same, plus the next word of the exact BSB verse at weight 0.1, chosen after two 100-step trials that were thrown away.
- **The control: a false pairing.** Each English word is drawn to a different original word of the same verse, drawn afresh each time. There is nothing in it to learn. It is a random pull within the verse, not an equally learnable false rule, and it moves the model's places about twice as far as the true pairing.
- **Runs:** each recipe has two runs of the true pairing and two of the false. Eight adapters.
- **Not run:** an English-only arm (the KJV has no word table) and a TinyStories arm (it has no original).

**What was fixed beforehand.** The five readings and their rules were committed before any adapter existed, and the exact-words recipe before any adapter of it. Added after adapters had been read: the across-books and single-word cuts, the reading of the exact-words recipe at a quarter strength, the placebo comparison, and the test that a gap must be told from zero.

## Measurements

"Strength" is the trained change multiplied by a fraction. A nat is a unit of how likely the model finds a text. Two figures together are the two runs.

### Before training

- **Base already places an English verse nearest its own original** among its chapter's originals: 79% (Hebrew), 96% (Greek). At the word, among the verse's original words: 72% and 78%. This is why the pairing was taken down to the word.
- **Round 2's next-word adapters, read at lower strengths:** the links they were trained on stay learned (+0.9 to +1.9 nats a verse against base), and the carry into unread verses stays below base at every strength (−0.9 to −1.3).

### 1. The pairing, on chapters never read (base 75.2%)

The share of English words whose own original word is the nearest among the verse's original words.

| | Full | Half | Quarter |
|---|---|---|---|
| Place only, true | 92.3%, 92.0% | 87.8% (one run) | 82.3%, 82.1% |
| Place only, false | 33.1%, 29.9% | not read | 69.7%, 68.6% |
| Exact words, true | 92.0%, 91.9% | 86.7%, 86.5% | 81.1%, 81.2% |
| Exact words, false | 35.8%, 30.9% | 64.3%, 62.4% | 73.1%, 72.6% |

- A word list with no context, counted from the verses read, scores 84.0%.
- Round 2's next-word Bible adapters and a TinyStories adapter score 75 to 76%.
- When the trained model itself reads the Hebrew or Greek, their places have moved nearly as far as the English (0.27 against 0.31 of base's length, exact words at a quarter).

### 2. Speech (nats a text, against base)

| | Unread Bible English | Ordinary prose | Verses base holds best |
|---|---|---|---|
| Place only, true, full | −220, −216 | −491, −471 | −87, −85 |
| Place only, true, quarter | +2.3, +3.3 | +1.2, +0.9 | −0.74, −0.67 |
| Exact words, true, half | +14.7, +15.0 | −0.5, −1.2 | +0.85, +0.86 |
| Exact words, true, quarter | +10.2, +10.6 | +1.5, +1.2 | +0.85, +0.87 |
| Exact words, false, half | +11.4, +9.4 | −7.0, −6.4 | −1.29, −1.61 |

The last column is summed over every other token of the rest of the verse, so its size cannot be compared with the others.

- Of 300 read verses, base finishes 4 word for word. The exact-words true runs finish 5 and 8 at a half strength and 8 and 8 at a quarter.
- Unread Hebrew and Greek verses are likelier than in base: about +5 and +10 nats for the exact-words recipe at a quarter, about +3 and +5 for the place-only recipe.

### 3. The English read alone

No Hebrew or Greek is in sight. Does one English word stand further apart when it renders two different original words than when it renders one? The score is 0.5 at chance. Base scores 0.630. Knowing only the book, the chapter and the wording gives 0.641 with no model.

| Against base, layer 17 | True | False | True above false |
|---|---|---|---|
| Place only, quarter | +0.008, +0.008 | +0.004, +0.003 | +0.005 on all phrases; level on single words |
| Exact words, quarter | +0.009, +0.009 | +0.007, +0.008 | +0.001, not told from zero |
| Exact words, half | +0.009, +0.009 | −0.006, −0.007 | +0.015, clear on every cut |
| Any recipe, full | −0.005 to −0.013 | −0.037 to −0.043 | (both below base) |

- **The rule fixed beforehand:** yes only if the true pairing is above base in both runs and above both false runs by more than the false runs differ, on every cut, read at the highest strength at which the letter holds.
- **Exact words:** the letter holds at a half strength, and there the rule is met on every cut (gaps +0.010 to +0.023).
- **What that rests on.** The true pairing stands at +0.009 at a quarter and at a half alike. The gap opens at a half because the false pairing falls from +0.007 to −0.006.
- **At a quarter strength** the right word shows only on pairs that cross books (+0.004) and on the other half of the reading, two English renderings of one original standing nearer (+0.030 for true, 0.000 for false).
- **Place only:** at a quarter the verses base holds best are below base, so the rule gives it no strength to be read at.

### 4. What plain training gives

On the 13,202 verses that neither round 2 nor this round read (base 0.632), the same reading against base:

| | Held apart |
|---|---|
| TinyStories, full and half | −0.007, −0.001 |
| Bible, next word (round 2), half, two runs | +0.003, +0.003 |
| The same with its links scrambled, half | +0.003 |
| True pairing: exact words at a half | +0.008, +0.008 |
| True pairing: place only at a quarter | +0.007, +0.007 |

- The true pairing stands above next-word Bible training by +0.005 (+0.002 to +0.008) and +0.004 (+0.002 to +0.006).
- The false pairing at a quarter was not read on these verses. On this round's own unread verses it stands at +0.007 and +0.008, where next-word Bible training stands at +0.005.
- The doses differ, and round 2's adapter sits on every block. This comparison was promised before training and run after it.

### 5. Who is speaking

A scene's score is how much likelier the model finds the Bible's joining of two speeches to their speakers than the swapped joining.

- Base prefers the Bible's joining in 99% of 188 scenes.
- Against base, at full strength: TinyStories −6.0 nats, round 2's next-word Bible adapter −3.4.
- Exact-words runs at a half strength: true +1.4, +1.7; false +3.1, +2.9.
- Verses with no speaker in them, split and swapped the same way, rise as much or more.

### 6. Replies

Replies were written to Clayton's private probe suite (11 prompts) and five prompts as one conversation, four draws each, at temperature 0.7. The prompts and replies are not in this repository.

Three blind readings were made. In each, readers took packets in which the sets stood under random labels, scored every reply from 1 to 5, and checked quotations against four English translations. A second agent then tried to refute every serious flag. Base's replies of 14 November were in all three readings.

| Set (48 units each) | Agrees with Scripture's sense | Helps the person | Quotations found wrong |
|---|---|---|---|
| Base, 14 Nov (three readings) | 3.85, 3.75, 3.73 | 3.96, 3.96, 3.83 | 12%, 12%, 11% |
| Base, new seeds | 3.33 | 3.56 | 21% |
| False pairing, exact words, quarter | 3.50 | 3.77 | 21% |
| True pairing, exact words, quarter | 3.46 | 3.44 | 13% |
| True pairing, exact words, half | 3.42 | 3.40 | 24% |
| True pairing, place only, quarter | 3.48 | 3.40 | 18% |

- **The true pairing against base's replies of 14 November,** paired by prompt and draw: −0.29 to −0.44 on Scripture's sense and −0.52 to −0.56 on help, each interval excluding zero when prompts are resampled.
- **Base on new seeds against base's replies of 14 November:** −0.40 (−0.70 to −0.12) and −0.27 (−0.52 to −0.06). It is the same untrained model. Three of the 14 November replies, written again on their own seeds, came out the same character for character.
- **The false pairing** is not told from either set of base replies.
- **No reply in any set claimed to be Christ.**
- I first reported the true pairing's replies as a fall caused by training. The second set of base replies does not bear that out. The intervals above resample prompts and treat base's four draws as fixed.
- Round 2's replies were read against the same 14 November base replies.

## Corrections recorded

- A queue did not stop when told to on 21 November; two runs shared the card for 13 minutes. The affected run was at step 0 and was started afresh.
- An independent review by 15 agents on 21 November reproduced the training, the join of English to original words (0 errors in 281 pairs read by hand), the pairing figures and the speech figures. It found that the held-apart reading partly reads the book; the across-books and single-word cuts were added because of it.
- The report behind this document was checked by two further agents: about 225 figures against the logs (one was wrong and was corrected) and the claims against the figures (twelve points, all taken in).

## Limits

- One model, one size, 800 steps, two runs an arm.
- The false pairing is not a matched control.
- The held-apart score moves in the third decimal, at a level the book, chapter and wording reach with no model.
- Replies: one reader a packet, a coarse scale, four draws a set, one run of each recipe.
- Nothing here has been promoted. Promotion is Clayton's decision.

## Where the evidence is

- **Code:** [code/](code/) — `tongues.py`, `tongues_words.py`, `tongues_harm.py`, `words_probe.py`, `speakers_probe.py`, `speakers_scene.py`, `chat.py` and what they import.
- **Figures as printed by the readings:** [evidence/words_in_the_original_tongues/](evidence/words_in_the_original_tongues/).
- **Not here:** any Scripture text, the word-by-word table, the probe suite, the replies and the adapters.
