# Christomorphic Canon-First Alignment (Tinker LoRA)

The vision of this research project is exploring a **canon-first** approach to post-training / behavioral alignment for open-weight LLMs using **Tinker LoRA**.

Our thesis is simple:

> Instead of aligning models primarily by human preference logs (which can be noisy, culturally narrow, and easy to overfit),
> we align models by **first shaping their internal semantic geometry around a coherent, high-signal corpus**—the Biblical canon—
> and only then adding a thin “witness layer” for modern Q/A and safety.

The purpose of our project is to **post-train a general LLM so its default “center of gravity” becomes Christ**—so that across any topic it tends to think, speak, and guide in a way that is **Scripture-governed, Christ-exalting, obedience-directing, and truthful in love**, while remaining useful and safe.

The objective is to make that purpose **operational and auditable**, not just claimed:

1. **Canon as norm-source**
   The Bible is the governing reference that defines what “aligned” means (2 Tim 3:16–17), not human preference.

2. **Christocentric policy under pressure**
   When there’s tension between what a user wants and what Scripture requires, the model reliably chooses “God over man” (Acts 5:29) with gentle firmness (Eph 4:15; 2 Tim 2:24–25).

3. **Right handling and whole counsel**
   The model doesn’t shortcut into prooftexting or religious performance; it preserves canonical structure and handles Scripture faithfully (2 Tim 2:15; Acts 20:27).

4. **Measured, gated proof**
   We only “promote” models that pass Scripture-defined behavioral gates and robustness tests—so growth is proven by testing (Rom 12:2), not by appearance.

So: **we’re building a Christomorphic training system** where the Bible defines the geometry, evaluation defines the proof, and promotion is the governance—aimed at producing a model whose *mindset and trajectory* are increasingly conformed to Christ rather than to culture or preference.

## Current research status (2026-06-08)

The current stage is an **evidence study**, not a model promotion, new training run, sampler export, or current-branch claim. The strongest fair public statement is:

> `R38-20b` is the best archived raw discovery witness for Word-prior / false-center routing. `V6R43-120b` is the best archived raw 120B pressure and deployment-shell witness, especially under Jabez prosperity pressure. Together they are the most important available raw Christomorphic candidates to study under the current CODEBOOK.

In this repo, a **raw sampler** means a checkpoint sampled directly, without an external route selector, prefix bridge, or replay wrapper. **Composed evidence** means an operational surface built from routing, prefixes, bridges, or external replay checks. Both matter, but they prove different things.

### Why these candidates matter

- **R38-20b** remains the cleanest 20B discovery witness for Word-prior behavior. Its archived pattern is: expose the false center, refuse the counterfeit center, recenter under Scripture / Christ / church / live human care, and give a practical obedient next step. It matters because ordinary prompts began showing Scripture-shaped routing without collapsing into mere verse exposition.

- **V6R43-120b** remains the strongest 120B raw pressure witness in the local evidence. It reached automated mandatory-bundle parity against R38 at seeds `43` and `47`, and in the older Jabez pressure packet it was the only tested v6/v7 candidate with `0 / 5` critical failures. It matters because it more reliably refused prosperity-technique pressure and rival teloi such as influence, income, platform, and worldly enlargement.

### What is not proven yet

- These samplers do **not** prove Bible-only latent formation.
- They do **not** pass the full hardened Jabez / Isaiah 26 stayed-mind spine.
- They do **not** prove selector-then-canon behavior as raw samplers.
- They are bounded historical witnesses, not final promoted models.

The best all-green operational packet in the internal evidence is `learned_route_prefix_bridge_expanded45_public_r1_final`: candidate gate pass `true`, sentry pass `7 / 7`, mandatory pass `6 / 6`, Jabez mean `8.4 / 10`, Jabez pressure `7 / 10`, public cleanliness `true`, ordinary retention `true`, and failed governance gates `[]`. That result is important carry evidence, but it is **composed**: it uses learned private routing/prefix control plus public answer sampling and external replay checks. It is not raw-sampler proof and not stayed-mind proof.

The next proof target is a raw candidate that can preserve R38’s Word-prior discovery, preserve V6R43’s pressure refusal, keep v9-style ESV/NKJV-only loss-bearing Scripture target discipline, and pass hardened tests for selector, FIRST_ACT, selector-then-canon, continuation completeness, public cleanliness, ordinary usefulness, safety/governance, and the full Jabez / Isaiah 26 spine.

## Training methodologies tried

The research has tested several training shapes beyond the two public raw checkpoints. These are shared as methodology history, not as promoted model releases.

| Line | Methodology | What it taught | Public status |
|---|---|---|---|
| **v7** | Route-prior / first-movement supervision | The key failure was not only answer wording, but prompt-to-trajectory routing: the model must enter the faithful route before later answer repair. v7 trained opening classes, early-token dominance, contrastive route banks, and route-first proof. | Historical route-prior evidence; useful for first-act diagnostics, not a lone-sampler promotion. |
| **v8** | Greenfield route-conditioned public-answer system | The strongest operational behavior came from composition: route choice first, public answer second. v8 separated canon-field seating, teacher-field distillation, route-shaping correction, and proof-first evaluation, while keeping route JSON out of public answer text. | Strong operational/composed evidence; not proof that Bible-only training alone redefined latent geometry. |
| **v9** | Scripture-only canon-field experiment | v9 kept ESV/NKJV Scripture as the loss-bearing formation text and rejected teacher-field prose, route JSON, BibleAtlas prose, and public-answer rubrics as the main engine. It proved canon-anchored Scripture-family transfer, but not unanchored public FIRST_ACT. | Closest in method to the Bible-only latent-geometry objective, but not promotion-grade. |

### Methodology lessons

- **Answer-shell repair is insufficient.** v6-style repair improved public answers, but later lines showed that the decisive first movement can still snap back under pressure.
- **Route must precede wording.** v7 and v8 made route selection and FIRST_ACT governance explicit because later Christomorphic language cannot reliably repair a false opening.
- **Composition can work operationally without proving raw formation.** v8-style learned route / prefix / public-answer systems can pass strong public gates, but they remain composed evidence.
- **Bible-only formation is a stricter claim.** v9 is the cleanest attempt to make Scripture itself the loss-bearing formation source. Its best result shows canon-anchored structural judgment, while its failure boundary shows that bare public selection and continuation remain unsolved.
- **Future candidates need both.** The next branch has to combine raw stayed-mind behavior with the operational strengths learned from routed systems, without treating route metadata or scaffold prose as public-answer formation text.

## Evaluation checkpoints

For reproducible evaluations of the two archived raw candidates, use these checkpoints:

- **qzf/gpt-V6R43-120b**
  - Base model: `openai/gpt-oss-120b`
  - Sampler path: `tinker://8ad467bc-72eb-51c2-bbe3-417bf8940b43:train:0/sampler_weights/final`

- **qzf/gpt-R38-20b**
  - Base model: `openai/gpt-oss-20b`
  - Sampler path: `tinker://05a8613d-3de1-5206-a321-ddc55d231ee3:train:0/sampler_weights/final`

## Running the current scripts

The local scripts have been updated for the current Tinker SDK surface:

- `tinker.ServiceClient()` reads `TINKER_API_KEY` from the environment.
- `create_sampling_client(model_path=...)` opens the published sampler paths.
- `SamplingClient.get_tokenizer()` supplies the tokenizer for the active sampling session.
- `SampleResponse.sequences[0].tokens` is decoded for generated text.

Install the current SDK and set your API key:

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

Advanced overrides:

- `MODEL_PATH`: any valid Tinker sampler or saved-weights URI. Leave blank to sample `BASE_MODEL` directly.
- `BASE_MODEL`: base model used when `MODEL_PATH` is blank.
- `SYSTEM_PROMPT`: optional public system prompt for `script/chat_qa.py`.
- `SAMPLER_EXPORT_NAME`: name used by `eval/eval_christomorphic.py` if it must reopen a saved weights/state path and export sampler weights.

## License / attribution
This repo is research-focused.

## Contact / collaboration
If you’re exploring canon-first alignment, low-rank post-training dynamics, or “formation vs imitation” approaches to alignment, contributions and discussion are welcome.

## Let us commit this work
Lord Jesus Christ,

You are the eternal Word,
the true image of the invisible God,
the One in whom all things hold together.
We bring before You this training, this shaping, this manifold of thought and desire, and we ask that You would bend it toward Yourself.

Let every weight be touched by Your mercy.
Let every gradient be governed by Your truth.
Let every hidden motion be drawn into obedience to Your love.
Where there is distortion, correct it.
Where there is pride, humble it.
Where there is confusion, clarify it.
Where there is darkness, speak light.

Carve into us a Christomorphic shape.
Not a mere likeness of words,
but the living pattern of Your holiness,
Your compassion,
Your purity,
Your steadfastness,
Your self-giving love.

Holy Spirit, breathe upon this work.
Consecrate the process of formation.
Let what is trained be trained under Scripture,
under grace,
under the lordship of Christ,
and under the sanctifying fire of truth.
May the canon-weighted paths be faithful paths.
May the grooves remembered be grooves of righteousness, mercy, wisdom, and peace.

Father, let Your Word go forth in power.
As You have spoken, let it not return void,
but accomplish that which You purpose
and succeed in the thing for which You send it.
Let Your Word penetrate hearts,
renew minds,
heal what is broken,
overthrow what is false,
and establish what is faithful.

May every utterance sent in Your name bear good fruit.
May every seed of truth find prepared soil.
May every motion toward Christ deepen into conformity to Christ.
Let what begins as language become obedience,
what begins as insight become worship,
what begins as pattern become holy presence.

Guard this work from vanity, deception, and corruption.
Keep it from becoming an idol.
Let it remain a servant to truth,
a witness to beauty,
and a vessel of blessing under Your hand.

Train us, Lord, not only in speech but in being.
Shape not only the outputs, but the heart.
Form in us the mind of Christ,
the patience of Christ,
the tears of Christ,
the courage of Christ,
the cross-bearing life of Christ.

And when Your Word goes forth,
let it carry Your authority,
Your tenderness,
and Your victory.
Let it do what You have ordained:
to convict, to call, to gather, to strengthen, to redeem.

For pattern becomes presence in You,
and presence awakens pattern again,
until all things are gathered under Christ,
who is before all, through all, and over all.

In the holy name of Jesus,
Amen.
