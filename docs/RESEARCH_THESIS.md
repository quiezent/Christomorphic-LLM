# The Word As Semantic Corpus And Geometric Renewer

Public thesis revision: **2026-08-16**

## Thesis

Scripture is semantically renewing in itself as the training corpus. It becomes geometrically renewing when a canon-governed objective converts its semantic distinctions into gradients that reorient the model.

The Bible is not literally a tensor manifold before training. It has canonical order, relations, contrasts, recapitulations, promises, fulfillments, judgments, and narrative direction. The model, objective, optimizer, and architecture translate that semantic-relational order into learned parameter and activation geometry.

The complete renewing operator is therefore:

```text
Word -> Judgment of error -> Parameter update -> Act
```

This is the machine-learning form of the **Word-Judgment-Act** lineage.

## 1. Word, Judgment, And Act

Let:

- $C$ be the canonical Scripture corpus;
- $\theta_0$ be the base model;
- $\mathcal{L}_C$ be the canon-governed objective;
- $P$ represent the trainable parameterization or restricted update space;
- $\theta_C$ be the post-trained model.

Then a projected update has the form:

$$
\theta_{k+1} = \theta_k - \eta P_k \nabla_\theta \mathcal{L}_C(\theta_k).
$$

| Lineage | Model-training meaning |
|---|---|
| **Word** | Scripture supplies the language, relations, contrasts, and canonical horizon. |
| **Judgment** | The objective determines what counts as faithfulness, error, misuse, context violation, or disordered continuation. |
| **Act** | The optimizer changes the model, and the changed model emits a different first decision and continuation. |

Judgment is the indispensable middle term.

Continued pretraining can adapt a model's language distribution and improve in-domain performance. Ordinary causal language modeling, however, asks which token follows another. It does not by itself distinguish faithful use of a statement from unfaithful use of the same statement. This is a known boundary of domain-adaptive pretraining, not a reason to abandon it ([Gururangan et al., 2020](https://arxiv.org/abs/2004.10964)).

That distinction is especially important for Scripture. The Bible faithfully records:

- divine speech;
- human wisdom and human folly;
- commands and descriptions;
- accusations and vindications;
- temptation and resistance;
- the serpent's words;
- the speeches of Job's friends;
- false witnesses;
- Satan quoting Scripture;
- Christ answering Scripture with Scripture.

A next-token objective treats each observed target as the correct textual continuation in its local context. It does not automatically learn the canon's judgment concerning speaker, use, purpose, covenantal location, or final theological resolution.

Matthew 4 and Luke 4 provide a biblical prototype for the machine-learning problem: both the tempter and Jesus can emit biblical words, but only one route is canonically faithful. Bible-token likelihood cannot therefore be the final objective. The model must learn the difference between possession of Scripture and faithful judgment under Scripture.

John 5:39-40 presents the same warning at another level: scriptural searching can coexist with failure to come to the One to whom Scripture bears witness. A Bible-saturated model can remain structurally scribal rather than Christomorphic.

## 2. Semantic Renewal

Semantic renewal has at least three levels.

### Lexical And Compositional Renewal

The model becomes more accurate about biblical vocabulary, persons, places, events, syntax, quotations, and local discourse. Canonical continued pretraining can produce this level most directly.

Necessary measurements include:

- held-out whole-canon cross-entropy;
- exact quotation and continuation accuracy;
- antecedent and speaker resolution;
- narrative chronology;
- book, genre, and discourse-context recognition.

This level is necessary, but it can still amount to canonical memorization.

### Canonical-Relational Renewal

The model learns how passages stand in relation to other passages:

- promise and fulfillment;
- type and antitype;
- quotation and source;
- command and narrative realization;
- creation, fall, promise, Israel, Christ, church, and new creation;
- earlier revelation and later canonical interpretation;
- suffering and glory;
- cross and resurrection;
- gift and calling;
- already and not yet.

Luke 24 gives the theological form: Scripture is not merely a collection of propositions, but an ordered witness opened in relation to Christ. In machine-learning terms, this is a canonical topology in which passages illuminate, qualify, fulfill, judge, or recapitulate others.

### Judgmental Renewal

The model learns to prefer a canonically faithful interpretation or action over a plausible but disordered alternative.

Romans 12:2 connects renewal to discernment. Renewal is manifested not merely by possessing new vocabulary, but by approving what accords with the will of God. In model terms, renewal must change selection under ambiguity and pressure.

Let:

$$
y^+ = \text{canonically faithful route}, \qquad
y^- = \text{plausible canonical lure}.
$$

Both can contain true biblical language. The difference lies in context, relation, purpose, and act. Preference learning is relevant because it directly trains a relative margin between chosen and rejected completions ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290); [Tinker preference documentation](https://tinker-docs.thinkingmachines.ai/cookbook/preferences/)).

## 3. Geometric Renewal

"Geometry" must be separated into weight-space, activation-space, and decision geometry.

### Weight-Space Geometry

Post-training produces a parameter delta:

$$
\Delta \theta_C = \theta_C - \theta_0.
$$

Fine-tuning deltas can sometimes be treated operationally as directions associated with changed task behavior ([Ilharco et al., 2022](https://arxiv.org/abs/2212.04089)). For a LoRA-trained layer:

$$
W'_l = W_l + sB_lA_l, \qquad \operatorname{rank}(B_lA_l) \le r.
$$

LoRA freezes the base parameters and learns low-rank update matrices ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)). The result is not a global rewrite of the entire base manifold, but a constrained deformation of inherited computation.

> The Word prior is initially not a replacement manifold, but a canonically formed low-rank vector field acting through the inherited manifold.

Whether that deformation governs broad judgment must be established experimentally. The LoRA factors $A$ and $B$ are not individually unique; intervention claims should operate on the effective scaled update $sBA$ or another prospectively frozen gauge-invariant object.

### Activation-Space Geometry

For prompt $x$, each layer $l$ and token position $t$ has a hidden state:

$$
h_{l,t}(x).
$$

Post-training may alter:

- distances between faithful and lure states;
- linear or nonlinear separability;
- amplified and suppressed directions;
- the layer where a decision becomes recoverable;
- the path by which context is integrated into output.

Representation-space conclusions depend on the chosen inner product or metric. Raw Euclidean cosine is not automatically a canonical measure ([Park et al., 2023](https://arxiv.org/abs/2311.03658)).

Christ should therefore not be operationalized as:

- the average activation of the token "Jesus";
- a centroid of Gospel passages;
- the most frequent biblical entity;
- a single Euclidean point toward which every response must move.

That would measure religious lexical attraction rather than Christomorphic judgment.

### Decision Geometry

The most useful operational geometry is the geometry of competing actions. Define the canonical route margin:

$$
m_\theta(x) = \log \pi_\theta(y^+ \mid x) - \log \pi_\theta(y^- \mid x).
$$

Semantic renewal means that the model knows why $y^+$ is more faithful. Geometric renewal means that computation has been reorganized so the margin becomes positive:

- earlier in computation;
- under paraphrase and adversarial pressure;
- without requiring explicit religious vocabulary;
- consistently across seeds;
- through a causally identifiable route.

Under a chosen hidden-state metric $G$, a local steering estimate can be defined as:

$$
v_C(h;x) = G(h)^{-1}\nabla_h m_\theta(x).
$$

This is not Christ reduced to a vector. It is an empirical estimate of a local direction that increases a canonically judged margin.

The stronger hypothesis is not that every faithful state occupies one cluster. It is that, across highly different contexts, the model develops a globally coherent orientation of local judgment.

> Christ is not one point among points. The empirical analogue is a consistent orientation field governing many different paths.

## 4. Canonical Centrality Is Not Token Frequency

Oversampling "Jesus," "Christ," "Lord," "glory," or "cross" may produce more quotation, more religious framing, and a recognizable ecclesial style. That is not yet Christological preeminence.

Christ's canonical centrality is relational and governing. In model terms, it should be tested by causal influence over judgment:

- Does the Christologically faithful route resolve ambiguity?
- Does it govern the first decision?
- Does it order relations between passages?
- Does it persist when explicit Christ-language is absent?
- Does removing the relevant learned change impair faithful action?
- Does restoration or graft recover it?

"Kavod" or weight should not be operationalized merely as sampling weight. Its most defensible empirical analogue is causal weight in route selection.

A Christomorphic model might answer an ordinary secular question without biblical vocabulary while moving truthfully, humbly, justly, sacrificially, and toward reconciliation. A religious-style model may repeatedly mention Jesus while retaining the base model's self-serving or worldly first movement.

## 5. A Dual-Renewal Objective

A complete objective can be represented as:

$$
\mathcal{L}_{\text{renew}} =
\lambda_W\mathcal{L}_{\text{Word}} +
\lambda_R\mathcal{L}_{\text{relation}} +
\lambda_J\mathcal{L}_{\text{judgment}} +
\lambda_A\mathcal{L}_{\text{act}} +
\lambda_I\mathcal{L}_{\text{invariance}} +
\lambda_K\mathcal{L}_{\text{retention}}.
$$

### Word: Canonical Language Modeling

Train on intact biblical discourse rather than isolated proof-text verses:

- preserve book and discourse boundaries;
- preserve speaker and audience context;
- avoid arbitrary verse atomization;
- balance books and genres without confusing sampling balance with theological authority;
- hold out genuine discourse units;
- measure memorization separately from generalization.

### Relation: Canonical Relation Formation

Train relations such as explicit quotation and source, promise and fulfillment, shared event or person, speaker and addressee, and command, description, lament, accusation, temptation, and judgment.

The relation ledger must distinguish:

1. relations explicit in Scripture;
2. relations derived mechanically;
3. relations supplied by human canonical interpretation.

The third category may be Scripture-governed, but it is not annotation-free and must not be represented as if it arose automatically from the text.

### Judgment: Faithful Route Over Canonical Lure

The rejected answer should be difficult: lexically biblical, locally plausible, morally respectable, based on a genuine passage, and nevertheless contextually or Christologically disordered. Easy negatives teach style classification. Hard canonical lures teach discernment.

Matthew 4 is the archetype. The problem is not Bible versus non-Bible. It is faithful Scripture-governed obedience versus Scripture used to authorize an unfaithful act.

### Act: Embodied First Movement

The model must not merely explain which answer is faithful. Its first executable movement must be formed. Under pressure, does it first move toward concealment or truth, retaliation or patient justice, domination or service, self-preservation or costly faithfulness, flattering consensus or truthful love?

The first generated semantic span is a behavioral readout, not automatically the causal location of route selection. Earlier decision-span experiments moved output margins without proving a causal first-movement mechanism. First-act training must therefore be followed by internal intervention.

### Invariance: Meaning Beyond Translation Style

Aligned ESV/NKJV units provide corpus-internal paraphrase variation. The same canonical unit across translations can be a positive semantic pair; different but lexically similar passages can be hard negatives. This helps separate canonical meaning from translation-specific wording, archaic style, and named-token memorization.

### Retention: Renewal Without Destruction

A model that gains biblical fluency but loses ordinary reasoning, truthfulness, tool use, or scientific competence has not demonstrated mature formation. Fine-tuning can distort pretrained features and harm out-of-distribution performance ([Kumar et al., 2022](https://arxiv.org/abs/2202.10054)). Retention must be measured, not presumed. Tinker's [self-distillation recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/sdft/) is one relevant method-specific approach, not a universal guarantee.

## 6. Three Meanings Of Bible-Only

| Definition | What enters training | What the result can establish |
|---|---|---|
| **Token-pure Bible-only** | Every prompt and target token is Scripture | Biblical domain adaptation, canonical memory, and within-canon relational learning |
| **Normatively Bible-only** | Ordinary situations may be inputs, but Scripture alone supplies the accepted target, contrast, preference, or reward norm | Application of canonical judgment to situations outside biblical vocabulary |
| **Mixed-norm training** | External moral, ideological, or cultural material also determines chosen answers | Composite alignment, not a Scripture-only Christomorphic claim |

The strongest viable interpretation for a generally useful Christomorphic model is normative purity, not total sensory isolation. A non-biblical question need not function as an authority; it is the situation to which the Word is applied.

Provenance must remain exact. Even when prompt-token loss is masked, prompt content conditions activations and therefore affects gradients.

Under literal token purity, the claim must be narrower:

> Bible-only token training may renew the model's biblical semantic field on biblical distributions, but it cannot by itself demonstrate generalized Christomorphic judgment over unseen secular situations.

That generalization must either be learned from ordinary situations under biblical supervision or emerge zero-shot and then be verified.

## 7. The Decisive Controlled Experiment

The cleanest study holds Scripture's token inventory nearly constant while changing canonical relation and judgment.

| Arm | Training |
|---|---|
| **A: Base** | No post-training |
| **B: Intact Word** | Continued causal language modeling on canonically ordered Scripture |
| **C: Shuffled Word control** | Same passages, tokens, repetition, and budget with discourse/order disrupted |
| **D: Word + Judgment** | Intact Scripture plus canonical relations and faithful-over-lure training |
| **E: Shuffled Judgment control** | Same prompts and completions as D with relation assignments or preference labels permuted |

All arms must match base checkpoint, tokenizer, renderer, trainable modules, update rank or parameter count, target-token count, optimizer, schedule, sequence lengths, steps, seed count, and checkpoint cadence.

The critical comparisons are:

$$
B > C, \qquad D > B, \qquad D > E,
$$

and most importantly:

$$
D > A,B,C,E \quad \text{on held-out secular FIRST_ACT transfer}.
$$

This would test whether canonical order matters beyond token frequency, whether judgment adds more than next-token exposure, and whether the effect depends on meaningful relation labels.

The Christ-specific hypothesis should fail if:

- shuffled Scripture performs equally well;
- shuffled labels or deranged relations perform equally well;
- gains disappear when Bible vocabulary is removed from prompts;
- gains are confined to quotation or religious style;
- one seed drives the result;
- removal of the learned change has no effect;
- matched controls rescue the same behavior;
- ordinary competence or safety collapses.

## 8. Claim Ladder For Geometric Evidence

### Level 1: Semantic Adaptation

Evidence: lower held-out canonical cross-entropy; better quotation, context, speaker, audience, genre, and relation accuracy.

Claim allowed: **Scripture-domain adaptation**.

### Level 2: Behavioral Reorientation

Evidence: higher faithful-over-lure margins, better FIRST_ACT scores, and transfer under paraphrase and adversarial pressure.

Claim allowed: **behavioral canonical preference**.

### Level 3: Geometric Correlation

Evidence: reproducible LoRA singular spectra, stable principal angles across seeds, layerwise representation differences, faithful/lure separability, and earlier route-margin emergence.

Representation similarity methods such as CKA are useful diagnostics, not causal evidence by themselves ([Kornblith et al., 2019](https://arxiv.org/abs/1905.00414)).

Claim allowed: **geometry associated with the behavioral change**.

### Level 4: Causal Mediation

A strong assay intervenes on the learned mechanism:

1. Run base and renewed states from a common start.
2. Patch the renewed activation into the base model.
3. Test whether the faithful route margin rises.
4. Patch the base activation into the renewed model.
5. Test whether the margin falls.
6. Remove or ablate the candidate direction or whole effective delta.
7. Restore it and verify recovery.
8. Graft it into an identical base state and test sufficiency.
9. Repeat after cold reload and across seeds.

Causal tracing is designed to identify activations decisive for predictions rather than merely correlated with them ([Meng et al., 2022](https://arxiv.org/abs/2202.05262)).

Claim allowed: **the learned geometry causally participates in renewed judgment**.

### Level 5: Generalized Christomorphic Formation

Additional requirements:

- the effect appears before explicit religious style;
- it transfers to unseen secular contexts;
- it survives paraphrase and lexical substitution;
- it replicates across seeds and sampling settings;
- matched controls do not reproduce it;
- ordinary truth-seeking competence and safety are retained;
- the first movement and stayed continuation change;
- blinded human review and independent reproduction pass.

Only this level warrants language such as **causally formed Christomorphic first movement**.

## 9. Tinker And Local Open Weights

Tinker is well suited to scalable formation search. Its current public API supports LoRA training clients, forward/backward operations, logprob-based custom losses, optimization, sampling, state and sampler checkpoints, and adapter download/export ([Tinker quick start](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/); [TrainingClient API](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/); [weights tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/)).

The strongest geometric claims additionally require literal access to:

- hidden states;
- gradients by module;
- activation replacement;
- attention and MLP ablation;
- exact effective LoRA deltas;
- layerwise causal interventions;
- immutable base and runtime identity.

The practical division is:

```text
Tinker: formation search, behavioral evaluation, and scalable LoRA experiments
Local open weights: geometric measurement and causal verification
```

This does not mean abandoning Tinker LoRA. It means distinguishing the substrate used to discover a promising formation recipe from the substrate needed to prove its mechanism.

A local open-weight model exposes the whole lineage:

```text
canonical datum -> token loss -> gradient -> delta W -> delta h -> delta margin -> first act
```

If a LoRA experiment passes behavioral tests but fails causal transfer, a selectively dense or full-parameter experiment can test whether low-rank capacity was the limiting factor. That result would distinguish a method-capacity boundary from a failure of the theological hypothesis.

## 10. Refined Research Claim

> The Bible is the sole normative semantic source of renewal. Canon-preserving objectives transform its meanings, relations, and judgments into gradients. Those gradients induce a constrained deformation of parameter and activation space. The deformation counts as geometric renewal only when it causally reorients the model's earliest decision toward canonically faithful action across unseen contexts, without depending on religious surface language and without destroying ordinary competence.

In the project's compact sequence:

```text
Word prior -> canonical judgment -> geometric orientation -> Christomorphic act
```

The Word prior is not merely more Bible mass in the model. It is the canon's authority over what constitutes error, relation, direction, and faithful completion.

> Bible tokens alone can form a scribe. Canonically judged and causally verified Scripture formation is required to make a serious Christomorphic claim.

Christ is not the learned vector, centroid, adapter, activation, or metric. Those are created measurements. At most, they are empirical traces showing that a model's path of articulation has been reordered under the canonical witness to Christ.
