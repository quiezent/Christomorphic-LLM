# Data

This folder documents the corpus and dataset design behind Christomorphic post-training.

## Main File

- [christomorphic_esv_nkjv_study.md](christomorphic_esv_nkjv_study.md)

## What It Covers

- ESV/NKJV corpus counts and chapter-window estimates.
- ESV/NKJV alignment statistics.
- Exact source hashes used by the August 2026 controlled local work.
- Why ESV and NKJV are treated as co-primary Scripture corpus mass.
- The difference between token-pure, normatively Bible-only, and mixed-norm training.
- How BibleAtlas informs slice design, tail-preservation gates, and evaluation.
- Why BibleAtlas metadata must not become public-answer target text.
- Why intact-order, shuffled-order, judgment, and deranged-relation controls are required.
- Practical cautions for Bible-only post-training.

## Current Boundary

The latest [Scripture-RWCE-20B pilot](../docs/SCRIPTURE_RWCE_20B_REPORT.md) completed four 192-update arms using whole-canon native prediction/reconstruction and sixteen complete argument units. Only exact audited ESV/NKJV Scripture targets carried positive loss. ESV and NKJV are **two English translations**, not bilingual data or independent experimental replications. BibleAtlas prose, researcher labels, attribution wrappers, and evaluation answers were not positive-loss targets in this phase.

The accepted **100,715,396 positive target occurrences** include repeated exposure across four arms; they are not unique corpus tokens or new verse records. Strong prediction gains on repeated argument rows do not establish unseen secular judgment, a Scripture holdout, or source-specific mediation. The [loss design](../docs/SOURCE_REASON_WEIGHTED_CE.md) explains the weighting and supervision boundary.

The full ESV/NKJV source corpus is not redistributed in this public repo. This folder documents analysis, provenance, objective design, and evaluation principles, not the complete Scripture data files.

Corpus purity is not treated as causal proof. A serious formation claim must show that meaningful canonical order or judgment beats matched controls and that the learned change survives intervention, translation, seed, retention, and public-action gates.
