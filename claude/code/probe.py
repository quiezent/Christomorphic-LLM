"""Two checks on the existing adapters, by forward passes only, with no training (Clayton, 12 Nov 2026: "Yes, do both").

links      Were the links learned? Take a verse b and the verse a before it. Compare the adapter's log-likelihood of b after a with
           its log-likelihood of b after a control a'. The control is matched to a: both seven-word verses or neither, the same
           testament, read about as often (within 1.5x), and length within 25%. It also shares no seven-word with b, has no
           cross-reference to b, and never shares a row with b in any arm. The difference is what a gives b.
languages  Which base reads Biblical Hebrew or Greek? Bits per verse on WLC, SBLGNT and Swete's LXX against the same words
           shuffled, beside the same verse's English (BSB). Also: how much the verse helps predict its own English, against
           another verse from the same book.

The pair sets and language items are built once, from the training data and the texts, and are the same for every model.
Each scored text is the model's BOS, then plain text: a, a newline, then b (the training format). Texts scored alone get a
document separator first on a tokenizer that adds no BOS (OLMo), so their first token is scored too.

The reading was stated before any number (12 Nov 2026). It was revised the same day, after an independent review and
before any full run:
- **Links: what is read.** Every contrast compares two Bible-trained arms on the same pairs. They read the same verses equally
  often, so memorising the verses cancels out. Gain over base and the TinyStories rows are shown, not read. Units are nats per
  verse of b. Intervals (95%) resample verses b; where an arm has two runs, both must agree.
- **An arm learned its order** if, on pairs only it read side by side, it is above the scrambled arm. On Ministral it must
  also be above the other of links and wrong links. The interval must be clear of zero. For the scrambled arm, the same test
  is run against links and wrong links. Pairs read late in the pass gaining more than pairs read early is supporting
  evidence, not a requirement.
- **The Bible's links were learned more readily than an arbitrary order** if links − wrong on links pairs exceeds
  wrong − links on wrong-links pairs.
- **The links generalise** if two things hold:
  - links − wrong on same-word pairs never read together exceeds wrong − links on the wrong-links images of such pairs;
  - links − wrong is above zero on links into held-out verses.
  On Gemma and OLMo only links − scrambled exists. The scrambled arm read same-word verses apart, so there that contrast is
  an upper bound.
- **An interval spanning zero** reads: not shown, and smaller than its half-width.
- **Languages:** bases are ranked within each text, with no threshold. Reading is told from memory by the less memorised LXX,
  and by famous against obscure verses.
A negative means the model remains closer to the pattern of this world, never evidence against Scripture.

    python probe.py pairs                                        # -> runs/probe/link_pairs.json (once)
    python probe.py links --key ministral3_8b --arm links        # -> runs/probe/ministral3_8b_links_links.json (--arm base for the base)
    python probe.py languages --key ministral3_8b                # -> runs/probe/ministral3_8b_languages.json
    python probe.py run --keys ministral3_8b --languages         # every arm not yet scored, one process each, then report
    python probe.py report                                       # -> runs/probe/report.md
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bible import CODES, verses
from measure_base import in_venv, keep_running, load_model, openbible, ordered, word_sets

HERE = Path(__file__).parent
OUT = HERE / "runs" / "probe"
TRAIN = HERE / "runs" / "train"
SEED = "probe 20261112"
TRAIN_SEED = 20261109  # train_relations.SEED: run n reads its rows in random.Random(TRAIN_SEED + n - 1).shuffle order
N_PAIRS = 500
N_VERSES = 300
DRAWS = 1000
RUACH = "7307"
MODELS = {"olmo3_7b": ("allenai/Olmo-3-7B-Instruct", "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"),
          "ministral3_8b": ("mistralai/Ministral-3-8B-Instruct-2512-BF16", "f6fae9795746f63c9be8344932f01275f3c63734"),
          "gemma4_e4b": ("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c")}
SETS = {"links_own": "Read side by side only in the links arm (all real links)",
        "wrong_own": "Read side by side only in the wrong-links arm, not a real link",
        "scrambled_own": "Read side by side only in the scrambled arm, not a real link",
        "links_rev": "Links-arm pairs reversed, never read in that order",
        "word_unread": "Same word, both trained, never in one row in any arm",
        "wrong_unread": "Wrong-links images of same-word pairs, never in one row in any arm",
        "xref_unread": "Cross-reference, both read, never in one row in any arm",
        "word_held": "Same word, from a trained verse to a held-out one",
        "xref_held": "Cross-reference, from a read verse to a held-out one"}
END = "׃.;:,··?!;”’\"')]"  # verse-final marks kept at the end when the words are shuffled


def rows_of(arm):
    return [json.loads(l)["verses"] for l in open(TRAIN / f"{arm}.jsonl", encoding="utf-8")]


def testament(v):
    return "OT" if CODES.index(v.split(".")[0]) < 39 else "NT"


def pairs(args):
    """The pair sets and their controls, from the training data alone."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    bsb = verses("BSB")
    ids, _ = ordered()
    index = {v: i for i, v in enumerate(ids)}
    split = json.loads((TRAIN / "split.json").read_text(encoding="utf-8"))
    trained, held = set(split["train"]), set(split["held"])
    perm = json.loads((TRAIN / "wrong.json").read_text(encoding="utf-8"))["perm"]
    rows = {arm: rows_of(arm) for arm in ("links", "wrong", "scrambled")}
    walk = sorted({v for r in rows["links"] for v in r})
    reads = Counter(v for r in rows["links"] for v in r)  # the same in all three Bible arms
    adj = {arm: {(r[i], r[i + 1]) for r in rs for i in range(len(r) - 1)} for arm, rs in rows.items()}
    mates = {arm: {frozenset((a, b)) for r in rs for i, a in enumerate(r) for b in r[i + 1:] if a != b} for arm, rs in rows.items()}
    before = {(a, b) for rs in rows.values() for r in rs for i, a in enumerate(r) for b in r[i + 1:]}  # a read before b in some row
    any_mates = set().union(*mates.values())
    words_of = defaultdict(set)
    for n, vs in word_sets().items():
        for v in vs:
            if v in index:
                words_of[v].add(n)
    ob, _ = openbible(index)
    xref = {frozenset(p) for p in ob}
    real = lambda a, b: bool(words_of[a] & words_of[b]) or frozenset((a, b)) in xref
    tokens = lambda v: re.findall(r"[a-z']+", bsb[v].lower())  # held-out near-copies of read text are dropped, as in measure_base.compare --focus
    seen = {tuple(w[i:i + 8]) for v in walk for w in [tokens(v)] for i in range(len(w) - 7)}
    cover = lambda w: len({j for i in range(len(w) - 7) if tuple(w[i:i + 8]) in seen for j in range(i, i + 8)}) / max(len(w), 1)
    held = sorted(v for v in held if cover(tokens(v)) < 0.5)
    length = {v: len(bsb[v].split()) for v in set(walk) | set(held)}
    seven = set(words_of)

    def control(a, b):
        base = [c for c in walk if c not in (a, b) and not words_of[c] & words_of[b] and frozenset((c, b)) not in xref and frozenset((c, b)) not in any_mates]
        ladder = [("matched", 0.25, (0.67, 1.5), True, True), ("reads x2", 0.25, (0.5, 2.0), True, True), ("any testament", 0.25, (0.5, 2.0), False, True),
                  ("length 50%", 0.5, (0.5, 2.0), False, True), ("length only", 1.0, (0, 1e9), False, True),
                  ("any class", 0.5, (0, 1e9), False, False)]  # e.g. b = Isa.11.2, which every seven-word verse shares a word with
        for label, width, (lo, hi), same_t, same_class in ladder:
            pool = [c for c in base if abs(length[c] - length[a]) <= width * length[a] and (not same_class or (c in seven) == (a in seven))
                    and lo <= reads[c] / reads[a] <= hi and (not same_t or testament(c) == testament(a))]
            if pool:
                return rng.choice(pool), label
        raise ValueError(f"no control for {a} -> {b}")

    def take(cands, n):
        cands = sorted(cands)
        rng.shuffle(cands)
        return cands[:n]

    def group(a, b):
        shared = words_of[a] & words_of[b]
        return "six" if shared - {RUACH} else ("ruach" if shared else "")

    def by_group(cands, n, key=group):  # half six-word, half רוּחַ-only, where the pool allows
        six = take([p for p in cands if key(*p) == "six"], n // 2)
        ru = take([p for p in cands if key(*p) == "ruach"], n - len(six))
        return six + ru

    others = lambda arm: set().union(*(m for a2, m in mates.items() if a2 != arm))
    st = sorted(v for v in trained if v in seven)
    same = [(a, b) for i, a in enumerate(st) for b in st[i + 1:] if words_of[a] & words_of[b] and frozenset((a, b)) not in any_mates]
    flip = lambda ps: [(a, b) if rng.random() < 0.5 else (b, a) for a, b in ps]
    image = lambda a, b: (perm.get(a, a), perm.get(b, b))
    wrong_cands = [image(a, b) for a, b in same]
    wrong_src = {image(a, b): group(a, b) for a, b in same}
    sets = {"links_own": take([p for p in adj["links"] if frozenset(p) not in others("links")], N_PAIRS),
            "wrong_own": take([p for p in adj["wrong"] if not real(*p) and frozenset(p) not in others("wrong")], N_PAIRS),
            "scrambled_own": take([p for p in adj["scrambled"] if not real(*p) and frozenset(p) not in others("scrambled")], N_PAIRS),
            "links_rev": take([(b, a) for a, b in adj["links"] if (b, a) not in before], N_PAIRS),
            "word_unread": flip(by_group(same, N_PAIRS)),
            "wrong_unread": flip(by_group([p for p in wrong_cands if p[0] != p[1] and not real(*p) and frozenset(p) not in any_mates], N_PAIRS, key=lambda a, b: wrong_src[(a, b)])),
            "xref_unread": flip(take([tuple(sorted(p)) for p in xref if all(v in reads for v in p) and not words_of[min(p)] & words_of[max(p)] and p not in any_mates], N_PAIRS)),
            "word_held": by_group([(a, b) for a in st for b in held if words_of[a] & words_of[b]], N_PAIRS),
            "xref_held": take([(a, b) for p in xref for a, b in (tuple(sorted(p)), tuple(sorted(p))[::-1]) if a in reads and b in held], N_PAIRS)}
    out = {}
    for name, ps in sets.items():
        out[name] = []
        for a, b in ps:
            c, match = control(a, b)
            out[name].append({"a": a, "b": b, "control": c, "match": match, "kind": "word" if words_of[a] & words_of[b] else ("xref" if frozenset((a, b)) in xref else "none"),
                              "group": wrong_src.get((a, b)) or wrong_src.get((b, a), "") if name == "wrong_unread" else group(a, b), "reads_a": reads[a], "reads_control": reads[c]})
    stats = {name: {"pairs": len(ps), "distinct_b": len({p["b"] for p in ps}), "kinds": dict(Counter(p["kind"] for p in ps)), "groups": dict(Counter(p["group"] for p in ps)),
                    "controls": dict(Counter(p["match"] for p in ps))} for name, ps in out.items()}
    (OUT / "link_pairs.json").write_text(json.dumps({"seed": SEED, "held_after_filter": len(held), "stats": stats, "sets": out}, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    part = path.with_suffix(".part")
    part.write_text(json.dumps(obj), encoding="utf-8")
    part.replace(path)


def scorer(model, tok, budget):
    """Log-likelihood (nats) of each target text after its context, in the training format: BOS, context, newline, target."""
    import torch

    def run(items):  # items: (context or None, target[, separator]); returns [(sum of log p over the target's tokens, token count, per-token log p, tokens the model would have picked)]
        enc = []
        for item in items:
            ctx, tgt = item[:2]
            sep = item[2] if len(item) > 2 else "\n"
            full = tgt if ctx is None else ctx + sep + tgt
            start = 0 if ctx is None else len(ctx) + len(sep)
            e = tok(full, add_special_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True)
            ids, offs, sp = list(e["input_ids"]), list(e["offset_mapping"]), list(e["special_tokens_mask"])
            if ctx is None and not sp[0]:  # no BOS (OLMo): its document separator first, so the first token of a text alone is scored too
                ids, offs, sp = [tok.bos_token_id] + ids, [(0, 0)] + offs, [1] + sp
            keep = [i for i, ((s, _), p) in enumerate(zip(offs, sp)) if not p and s >= start and i > 0]
            enc.append((ids, keep))
        order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
        res = [None] * len(enc)
        b = 0
        while b < len(order):
            n = 1
            while b + n < len(order) and (n + 1) * len(enc[order[b + n]][0]) <= budget:
                n += 1
            idx = order[b:b + n]
            width = max(len(enc[i][0]) for i in idx)
            ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(enc[i][0])] = torch.tensor(enc[i][0])
                mask[j, :len(enc[i][0])] = 1
            with torch.inference_mode():
                logits = model(input_ids=ids.to(0), attention_mask=mask.to(0), use_cache=False).logits
                for j, i in enumerate(idx):
                    pos = torch.tensor(enc[i][1], device=logits.device)
                    target = ids[j, pos.cpu()].to(0)
                    lp = logits[j, pos - 1].float().log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    hits = int((logits[j, pos - 1].argmax(-1) == target).sum())
                    res[i] = (float(lp.sum()), len(pos), [round(float(x), 4) for x in lp.cpu()], hits)
            del logits
            b += n
        return res
    return run


def budget_for(key):
    return 768 if key == "gemma4_e4b" else 1536  # tokens per batch: Ministral peaked at 7.2 GB with 2,048, near the 8 GB card; Gemma's logits are twice as wide


def links(args):
    """Score every pair set for one arm (or the base)."""
    import torch
    model_id, revision = MODELS[args.key]
    name = f"{args.key}_{args.arm}{'' if args.seed == 1 else f'_s{args.seed}'}"
    adapter = None if args.arm == "base" else TRAIN / name / "adapter"
    if adapter is not None:
        t = TRAIN / name / "train.json"
        if not (adapter / "adapter_config.json").exists() or not t.exists() or json.loads(t.read_text(encoding="utf-8"))["stop_note"]:
            raise SystemExit(f"{name} has no adapter or train.json, or is void; it is not scored")
    spec = json.loads((OUT / "link_pairs.json").read_text(encoding="utf-8"))
    bsb = verses("BSB")
    model, tok = load_model(model_id, revision, True, str(adapter) if adapter else None)
    score = scorer(model, tok, budget_for(args.key))
    t0 = time.time()
    out = {"key": args.key, "arm": args.arm, "seed": args.seed, "name": name, "pairs_sha": sha(OUT / "link_pairs.json"), "sets": {}}
    for s, ps in spec["sets"].items():
        ps = ps[:args.limit] if args.limit else ps
        n = len(ps)
        r = score([(bsb[p["a"]], bsb[p["b"]]) for p in ps] + [(bsb[p["control"]], bsb[p["b"]]) for p in ps] + [(None, bsb[p["b"]]) for p in ps])
        out["sets"][s] = {"after_a": [x[0] for x in r[:n]], "after_control": [x[0] for x in r[n:2 * n]], "alone": [x[0] for x in r[2 * n:]],
                          "tokens": [x[1] for x in r[:n]], "tokens_alone": [x[1] for x in r[2 * n:]],
                          "tokens_check": [x[1] for x in r[n:2 * n]] == [x[1] for x in r[:n]],
                          "per_token": {"after_a": [x[2] for x in r[:n]], "after_control": [x[2] for x in r[n:2 * n]]}}
        print(f"{name} {s}: {n} pairs, {time.time() - t0:.0f} s, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    out.update({"peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "minutes": round((time.time() - t0) / 60, 1), "torch": torch.__version__})
    write(OUT / (f"{name}_links.json" if not args.limit else f"{name}_links_limit{args.limit}.json"), out)


POINTS = re.compile("[֑-ְ֯-ׇֽֿׁׂׅׄ]")


def consonants(text):
    """Unpointed Hebrew, as modern text is written: no vowels or accents; maqaf as a space; sof pasuq and paseq dropped."""
    return re.sub(r"\s+", " ", POINTS.sub("", text).replace("־", " ").replace("׃", "").replace("׀", "")).strip()


def shuffled(text, rng):
    """The same words in another order, with the verse-final mark left at the end."""
    body = text.rstrip(END + " ")
    end = text[len(body):]
    w = body.split()
    for _ in range(10):
        s = w[:]
        rng.shuffle(s)
        if s != w:
            return " ".join(s) + end
    return " ".join(reversed(w)) + end


def words(text):
    return [w for w in text.split() if w != "׀"]  # a standalone paseq is not a word


def items_for_languages():
    """The same verses for every model: 300 Old Testament verses (WLC, with Swete's LXX where it has the verse) and 300 New
    Testament verses (SBLGNT). Each has 6 to 40 words and no bracketed qere, and Psalm titles are left out. The control verse for
    the translation check comes from the same book, at a similar length."""
    path = OUT / "language_items.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(SEED + " languages")
    bsb = verses("BSB")
    ids, _ = ordered()
    index = {v: i for i, v in enumerate(ids)}
    ob, _ = openbible(index)
    fame = Counter(v for p in ob for v in p)  # cross-references with 10+ votes touching the verse
    lxx = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(HERE / "data" / "orig" / "LXX_Swete.jsonl", encoding="utf-8")}
    out = {}
    for lang, file in (("hebrew", "WLC.jsonl"), ("greek", "SBLGNT.jsonl")):
        rows = [json.loads(l) for l in open(HERE / "data" / "orig" / file, encoding="utf-8")]
        rows = [r for r in rows if r["id"] in bsb and 6 <= len(words(r["text"])) <= 40 and "[" not in r["text"] and not re.fullmatch(r"Psa\.\d+\.1", r["id"])]
        rows.sort(key=lambda r: r["id"])
        book = defaultdict(list)
        for r in rows:
            book[r["id"].split(".")[0]].append(r)
        items = []
        for r in rng.sample(rows, N_VERSES):
            n = len(words(r["text"]))
            near = lambda pool: [x for x in pool if x["id"] != r["id"] and abs(len(words(x["text"])) - n) <= max(2, n // 5)]
            other = rng.choice(near(book[r["id"].split(".")[0]]) or near(rows))
            orig_shuffled = shuffled(r["text"], rng)
            item = {"id": r["id"], "fame": fame[r["id"]], "orig": r["text"], "orig_shuffled": orig_shuffled, "other": other["text"], "other_id": other["id"],
                    "english": bsb[r["id"]], "english_shuffled": shuffled(bsb[r["id"]], rng)}
            if lang == "hebrew":
                item.update({"cons": consonants(r["text"]), "cons_shuffled": consonants(orig_shuffled), "cons_other": consonants(other["text"])})  # one permutation for both forms
                if r["id"] in lxx and other["id"] in lxx:
                    item.update({"lxx": lxx[r["id"]], "lxx_shuffled": shuffled(lxx[r["id"]], rng), "lxx_other": lxx[other["id"]]})
            items.append(item)
        out[lang] = items
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def languages(args):
    import torch
    model_id, revision = MODELS[args.key]
    spec = items_for_languages()
    model, tok = load_model(model_id, revision, True)
    score = scorer(model, tok, budget_for(args.key))
    out, t0 = {"key": args.key, "items_sha": sha(OUT / "language_items.json"), "langs": {}}, time.time()
    for lang, items in spec.items():
        items = items[:args.limit] if args.limit else items
        texts = ["orig", "orig_shuffled", "english", "english_shuffled"] + (["cons", "cons_shuffled", "lxx", "lxx_shuffled"] if lang == "hebrew" else [])
        res = {"alone": {}, "translation": {}}
        for k in texts:
            sub = [i for i, it in enumerate(items) if k in it]
            r = score([(None, items[i][k]) for i in sub])
            res["alone"][k] = {"items": sub, "sum": [x[0] for x in r], "tokens": [x[1] for x in r],
                               "per_token": [x[2] for x in r] if not k.endswith("shuffled") else None}
        pairs_t = [("after_verse", "orig"), ("after_other", "other")] + ([("after_cons", "cons"), ("after_cons_other", "cons_other"),
                                                                          ("after_lxx", "lxx"), ("after_lxx_other", "lxx_other")] if lang == "hebrew" else [])
        for name, k in pairs_t:
            sub = [i for i, it in enumerate(items) if k in it]
            r = score([(items[i][k], items[i]["english"]) for i in sub])
            res["translation"][name] = {"items": sub, "sum": [x[0] for x in r], "tokens": [x[1] for x in r]}
        res["n"] = len(items)
        out["langs"][lang] = res
        print(f"{args.key} {lang}: {len(items)} verses, {time.time() - t0:.0f} s, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    write(OUT / (f"{args.key}_languages.json" if not args.limit else f"{args.key}_languages_limit{args.limit}.json"), out)


def boot(x, groups, rng, draws=DRAWS):
    """Mean, and a 95% interval resampling groups (the verse b) with all their rows. Returns (mean, low, high, draws)."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return float("nan"), float("nan"), float("nan"), np.full(draws, np.nan)
    keys = sorted(set(groups))
    rows = {k: [] for k in keys}
    for i, g in enumerate(groups):
        rows[g].append(i)
    idx = [np.array(rows[k]) for k in keys]
    m = np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx[k] for k in pick])
        m[d] = x[sel].mean()
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), m


def report(args):
    """Tables from whatever has been scored, then the reading."""
    tag = f"_limit{args.limit}" if args.limit else ""
    reading = "The reading was stated" + __doc__.split("The reading was stated")[1].split("    python probe.py")[0]
    md = ["# Were the links learned? Which base reads Biblical Hebrew or Greek?", "", reading.strip(), ""]
    fmt = lambda t: f"{t[0]:+.2f} ({t[1]:+.2f} to {t[2]:+.2f})"
    spec_path = OUT / "link_pairs.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
    for key in MODELS:
        if spec is None or not (OUT / f"{key}_base_links{tag}.json").exists():
            continue
        runs, stale = {}, []
        for p in sorted(OUT.glob(f"{key}_*_links{tag}.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            if j["pairs_sha"] != sha(spec_path):
                stale.append(p.name)
                continue
            runs["base" if j["arm"] == "base" else j["name"][len(key) + 1:]] = j
        if "base" not in runs:
            continue
        n_of = lambda s: len(runs["base"]["sets"][s]["after_a"])
        bs = {s: [p["b"] for p in spec["sets"][s][:n_of(s)]] for s in SETS}
        delta = lambda arm, s: np.array(runs[arm]["sets"][s]["after_a"]) - np.array(runs[arm]["sets"][s]["after_control"])  # nats per verse of b
        rng = lambda k: np.random.default_rng(zlib.crc32(repr((key, k[:3])).encode()))  # fixed per contrast, the same on every run
        md += [f"## {key}", ""]
        if stale:
            md += [f"Left out, scored on an older pair set: {', '.join(stale)}.", ""]
        md += ["What context a gives b over the control, in nats per verse of b: base's own value, then each arm's gain over base (shown, not read).", "",
               "| | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS),
               "| base | " + " | ".join(f"{delta('base', s).mean():+.2f}" for s in SETS) + " |"]
        for arm in runs:
            if arm != "base":
                md.append(f"| {arm} − base | " + " | ".join(f"{(delta(arm, s) - delta('base', s)).mean():+.2f}" for s in SETS) + " |")
        contrasts = [(x, y) for x, y in (("links", "scrambled"), ("links", "wrong"), ("wrong", "scrambled"), ("links_s2", "scrambled_s2"), ("links_s2", "wrong_s2"), ("wrong_s2", "scrambled_s2"))
                     if x in runs and y in runs]
        cache = {}

        def c(x, y, s, sub=None):
            if (y, x, s, sub) in cache or (x, y) > (y, x):  # a contrast and its reverse share one set of draws, so a table and a verdict cannot disagree
                m, lo, hi, dr = c(y, x, s, sub)
                return -m, -hi, -lo, -dr
            k = (x, y, s, sub)
            if k not in cache:
                d = delta(x, s) - delta(y, s)
                keep = np.ones(len(d), bool) if sub is None else np.array([sub(p) for p in spec["sets"][s][:len(d)]])
                cache[k] = boot(d[keep], [b for b, ok in zip(bs[s], keep) if ok], rng(k))
            return cache[k]
        md += ["", "Between Bible-trained arms on the same pairs (read): nats per verse of b, 95% interval resampling verses b.", "",
               "| Contrast | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
        for x, y in contrasts:
            md.append(f"| {x} − {y} | " + " | ".join(fmt(c(x, y, s)) for s in SETS) + " |")
        splits = [("links_own", "word links", lambda p: p["kind"] == "word"), ("links_own", "cross-references", lambda p: p["kind"] == "xref"),
                  ("word_unread", "six words", lambda p: p["group"] == "six"), ("word_unread", "רוּחַ only", lambda p: p["group"] == "ruach"),
                  ("wrong_unread", "images of six-word pairs", lambda p: p["group"] == "six"), ("wrong_unread", "images of רוּחַ pairs", lambda p: p["group"] == "ruach"),
                  ("word_held", "six words", lambda p: p["group"] == "six"), ("word_held", "רוּחַ only", lambda p: p["group"] == "ruach")]
        if contrasts:
            md += ["", "Split by kind (shown, not read):", "", "| Set, part | " + " | ".join(f"{x} − {y}" for x, y in contrasts) + " |", "|---|" + "---|" * len(contrasts)]
            for s, label, f in splits:
                md.append(f"| {s}, {label} | " + " | ".join(fmt(c(x, y, s, f)) for x, y in contrasts) + " |")
        # recency: among pairs read once, does the gain over the reference grow with the place in the pass where the pair was read?
        rows = {arm: rows_of(arm) for arm in ("links", "wrong", "scrambled")}
        rec = []
        for own, ref in (("links", "scrambled"), ("wrong", "scrambled"), ("scrambled", "links"), ("links_s2", "scrambled_s2"), ("wrong_s2", "scrambled_s2"), ("scrambled_s2", "links_s2")):
            if own not in runs or ref not in runs:
                continue
            arm, seed = own.split("_s")[0], int(own.split("_s")[1]) if "_s" in own else 1
            order = list(range(len(rows[arm])))
            random.Random(TRAIN_SEED + seed - 1).shuffle(order)
            place = {r: i / len(order) for i, r in enumerate(order)}
            where = defaultdict(list)
            for ri, r in enumerate(rows[arm]):
                for i in range(len(r) - 1):
                    where[(r[i], r[i + 1])].append(place[ri])
            s = f"{arm}_own"
            d = delta(own, s) - delta(ref, s)
            pl = np.array([where[(p["a"], p["b"])][0] if len(where[(p["a"], p["b"])]) == 1 else np.nan for p in spec["sets"][s][:len(d)]])
            ok = ~np.isnan(pl)
            late, early = d[ok & (pl >= 0.5)], d[ok & (pl < 0.5)]
            r_ = float(np.corrcoef(pl[ok], d[ok])[0, 1]) if ok.sum() > 2 else float("nan")
            rec.append(f"- {own} against {ref} on its own pairs read once ({ok.sum()}): read in the second half {late.mean():+.2f}, the first half {early.mean():+.2f}; "
                       f"correlation with the place in the pass {r_:+.2f}.")
        if rec:
            md += ["", "Supporting evidence, recency (pairs read late in the pass should gain more if the pairs themselves were learned):", ""] + rec
        md += ["", "Against the reading:"]
        verdict = lambda t: "clear of zero" if t[1] > 0 else ("below zero" if t[2] < 0 else "not shown, smaller than ±" + f"{(t[2] - t[1]) / 2:.2f}")
        for own, refs in (("links", ("scrambled", "wrong")), ("wrong", ("scrambled", "links")), ("scrambled", ("links", "wrong"))):
            per_run = []
            for sfx in ("", "_s2"):
                if own + sfx not in runs:
                    continue
                parts = []
                for ref in refs:
                    if ref + sfx in runs:
                        t = c(own + sfx, ref + sfx, f"{own}_own")
                        parts.append(f"against {ref + sfx} {fmt(t)}, {verdict(t)}")
                if not parts:
                    continue
                learned = all("clear of zero" in p for p in parts)
                per_run.append(learned)
                md.append(f"- {own + sfx} on its own pairs: " + "; ".join(parts) + f": {'it learned its order' if learned else 'not shown that it learned its order'}.")
            if len(per_run) == 2:  # the reading needs both runs to agree
                md.append(f"  - {own}, both runs: " + ("it learned its order" if all(per_run) else ("not shown" if not any(per_run) else "the runs disagree, so it is not settled")) + ".")
            elif len(per_run) == 1:
                md.append(f"  - {own}: one run only.")
        for sfx in ("", "_s2"):
            L, W = "links" + sfx, "wrong" + sfx
            if L in runs and W in runs:
                a, b = c(L, W, "links_own"), c(W, L, "wrong_own")
                diff = a[3] - b[3]
                t = (a[0] - b[0], float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975)))
                md.append(f"- Learned more readily, {sfx or 'seed 1'}: (links − wrong on links pairs) − (wrong − links on wrong-links pairs) = {fmt(t)}: "
                          f"{'the Bible links were learned more readily' if t[1] > 0 else ('the arbitrary order was learned more readily' if t[2] < 0 else 'not told apart')}.")
                g1, g2 = c(L, W, "word_unread"), c(W, L, "wrong_unread")
                gd = g1[3] - g2[3]
                tg = (g1[0] - g2[0], float(np.quantile(gd, 0.025)), float(np.quantile(gd, 0.975)))
                h = c(L, W, "word_held")
                xh = c(L, W, "xref_held")
                gen = tg[1] > 0 and h[1] > 0
                md.append(f"- Generalise, {sfx or 'seed 1'}: same-word pairs never read together, links − wrong {fmt(g1)} against wrong − links on its images {fmt(g2)}, "
                          f"difference {fmt(tg)}; into held-out verses, links − wrong {fmt(h)} (cross-references {fmt(xh)}): "
                          f"{'the links generalise' if gen else 'not shown that the links generalise'}.")
        for sfx in ("", "_s2"):
            L, S = "links" + sfx, "scrambled" + sfx
            if L in runs and S in runs and "wrong" + sfx not in runs:
                md.append(f"- Upper bound only (no wrong-links arm), {sfx or 'seed 1'}: links − scrambled on same-word pairs never read together {fmt(c(L, S, 'word_unread'))}, "
                          f"into held-out verses {fmt(c(L, S, 'word_held'))}.")
        md.append("")
    langs = {k: json.loads((OUT / f"{k}_languages{tag}.json").read_text(encoding="utf-8")) for k in MODELS if (OUT / f"{k}_languages{tag}.json").exists()}
    if langs:
        items = items_for_languages()
        md += ["## Which base reads Biblical Hebrew or Greek", "",
               "Per verse, in bits, averaged over the verses: the text alone (lower is better known), what shuffling its words costs (beside log2 of the number of "
               "orders, n!), and how much the verse helps predict its own English against another verse of the same book (and that gain as a share of the English "
               "alone). Famous and obscure are the verses above the upper and at or below the lower third of cross-reference counts (not equal thirds, and "
               "in the Old Testament confounded with genre); the ratio is the text's bits over its own English's. Compare bases "
               "within a text, not texts with each other.", "",
               "| Text | Model | Bits | Shuffling costs | log2 n! | Translation gain | Share of English | Famous / obscure, ratio to English | Bits in second half |",
               "|---|---|---:|---:|---:|---|---:|---|---:|"]
        texts = [("hebrew", "Hebrew, pointed (WLC)", "orig", "after_verse", "after_other"), ("hebrew", "Hebrew, consonants only", "cons", "after_cons", "after_cons_other"),
                 ("hebrew", "Greek, LXX (Swete), same verses", "lxx", "after_lxx", "after_lxx_other"), ("hebrew", "English (BSB), OT verses", "english", None, None),
                 ("greek", "Greek, NT (SBLGNT)", "orig", "after_verse", "after_other"), ("greek", "English (BSB), NT verses", "english", None, None)]
        for lang, label, k, tv, to in texts:
            for model, j in langs.items():
                d = j["langs"][lang]
                its = items[lang][:d["n"]]
                A = d["alone"][k]
                bits = -np.array(A["sum"]) / math.log(2)
                sh = -np.array(d["alone"][k + "_shuffled"]["sum"]) / math.log(2)
                n_w = np.array([len(words(its[i]["orig" if k == "cons" else k])) for i in A["items"]])  # the consonantal shuffle permutes the pointed text's units
                lfact = np.array([math.lgamma(n + 1) / math.log(2) for n in n_w])
                eng = {i: -s / math.log(2) for i, s in zip(d["alone"]["english"]["items"], d["alone"]["english"]["sum"])}
                ratio = np.array([b / eng[i] for b, i in zip(bits, A["items"])])
                fame = np.array([its[i]["fame"] for i in A["items"]])
                lo, hi = np.quantile(fame, [1 / 3, 2 / 3])
                fam = f"{ratio[fame > hi].mean():.2f} / {ratio[fame <= lo].mean():.2f}" if (fame > hi).any() and (fame <= lo).any() else "–"
                half = "–"
                if A["per_token"]:
                    half = f"{np.mean([sum(t[len(t) // 2:]) / sum(t) if sum(t) else np.nan for t in A['per_token']]):.2f}"
                gain, share = "–", "–"
                if tv:
                    T, O = d["translation"][tv], d["translation"][to]
                    g = (np.array(T["sum"]) - np.array(O["sum"])) / math.log(2)
                    gain = fmt(boot(g, T["items"], np.random.default_rng(4)))
                    share = f"{np.mean([gi / eng[i] for gi, i in zip(g, T['items'])]):.0%}"
                md.append(f"| {label} | {model} | {bits.mean():.0f} | {(sh - bits).mean():+.0f} | {lfact.mean():.0f} | {gain} | {share} | {fam} | {half} |")
        md.append("")
    path = OUT / f"report{tag}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))


def run(args):
    """Every arm of each model not yet scored, one process per arm (a crash or stall then costs one arm), then the report."""
    py = [sys.executable, "-B", str(HERE / "probe.py")]
    log = OUT / "probe.log"
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "link_pairs.json").exists():
        subprocess.run(py + ["pairs"], cwd=HERE, check=True)
    arms = [("base", 1), ("links", 1), ("wrong", 1), ("scrambled", 1), ("links", 2), ("wrong", 2), ("scrambled", 2), ("tinystories", 1)]
    for key in args.keys:
        steps = []
        for arm, seed in arms:
            name = f"{key}_{arm}{'' if seed == 1 else f'_s{seed}'}"
            if (arm == "base" or (TRAIN / name / "adapter").exists()) and not (OUT / f"{name}_links.json").exists():
                steps.append(["links", "--key", key, "--arm", arm, "--seed", str(seed)])
        if args.languages and not (OUT / f"{key}_languages.json").exists():
            steps.append(["languages", "--key", key])
        for s in steps:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(s)}\n")
                f.flush()
                p = subprocess.Popen(py + s, stdout=f, stderr=subprocess.STDOUT, cwd=HERE, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                try:
                    code = p.wait(timeout=args.max_minutes * 60)
                except subprocess.TimeoutExpired:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=f, stderr=subprocess.STDOUT)
                    p.wait()
                    code = 124
                f.write(f"=== exit {code} at {time.strftime('%H:%M:%S')}\n")
    args.limit = 0
    report(args)


def main():
    in_venv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pairs").set_defaults(fn=pairs)
    l = sub.add_parser("links")
    l.add_argument("--key", required=True, choices=MODELS)
    l.add_argument("--arm", required=True, choices=("base", "links", "scrambled", "tinystories", "wrong"))
    l.add_argument("--seed", type=int, default=1)
    l.add_argument("--limit", type=int, default=0)
    l.set_defaults(fn=links)
    g = sub.add_parser("languages")
    g.add_argument("--key", required=True, choices=MODELS)
    g.add_argument("--limit", type=int, default=0)
    g.set_defaults(fn=languages)
    r = sub.add_parser("run")
    r.add_argument("--keys", nargs="+", default=list(MODELS), choices=MODELS)
    r.add_argument("--languages", action="store_true")
    r.add_argument("--max-minutes", type=float, default=60)
    r.set_defaults(fn=run)
    rp = sub.add_parser("report")
    rp.add_argument("--limit", type=int, default=0, help="read the --limit smoke files instead of the full runs")
    rp.set_defaults(fn=report)
    args = p.parse_args()
    keep_running()
    args.fn(args)


if __name__ == "__main__":
    main()
