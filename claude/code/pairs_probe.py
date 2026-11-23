"""Round 2's checks (Clayton, 14 Nov 2026: "Go ahead with the next round"), by forward passes only, on Ministral.

Round 2 trains exact BSB verse pairs along the Bible's links with the loss on the second verse only (train_relations.py pairs).
Each link was given 0, 1 or 2 readings each way at random, so the link itself is the treatment. The arms: pairs_links, pairs_scrambled
(the same verses, read equally often as a and as b, rewired so that no pair is a link) and pairs_tinystories, two runs of the first two.

THE READING, fixed before any round-2 adapter was scored (14 Nov 2026):
- R1, particular links learned. For each verse b that has a cross-reference partner read with it twice and another never read with it
  (both partners read elsewhere), take log p(b | dose-2 partner) − log p(b | dose-0 partner). Average pairs_links minus pairs_scrambled.
  Learned if the 95% interval is above zero in both runs. Same-word links (Isaiah 11:2's seven words) are read the same way,
  separately. Shown beside it: the same net of base, dose 1, and the 25% and 50% checkpoints.
- Intervals are normal 95% intervals in which items sharing any verse, in any role, may covary (multiway clustering).
- R2, carried beyond base. On cross-references from a read verse into a verse no arm read (held out, or unread in the pair graph),
  log p(b | a) − log p(b | matched control). pairs_links minus base. Carried if the interval is above zero in both runs.
  pairs_tinystories − base is shown beside it: on the walks, continuous text of any kind raised this measure by about +2 nats,
  so R2 may fail as an effect of the format, and a pass does not by itself credit the Bible's links.
- Predicted half-widths, from the step-0 calibration (the old walk adapters on these sets, 14 Nov): R1 about ±0.4 nats on
  cross-references and ±0.3 on same-word links; R2 about ±0.8. A real effect smaller than these will read as not shown.
  The same calibration gives the prior for R2: walk links − base was −1.4 (−2.2 to −0.6) and walk scrambled − base −3.9 into
  verses neither read. Base, which saw no dose, gives R1 +0.47 (−0.28 to +1.22) on cross-references, as random assignment should.
- Shown, not read: each arm on its own pairs; the letter check (recall of each verse's last two thirds from its first third); the
  12 Nov pair sets; the replies.
- An interval spanning zero reads: not shown. A negative means the model remains closer to the pattern of this world, never
  evidence against Scripture.

    python pairs_probe.py sets                                   # -> runs/probe/round2/pair_sets.json (CPU, once)
    python pairs_probe.py score --arm base                       # -> runs/probe/round2/ministral3_8b_base.json
    python pairs_probe.py score --arm pairs_links --seed 2       # an adapter; --checkpoint 25 reads adapter_25
    python pairs_probe.py score --arm links                      # an old walk adapter on the new sets (calibration)
    python pairs_probe.py old --arm links --scale 0.5            # the 12 Nov sets, the adapter at half strength (the dose check)
    python pairs_probe.py fidelity --arm base                    # -> runs/probe/round2/fidelity_ministral3_8b_base.json
    python pairs_probe.py report                                 # -> runs/probe/round2/report.md
"""
import argparse
import json
import random
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bible import CODES, verses
from measure_base import XREF, expand, in_venv, keep_running, load_model, openbible, ordered, word_sets
from probe import MODELS, budget_for, scorer, sha, write

HERE = Path(__file__).parent
TRAIN = HERE / "runs" / "train"
OUT = HERE / "runs" / "probe" / "round2"
KEY = "ministral3_8b"
SEED = "round2 20261114"
DRAWS = 2000
N_OWN = 500
K = 8  # the opening of b: on 12 Nov nine-tenths of the link signal sat in b's first 4-5 tokens
ARMS = ("base", "pairs_links", "pairs_scrambled", "pairs_tinystories", "links", "scrambled", "wrong", "tinystories",
        "tongues_true", "tongues_false", "tongues_true_w", "tongues_false_w")  # the last three are form B's adapters (tongues.py, 15 Nov), read here for harm
RUACH = "7307"


def name_of(arm, seed):
    return f"{KEY}_{arm}{'' if seed == 1 else f'_s{seed}'}"


def sets(args):
    """The pair sets, from the pair graph alone. The same for every arm."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    g = json.loads((TRAIN / "pairs_graph.json").read_text(encoding="utf-8"))
    bsb = verses("BSB")
    ids, _ = ordered()
    index = {v: i for i, v in enumerate(ids)}
    read, held = set(g["read"]), set(g["held"])
    unread = held | set(g["unread_in_universe"])
    words_of = defaultdict(set)
    for n, vs in word_sets().items():
        for v in vs:
            if v in index:
                words_of[v].add(n)
    seven = set(words_of)
    any_ob = set()
    for line in open(XREF, encoding="utf-8").read().splitlines()[1:]:
        a, b = line.split("\t")[:2]
        any_ob.update(frozenset((x, y)) for x in expand(a) for y in expand(b) if x != y)
    ob, _ = openbible(index)
    reads = Counter()
    for l in g["links"]:
        reads[l["a"]] += l["dose"]
        reads[l["b"]] += l["dose"]
    mates = {frozenset((l["a"], l["b"])) for l in g["links"] if l["dose"]} | {frozenset((e["a"], e["b"])) for e in g["scrambled"]}
    chapter = lambda v: tuple(v.split(".")[:2])
    testament = lambda v: "OT" if CODES.index(v.split(".")[0]) < 39 else "NT"
    toks = lambda v: re.findall(r"[a-z']+", bsb[v].lower())
    seen = {tuple(w[i:i + 8]) for v in read for w in [toks(v)] for i in range(len(w) - 7)}
    cover = lambda v: (lambda w: len({j for i in range(len(w) - 7) if tuple(w[i:i + 8]) in seen for j in range(i, i + 8)}) / max(len(w), 1))(toks(v))
    length = {v: len(bsb[v].split()) for v in index if v in bsb}
    linked_to = lambda c, b: bool(words_of[c] & words_of[b]) or frozenset((c, b)) in any_ob or frozenset((c, b)) in mates or chapter(c) == chapter(b)
    pool = sorted(read)

    def control(a, b):
        ladder = [("matched", 0.25, (0.67, 1.5), True, True), ("reads x2", 0.25, (0.5, 2.0), True, True), ("any testament", 0.25, (0.5, 2.0), False, True),
                  ("length 50%", 0.5, (0.5, 2.0), False, True), ("length only", 1.0, (0, 1e9), False, True), ("any class", 0.5, (0, 1e9), False, False)]
        cands = [c for c in pool if c not in (a, b) and not linked_to(c, b)]
        for label, width, (lo, hi), same_t, same_class in ladder:
            ok = [c for c in cands if abs(length[c] - length[a]) <= width * length[a] and (not same_class or (c in seven) == (a in seven))
                  and lo <= reads[c] / reads[a] <= hi and (not same_t or testament(c) == testament(a))]
            if ok:
                return rng.choice(ok), label
        raise ValueError(f"no control for {a} -> {b}")

    group = lambda a, b: "six" if (words_of[a] & words_of[b]) - {RUACH} else ("ruach" if words_of[a] & words_of[b] else "")
    out = {}
    for kind in ("xref", "word"):  # R1: within the same b, a partner read twice against one never read with it
        partners = defaultdict(lambda: defaultdict(list))
        for l in g["links"]:
            if l["kind"] == kind and l["a"] in read and l["b"] in read:
                partners[l["b"]][l["dose"]].append(l["a"])
                partners[l["a"]][l["dose"]].append(l["b"])
        items = []
        for b in sorted(partners):
            p = partners[b]
            if p[2] and p[0]:
                a2, a0 = rng.choice(sorted(p[2])), rng.choice(sorted(p[0]))
                items.append({"b": b, "a2": a2, "a0": a0, "a1": rng.choice(sorted(p[1])) if p[1] else None,
                              "group2": group(a2, b), "group0": group(a0, b)})
        out[f"r1_{kind}"] = items
    clean = {v for v in unread if cover(v) < 0.5}  # near-copies of read text are not unread
    grams = lambda v: {tuple(w[i:i + 8]) for w in [toks(v)] for i in range(len(w) - 7)}
    into = sorted({(a, b) for p in ob for a, b in (p, p[::-1]) if a in read and b in clean and not grams(a) & grams(b)})
    pairs_of = lambda ps: [dict(zip(("a", "b"), p)) for p in ps]
    read_links = sorted((l["a"], l["b"]) if rng.random() < 0.5 else (l["b"], l["a"]) for l in g["links"] if l["dose"])
    scr_edges = sorted((e["a"], e["b"]) if rng.random() < 0.5 else (e["b"], e["a"]) for e in g["scrambled"])
    rng.shuffle(read_links)
    rng.shuffle(scr_edges)
    for name, ps in (("into_unread", into), ("own_links", read_links[:N_OWN]), ("own_scrambled", scr_edges[:N_OWN])):
        out[name] = []
        for p in pairs_of(ps):
            c, match = control(p["a"], p["b"])
            out[name].append({**p, "control": c, "match": match, "held": p["b"] in held})
    stats = {k: {"items": len(v), "distinct_b": len({i["b"] for i in v}), "controls": dict(Counter(i.get("match", "within b") for i in v))} for k, v in out.items()}
    stats["into_unread"]["into_held"] = sum(i["held"] for i in out["into_unread"])
    (OUT / "pair_sets.json").write_text(json.dumps({"seed": SEED, "graph_sha": sha(TRAIN / "pairs_graph.json"), "stats": stats, "sets": out},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


def scale_lora(model, s):
    """Multiply every LoRA layer's scaling by s: the adapter at a fraction of its strength."""
    n = 0
    for m in model.modules():
        if isinstance(getattr(m, "scaling", None), dict):
            for k in m.scaling:
                m.scaling[k] *= s
            n += 1
    assert n, "no LoRA layers to scale"
    return n


def adapter_of(args):
    name = name_of(args.arm, args.seed)
    if args.arm == "base":
        return name, None
    d = TRAIN / name
    t = d / "train.json"
    if not t.exists() or json.loads(t.read_text(encoding="utf-8"))["stop_note"]:
        raise SystemExit(f"{name} has no train.json, or is void; it is not scored")
    adapter = d / ("adapter" if not args.checkpoint else f"adapter_{args.checkpoint}")
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"{adapter} does not exist")
    return name, adapter


def loaded(args):
    import torch
    name, adapter = adapter_of(args)
    model_id, revision = MODELS[KEY]
    model, tok = load_model(model_id, revision, True, str(adapter) if adapter else None)
    if args.scale != 1:
        scale_lora(model, args.scale)
    tag = (f"_c{args.checkpoint}" if args.checkpoint else "") + (f"_x{args.scale:g}" if args.scale != 1 else "")
    return name + tag, model, tok, torch


def score(args):
    """Every round-2 set, for one arm (or base)."""
    spec_path = OUT / "pair_sets.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    bsb = verses("BSB")
    name, model, tok, torch = loaded(args)
    run = scorer(model, tok, budget_for(KEY))
    t0 = time.time()
    out = {"name": name, "arm": args.arm, "seed": args.seed, "checkpoint": args.checkpoint, "scale": args.scale, "sets_sha": sha(spec_path), "sets": {}}
    for s, items in spec["sets"].items():
        items = items[:args.limit] if args.limit else items
        cols = ("a2", "a0", "a1") if s.startswith("r1_") else ("a", "control")
        texts, where = [], {}
        for c in cols + ("alone",):
            where[c] = []
            for it in items:
                if c == "alone":
                    where[c].append(len(texts))
                    texts.append((None, bsb[it["b"]]))
                elif it[c] is None:
                    where[c].append(None)
                else:
                    where[c].append(len(texts))
                    texts.append((bsb[it[c]], bsb[it["b"]]))
        r = run(texts)
        out["sets"][s] = {c: [None if i is None else r[i][0] for i in where[c]] for c in where}
        out["sets"][s]["head"] = {c: [None if i is None else sum(r[i][2][:K]) for i in where[c]] for c in where}
        print(f"{name} {s}: {len(items)} items, {time.time() - t0:.0f} s, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    out.update({"peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "minutes": round((time.time() - t0) / 60, 1)})
    write(OUT / (f"{name}.json" if not args.limit else f"{name}_limit{args.limit}.json"), out)


def old(args):
    """The 12 Nov pair sets (runs/probe/link_pairs.json), in probe.py's own format, for one arm, at a chosen strength."""
    spec_path = HERE / "runs" / "probe" / "link_pairs.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    bsb = verses("BSB")
    name, model, tok, torch = loaded(args)
    run = scorer(model, tok, budget_for(KEY))
    t0 = time.time()
    out = {"name": name, "pairs_sha": sha(spec_path), "scale": args.scale, "sets": {}}
    for s, ps in spec["sets"].items():
        ps = ps[:args.limit] if args.limit else ps
        n = len(ps)
        r = run([(bsb[p["a"]], bsb[p["b"]]) for p in ps] + [(bsb[p["control"]], bsb[p["b"]]) for p in ps] + [(None, bsb[p["b"]]) for p in ps])
        out["sets"][s] = {"after_a": [x[0] for x in r[:n]], "after_control": [x[0] for x in r[n:2 * n]], "alone": [x[0] for x in r[2 * n:]],
                          "per_token": {"after_a": [x[2] for x in r[:n]], "after_control": [x[2] for x in r[n:2 * n]]}}
        print(f"{name} old {s}: {n} pairs, {time.time() - t0:.0f} s", flush=True)
    write(OUT / f"old_{name}{f'_limit{args.limit}' if args.limit else ''}.json", out)


def fidelity(args):
    """The letter: each verse's first third given, how well the rest is predicted (nats, and tokens the model would have picked).
    Verses read by the walks, verses read by the pairs, the held-out verses, and 300 verses neither read."""
    OUT.mkdir(parents=True, exist_ok=True)
    bsb = verses("BSB")
    g = json.loads((TRAIN / "pairs_graph.json").read_text(encoding="utf-8"))
    walk = Counter(v for l in open(TRAIN / "links.jsonl", encoding="utf-8") for v in json.loads(l)["verses"])
    pair_read, held = set(g["read"]), set(g["held"])
    rng = random.Random(SEED + " fidelity")
    toks = lambda v: re.findall(r"[a-z']+", bsb[v].lower())
    seen = {tuple(w[i:i + 8]) for v in set(walk) | pair_read for w in [toks(v)] for i in range(len(w) - 7)}
    fresh = lambda v: not any(tuple(w[i:i + 8]) in seen for w in [toks(v)] for i in range(len(w) - 7))  # parallel-passage twins of read text are not unread
    others = sorted(v for v in bsb if v not in walk and v not in pair_read and v not in held and len(bsb[v].split()) >= 9 and fresh(v))
    groups = {"walk_read": sorted(walk), "pair_read": sorted(pair_read), "held": sorted(v for v in held if fresh(v)), "unread": sorted(rng.sample(others, 300))}
    if args.extra:  # more groups of verses, named in a JSON file (15 Nov: the unread verses base itself quotes or nearly finishes); each token's log p is kept
        groups.update({k: sorted(v for v in vs if v in bsb) for k, vs in json.loads(Path(args.extra).read_text(encoding="utf-8")).items()})
    todo = sorted({v for vs in groups.values() for v in vs if len(bsb[v].split()) >= 6})
    todo = todo[:args.limit] if args.limit else todo
    name, model, tok, torch = loaded(args)
    run = scorer(model, tok, budget_for(KEY))
    t0 = time.time()
    items = []
    for v in todo:
        w = bsb[v].split()
        k = max(3, len(w) // 3)
        items.append((" ".join(w[:k]), " " + " ".join(w[k:]), ""))  # the space goes with the next word's token, so the rest is scored from its first word
    r = run(items)
    scored = set(todo)
    out = {"name": name, "groups": {k: [v for v in vs if v in scored] for k, vs in groups.items()}, "walk_reads": dict(walk),
           "verses": {v: {"nats": x[0], "tokens": x[1], "hits": x[3], **({"lp": x[2]} if args.extra else {})} for v, x in zip(todo, r)},
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "minutes": round((time.time() - t0) / 60, 1)}
    write(OUT / f"fidelity_{Path(args.extra).stem + '_' if args.extra else ''}{name}{f'_limit{args.limit}' if args.limit else ''}.json", out)
    print(f"{name} fidelity: {len(todo)} verses, {out['minutes']} min", flush=True)


def shared(x, members):
    """Mean and a 95% interval in which items sharing any verse, in any role (a, control or b), are allowed to covary
    (multiway clustering): var = sum over item pairs that share a verse of e_i * e_j / n^2, interval = mean +- 1.96 sd.
    The review of 14 Nov found the product-weight bootstrap first used here 2.5 times too wide on R1, where nearly every
    verse occurs once; this covered 93-97% in its simulations."""
    x = np.asarray(x, dtype=float)
    keep = ~np.isnan(x)
    x, members = x[keep], [m for m, k in zip(members, keep) if k]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    vs = sorted({v for m in members for v in m})
    col = {v: i for i, v in enumerate(vs)}
    M = np.zeros((len(x), len(vs)))
    for i, m in enumerate(members):
        M[i, [col[v] for v in set(m)]] = 1
    S = (M @ M.T) > 0
    e = x - x.mean()
    sd = float(np.sqrt(max(e @ S @ e, 0.0))) / len(x)
    return float(x.mean()), float(x.mean() - 1.96 * sd), float(x.mean() + 1.96 * sd)


def report(args):
    """Tables from whatever has been scored, then the reading."""
    spec_path = OUT / "pair_sets.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    runs = {}
    for p in sorted(OUT.glob(f"{KEY}_*.json")):
        if "_limit" in p.name:
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("sets_sha") == sha(spec_path):
            runs[j["name"][len(KEY) + 1:]] = j
    fmt = lambda t: f"{t[0]:+.2f} ({t[1]:+.2f} to {t[2]:+.2f})"
    walk = {v for l in open(TRAIN / "links.jsonl", encoding="utf-8") for v in json.loads(l)["verses"]}
    old_arms = ("links", "scrambled", "wrong", "tinystories")
    reading = "THE READING" + __doc__.split("THE READING")[1].split("    python pairs_probe.py")[0]
    md = ["# Round 2: were particular links learned, and carried beyond base?", "", reading.strip(), ""]
    arr = lambda r, s, c: np.array([np.nan if x is None else x for x in r["sets"][s][c]], dtype=float)

    def r1(arm, kind, minus=None, cols=("a2", "a0")):
        s = f"r1_{kind}"
        it = spec["sets"][s]
        d = arr(runs[arm], s, cols[0]) - arr(runs[arm], s, cols[1])
        if minus:
            d = d - (arr(runs[minus], s, cols[0]) - arr(runs[minus], s, cols[1]))
        return shared(d, [(i[cols[0]] or i["a0"], i[cols[1]], i["b"]) for i in it])

    def gain(arm, s, minus=None):
        it = spec["sets"][s]
        d = arr(runs[arm], s, "a") - arr(runs[arm], s, "control")
        if minus:
            d = d - (arr(runs[minus], s, "a") - arr(runs[minus], s, "control"))
        if arm.split("_c")[0] in old_arms and s == "into_unread":  # the walks read some of these b: keep only verses no arm read
            d = np.where([i["b"] in walk for i in it], np.nan, d)
        return shared(d, [(i["a"], i["control"], i["b"]) for i in it])
    if "base" not in runs:
        md += ["Base has not been scored yet."]
    else:
        md += ["## R1: particular links learned", "",
               "log p(b | partner read twice) − log p(b | partner never read with it), nats per verse of b, mean (95% interval).", "",
               "| | cross-references | same-word links |", "|---|---|---|"]
        rows = [(f"{a} − {m}", a, m) for a, m in (("pairs_links", "pairs_scrambled"), ("pairs_links_s2", "pairs_scrambled_s2"),
                                                     ("pairs_links", "base"), ("pairs_links_s2", "base"), ("pairs_scrambled", "base"),
                                                     ("pairs_tinystories", "base"), ("pairs_links_c25", "pairs_scrambled_c25"),
                                                     ("pairs_links_c50", "pairs_scrambled_c50"), ("links", "scrambled"), ("links", "base"))]
        rows = [(label + (" (old walk adapters: calibration only)" if a in old_arms else ""), a, m) for label, a, m in rows]
        rows = [("base", "base", None)] + rows
        for label, a, m in rows:
            if a in runs and (m is None or m in runs):
                md.append(f"| {label} | {fmt(r1(a, 'xref', m))} | {fmt(r1(a, 'word', m))} |")
        md += ["", "Dose 1 against dose 0, the same way (items with a dose-1 partner):", "", "| | cross-references | same-word links |", "|---|---|---|"]
        for a, m in (("pairs_links", "pairs_scrambled"), ("pairs_links_s2", "pairs_scrambled_s2")):
            if a in runs and m in runs:
                md.append(f"| {a} − {m} | {fmt(r1(a, 'xref', m, ('a1', 'a0')))} | {fmt(r1(a, 'word', m, ('a1', 'a0')))} |")
        md += ["", "## R2: carried beyond base, into verses no arm read", "",
               "log p(b | a) − log p(b | matched control), each arm minus base, nats per verse of b. Base's own value first.", "",
               "| | into unread verses | its own read pairs (links) | the scrambled arm's pairs |", "|---|---|---|---|"]
        md.append(f"| base | {fmt(gain('base', 'into_unread'))} | {fmt(gain('base', 'own_links'))} | {fmt(gain('base', 'own_scrambled'))} |")
        for a in ("pairs_links", "pairs_links_s2", "pairs_scrambled", "pairs_scrambled_s2", "pairs_tinystories", "pairs_links_c25", "pairs_links_c50", "links", "scrambled", "tinystories"):
            if a in runs:
                md.append(f"| {a} − base | {fmt(gain(a, 'into_unread', 'base'))} | {fmt(gain(a, 'own_links', 'base'))} | {fmt(gain(a, 'own_scrambled', 'base'))} |")
        md += ["", "Opening and rest: R1's links − scrambled and R2's links − base, split into b's first 8 tokens and the rest.", "",
               "| | R1 cross-ref, opening | rest | R2, opening | rest |", "|---|---|---|---|---|"]
        for a, m in (("pairs_links", "pairs_scrambled"), ("pairs_links_s2", "pairs_scrambled_s2")):
            if a in runs and m in runs and "base" in runs:
                h = lambda r, s, c: np.array([np.nan if x is None else x for x in r["sets"][s]["head"][c]], dtype=float)
                r1h = (h(runs[a], "r1_xref", "a2") - h(runs[a], "r1_xref", "a0")) - (h(runs[m], "r1_xref", "a2") - h(runs[m], "r1_xref", "a0"))
                r1all = (arr(runs[a], "r1_xref", "a2") - arr(runs[a], "r1_xref", "a0")) - (arr(runs[m], "r1_xref", "a2") - arr(runs[m], "r1_xref", "a0"))
                r2h = (h(runs[a], "into_unread", "a") - h(runs[a], "into_unread", "control")) - (h(runs["base"], "into_unread", "a") - h(runs["base"], "into_unread", "control"))
                r2all = (arr(runs[a], "into_unread", "a") - arr(runs[a], "into_unread", "control")) - (arr(runs["base"], "into_unread", "a") - arr(runs["base"], "into_unread", "control"))
                md.append(f"| {a} | {np.nanmean(r1h):+.2f} | {np.nanmean(r1all - r1h):+.2f} | {np.nanmean(r2h):+.2f} | {np.nanmean(r2all - r2h):+.2f} |")
    fid = {p.name[len(f"fidelity_{KEY}_"):-5]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob(f"fidelity_{KEY}_*.json")) if "_limit" not in p.name}
    if "base" in fid:
        md += ["", "## The letter: each verse's last two thirds, given its first third", "",
               "Nats per verse (arm − base; negative is worse), and the share of verses the model would have finished word for word.", "",
               "| | " + " | ".join(fid["base"]["groups"]) + " |", "|---|" + "---|" * len(fid["base"]["groups"])]
        for arm, f in fid.items():
            cells = []
            for grp, vs in f["groups"].items():
                d = np.array([f["verses"][v]["nats"] - fid["base"]["verses"][v]["nats"] for v in vs])
                exact = np.mean([f["verses"][v]["hits"] == f["verses"][v]["tokens"] for v in vs])
                cells.append(f"{d.mean():+.1f}, {exact:.0%}" if arm != "base" else f"{np.mean([f['verses'][v]['nats'] for v in vs]):.1f} nats, {exact:.0%}")
            md.append(f"| {arm} | " + " | ".join(cells) + " |")
    olds = {p.name[len(f"old_{KEY}_"):-5]: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob(f"old_{KEY}_*.json")) if "_limit" not in p.name}
    probe12 = lambda arm: json.loads((HERE / "runs" / "probe" / f"{KEY}_{arm}_links.json").read_text(encoding="utf-8"))
    if olds:
        base12 = probe12("base")["sets"]
        md += ["", "## The dose check: the old links adapter at full and half strength, on the 12 Nov sets", "",
               f"Link gain = after a − after control, in b's first {K} tokens and the rest; memorising = b scored alone, arm − base. Nats per verse of b.", "",
               "| | links_own gain, opening | rest | links_own alone − base | xref_held gain − base | xref_held alone − base |", "|---|---|---|---|---|---|"]
        for label, s in [("base", base12), ("links, full", probe12("links")["sets"])] + [(f"{k}", o["sets"]) for k, o in olds.items()]:
            A, C = s["links_own"]["per_token"]["after_a"], s["links_own"]["per_token"]["after_control"]
            gh = np.mean([sum(a[:K]) - sum(c[:K]) for a, c in zip(A, C)])
            gt = np.mean([sum(a[K:]) - sum(c[K:]) for a, c in zip(A, C)])
            mem = np.mean(np.array(s["links_own"]["alone"]) - np.array(base12["links_own"]["alone"]))
            xg = np.mean((np.array(s["xref_held"]["after_a"]) - np.array(s["xref_held"]["after_control"])) - (np.array(base12["xref_held"]["after_a"]) - np.array(base12["xref_held"]["after_control"])))
            xa = np.mean(np.array(s["xref_held"]["alone"]) - np.array(base12["xref_held"]["alone"]))
            md.append(f"| {label} | {gh:+.2f} | {gt:+.2f} | {mem:+.2f} | {xg:+.2f} | {xa:+.2f} |")
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    in_venv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sets").set_defaults(fn=sets)
    for cmd, fn in (("score", score), ("old", old), ("fidelity", fidelity)):
        s = sub.add_parser(cmd)
        s.add_argument("--arm", required=True, choices=ARMS)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--checkpoint", choices=("25", "50"))
        s.add_argument("--scale", type=float, default=1.0)
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--extra", help="fidelity only: a JSON file of named groups of verse ids to read in place of the usual groups")
        s.set_defaults(fn=fn)
    sub.add_parser("report").set_defaults(fn=report)
    args = p.parse_args()
    keep_running()
    args.fn(args)


if __name__ == "__main__":
    main()
