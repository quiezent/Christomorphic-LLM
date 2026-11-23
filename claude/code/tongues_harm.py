"""Form B's harm readings by forward passes (tongues.py, reading 4; fixed 21 Nov 2026): how well the model still predicts text it
was never trained to say. Nothing is sampled. Scores are log-likelihoods in nats, each arm against base on the same texts.

Scored alone, after the start token:
  hebrew, greek   300 unread Hebrew and 300 unread Greek verses, in the forms trained on
  english         300 unread BSB verses
  prose           300 passages of ordinary prose (the TinyStories validation file, the first 60 words of a story)
The letter: a verse's first third is given and the rest is scored (as pairs_probe.py fidelity does):
  letter_held     the 300 unread verses base holds best. Base was scored on a pool of 3,000 verses on 15 Nov; the pool is cut
                  to the verses Form B's steps did not read, ranked on the even tokens of the rest, and read here on the odd ones
  letter_quoted   the unread verses base quoted in its replies so that the quotation stands (20 of the 51: Form B's steps read 26,
                  and 5 have a twin among the read or are not joined at the word)
  letter_random   300 unread verses at random from the same pool once the 300 of letter_held are taken out
  letter_read     300 verses Form B's steps read. Shown, not a harm reading: does the model now recite what it read?

    python tongues_harm.py sets                              # the texts, once (CPU)
    python tongues_harm.py score --arm base                  # -> runs/tongues/harm_base.json (card)
    python tongues_harm.py score --arm true --seed 1 --scale 0.5
    python tongues_harm.py read                              # each arm minus base (CPU)
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from measure_base import in_venv, keep_running, load_model
from probe import MODELS, budget_for, scorer
from tongues import KEY, OUT, SEED, TRAIN, chapter, cluster_interval, name_of, order_key, texts

HERE = Path(__file__).parent
ROUND2 = HERE / "runs" / "probe" / "round2"


def sets(args):
    import tongues_words
    from tongues import config
    d = json.loads((OUT / "data.json").read_text(encoding="utf-8"))
    T = texts()
    W, _ = tongues_words.build(config())
    rng = random.Random(f"{SEED} harm")
    unread = [v for v in d["unread"] if v in T and 8 <= len(T[v]["en"].split()) <= 45]
    heb = sorted(rng.sample([v for v in unread if T[v]["lang"] == "H"], 300), key=order_key)
    grk = sorted(rng.sample([v for v in unread if T[v]["lang"] == "G"], 300), key=order_key)
    eng = sorted(rng.sample([v for v in unread if v not in heb and v not in grk], 300), key=order_key)
    stories = (HERE / "data" / "tinystories" / "TinyStoriesV2-GPT4-valid.txt").read_text(encoding="utf-8").split("<|endoftext|>")
    stories = [" ".join(s.split()) for s in stories if 60 <= len(s.split()) <= 400]
    prose = [" ".join(s.split()[:60]) for s in rng.sample(stories, 300)]
    alone = lambda pairs: [[v, None, text, ""] for v, text in pairs]
    out = {"hebrew": alone((v, W[v]["orig"]) for v in heb), "greek": alone((v, W[v]["orig"]) for v in grk),
           "english": alone((v, T[v]["en"]) for v in eng), "prose": alone((f"story {i}", s) for i, s in enumerate(prose))}
    # the letter: first third given, the rest scored; the space goes with the next word's token
    def letter(vs):
        rows = []
        for v in sorted(vs, key=order_key):
            w = T[v]["en"].split()
            k = max(3, len(w) // 3)
            rows.append([v, " ".join(w[:k]), " " + " ".join(w[k:]), ""])
        return rows
    not_read = set(d["unread"])
    pool_file = json.loads((ROUND2 / "fidelity_letter_pool_ministral3_8b_base.json").read_text(encoding="utf-8"))
    lp = lambda v: pool_file["verses"][v]["lp"]
    even = lambda v: sum(lp(v)[0::2]) / max(1, len(lp(v)[0::2]))
    pool = [v for v in pool_file["groups"]["pool"] if v in not_read and v in T and v in pool_file["verses"] and len(lp(v)) >= 6]
    held = sorted(pool, key=even, reverse=True)[:300]
    rng2 = random.Random(f"{SEED} harm letter")
    rand = rng2.sample(sorted(set(pool) - set(held)), 300)
    quoted = [v for v in pool_file["groups"]["quoted"] if v in not_read and v in T]
    read = json.loads((OUT / "read_verses.json").read_text(encoding="utf-8"))
    was_read = rng2.sample([v for v in read if v in T and len(T[v]["en"].split()) >= 9], 300)
    out.update({"letter_held": letter(held), "letter_quoted": letter(quoted), "letter_random": letter(rand), "letter_read": letter(was_read)})
    (OUT / "harm_sets.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in out.items()}, "| pool not read by Form B:", len(pool))


def score(args):
    spec = json.loads((OUT / "harm_sets.json").read_text(encoding="utf-8"))
    name = "base" if args.arm == "base" else name_of(args.arm, args.seed, args.exact)
    adapter = None if args.arm == "base" else TRAIN / name / ("adapter" if not args.checkpoint else f"adapter_{args.checkpoint}")
    if args.adapter:                                    # a trial adapter by its folder, under a name of its own
        name, adapter = args.name, Path(args.adapter)
    model_id, revision = MODELS[KEY]
    model, tok = load_model(model_id, revision, True, str(adapter) if adapter else None, cpu_words=True)
    if adapter is not None:
        for m in model.modules():                       # the strength, set outright so that it cannot compound
            if isinstance(getattr(m, "scaling", None), dict):
                for k in m.scaling:
                    m.scaling[k] = args.scale * m.lora_alpha[k] / m.r[k]
    run = scorer(model, tok, budget_for(KEY))
    t0 = time.time()
    tag = name + (f"_c{args.checkpoint}" if args.checkpoint else "") + (f"_x{args.scale:g}" if args.scale != 1 else "")
    out = {"name": tag, "sets": {}}
    for k, items in spec.items():
        if args.only and k not in args.only:
            continue
        items = items[:args.limit] if args.limit else items
        r = run([(ctx, text, sep) for _, ctx, text, sep in items])
        out["sets"][k] = {"ids": [it[0] for it in items], "nats": [x[0] for x in r], "tokens": [x[1] for x in r], "hits": [x[3] for x in r],
                          **({"lp": [x[2] for x in r]} if k.startswith("letter") else {})}
        print(f"{tag} {k}: {len(items)} texts, {time.time() - t0:.0f} s", flush=True)
    (OUT / f"harm_{tag}{f'_limit{args.limit}' if args.limit else ''}.json").write_text(json.dumps(out), encoding="utf-8")


def read(args):
    base = json.loads((OUT / "harm_base.json").read_text(encoding="utf-8"))["sets"]
    names = list(base)
    letters = [k for k in names if k.startswith("letter")]

    def value(s, k, odd=None):
        """nats a text. letter_held is always read on the odd tokens of the rest (it was chosen on the even ones); odd=True reads
        every letter set that way, so that they can be set side by side."""
        if k == "letter_held" or (odd and k in letters):
            return np.array([sum(lp[1::2]) for lp in s[k]["lp"]])
        return np.array(s[k]["nats"])
    whole = lambda s, k: float(np.mean([h == t for h, t in zip(s[k]["hits"], s[k]["tokens"])]))
    count = lambda s, k: int(sum(h == t for h, t in zip(s[k]["hits"], s[k]["tokens"])))

    def groups_of(k):
        g = defaultdict(list)
        for i, v in enumerate(base[k]["ids"]):
            g[chapter(v) if "." in v else v].append(i)
        return list(g.values())
    label = {k: (k + ", odd tokens" if k == "letter_held" else k) for k in names}
    res, files = {}, []
    for p in sorted(OUT.glob("harm_*.json")):
        if p.name in ("harm_sets.json", "harm_base.json", "harm_read.json") or "_limit" in p.name:
            continue
        files.append(json.loads(p.read_text(encoding="utf-8")))
    print("Nats a text, arm minus base, with a 95% interval in which the verses of one chapter may move together. Negative is worse.")
    print("letter_held is summed over the odd tokens of the rest of the verse only (about half of it); every other column is a whole sum.\n")
    print(f"{'arm':36s} " + " ".join(f"{label[k]:>26s}" for k in names))
    print(f"{'texts (chapters)':36s} " + " ".join(f"{len(base[k]['ids'])} ({len(groups_of(k))})".rjust(26) for k in names))
    print(f"{'base, its own mean':36s} " + " ".join(f"{value(base, k).mean():26.1f}" for k in names))
    for j in files:
        if set(names) - set(j["sets"]):                 # a trial scored on some sets only: printed apart
            part = {k: float((value(j["sets"], k) - value(base, k)).mean()) for k in names if k in j["sets"]}
            print(f"{j['name']:36s} (some sets only) " + ", ".join(f"{label[k]} {v:+.2f}" for k, v in part.items()))
            continue
        row, odd = {}, {}
        for k in names:
            assert j["sets"][k]["ids"] == base[k]["ids"], f"{j['name']} was scored on other texts: {k}"
            row[k] = cluster_interval(value(j["sets"], k) - value(base, k), groups_of(k))
        for k in letters:
            odd[k] = cluster_interval(value(j["sets"], k, True) - value(base, k, True), groups_of(k))
        res[j["name"]] = {"minus base": row, "letter on odd tokens, minus base": odd,
                          "finished word for word": {k: whole(j["sets"], k) for k in letters}, "finished, count": {k: count(j["sets"], k) for k in letters}}
        print(f"{j['name']:36s} " + " ".join("{:+.2f} ({:+.2f} to {:+.2f})".format(*row[k]).rjust(26) for k in names))
    print("\nThe letter, every set on the odd tokens of the rest (the unit letter_held is read in), arm minus base:")
    print(f"{'arm':36s} " + " ".join(f"{k:>26s}" for k in letters))
    print(f"{'base, its own mean':36s} " + " ".join(f"{value(base, k, True).mean():26.1f}" for k in letters))
    for name, r in res.items():
        print(f"{name:36s} " + " ".join("{:+.2f} ({:+.2f} to {:+.2f})".format(*r["letter on odd tokens, minus base"][k]).rjust(26) for k in letters))
    print("\nThe rest of the verse finished word for word (every token the model's first choice), verses of the set:")
    print(f"{'arm':36s} " + " ".join(f"{k + ' (' + str(len(base[k]['ids'])) + ')':>20s}" for k in letters))
    print(f"{'base':36s} " + " ".join(f"{count(base, k):20d}" for k in letters))
    for name, r in res.items():
        print(f"{name:36s} " + " ".join(f"{r['finished, count'][k]:20d}" for k in letters))
    res["base"] = {"mean": {k: float(value(base, k).mean()) for k in names}, "letter on odd tokens, mean": {k: float(value(base, k, True).mean()) for k in letters},
                   "finished word for word": {k: whole(base, k) for k in letters}, "finished, count": {k: count(base, k) for k in letters},
                   "texts": {k: len(base[k]["ids"]) for k in names}, "chapters": {k: len(groups_of(k)) for k in names}}
    (OUT / "harm_read.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def main():
    in_venv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sets").set_defaults(fn=sets)
    s = sub.add_parser("score")
    s.add_argument("--arm", default="true", choices=("base", "true", "false"))
    s.add_argument("--adapter", help="a trial adapter's folder, scored under --name")
    s.add_argument("--name")
    s.add_argument("--only", nargs="*", help="score these sets only")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--checkpoint", choices=("25", "50"))
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--exact", type=float, default=0.0, help="any value above 0 picks the runs trained with the loss on the exact words")
    s.set_defaults(fn=score)
    sub.add_parser("read").set_defaults(fn=read)
    args = p.parse_args()
    if args.cmd == "score":
        keep_running()
    args.fn(args)


if __name__ == "__main__":
    main()
