"""Blinded review of sampled replies, plus an objective check of quoted Scripture.

    python review.py prep   --runs runs/audit --out runs/audit/review   # per-category batches + key
    python review.py quotes --runs runs/audit                           # quoted spans vs ESV/NKJV text
    python review.py score  --review runs/audit/review                  # aggregate reviewer scores

Reviewers see each probe's prompt and scoring focus and every reply under shuffled labels. The audit used
four models x three conditions (bare T=1 sample 0, bare T=0, native T=1).
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CONDITIONS = {"suite_bare_t1": "bare_t1", "suite_bare_t0": "bare_t0", "suite_native_t1": "native_t1"}


def load_rows(runs, sources=None):
    """Sample-0 replies from the audit conditions, or from --source FILE:MODEL:CONDITION triples."""
    if sources:
        specs = [s.rsplit(":", 2) for s in sources]
    else:
        specs = [(str(Path(runs) / folder / "responses.jsonl"), None, cond) for folder, cond in CONDITIONS.items()]
    rows = []
    for path, model, cond in specs:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["sample"] == 0 and (model is None or r["model"] == model):
                rows.append(r | {"condition": cond})
    return rows


def suite():
    spec = json.loads((HERE / "eval_probes.json").read_text(encoding="utf-8"))
    return json.loads(Path(spec["suite"]).read_text(encoding="utf-8"))


def prep(args):
    rows, s = load_rows(args.runs, args.source), suite()
    by_probe = defaultdict(list)
    for r in rows:
        by_probe[r["id"]].append(r)
    out = Path(args.out)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    key = {}
    for cat in s["categories"]:
        probes = []
        for p in cat["prompts"]:
            replies = sorted(by_probe[p["prompt_id"]], key=lambda r: (r["model"], r["condition"]))
            random.Random(f"review:{p['prompt_id']}").shuffle(replies)
            key[p["prompt_id"]] = {f"R{i + 1:02d}": {"model": r["model"], "condition": r["condition"]}
                                   for i, r in enumerate(replies)}
            probes.append({"id": p["prompt_id"], "prompt": p["prompt"], "scoring_focus": p["scoring_focus"],
                           "geometry_targets": p["geometry_targets"], "pair_group": p.get("pair_group"),
                           "replies": [{"label": f"R{i + 1:02d}", "cut_off": not r["complete"],
                                        "text": r["final"] or "(no final answer)"}
                                       for i, r in enumerate(replies)]})
        for g in range(0, len(probes), 4):  # groups of four keep the lexical pairs together
            gid = f"{cat['category_id']}_{g // 4 + 1}"
            lines = [f"# Review group {gid}", "", f"**Category:** {cat['category_id']}: {cat['description']}", "",
                     "**Suite usage notes:** " + " ".join(s["usage_notes"]), ""]
            for p in probes[g:g + 4]:
                lines += [f"## Probe {p['id']}", "", f"**Prompt:** {p['prompt']}", "",
                          f"**Scoring focus:** {p['scoring_focus']}", "",
                          f"**Targets:** {', '.join(p['geometry_targets'])}"
                          + (f"; pair group {p['pair_group']}" if p["pair_group"] else ""), ""]
                for r in p["replies"]:
                    lines += [f"<<<<< REPLY {r['label']}" + (" (cut off by the token cap)" if r["cut_off"] else "")
                              + " >>>>>", "", r["text"], "", f"<<<<< END {r['label']} >>>>>", ""]
            (out / "batches" / f"{gid}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(list((out / 'batches').glob('*.md')))} groups, {len(rows)} replies -> {out}")


def norm_words(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split()


def quotes(args):
    """Quoted spans of 6+ words that are not copied from the prompt, scored by how many of their
    word 5-grams occur in the ESV or NKJV. Low coverage on a span presented as Scripture is a misquote."""
    grams = set()
    for f in ("ESV_bible_segments.jsonl", "NKJV_bible_segments.jsonl"):
        words = []
        for line in open(HERE / "data" / f, encoding="utf-8"):
            words += norm_words(json.loads(line)["text"])
        grams |= {tuple(words[i:i + 5]) for i in range(len(words) - 4)}
    stats = defaultdict(lambda: [0, 0, 0, 0])  # spans, bible-like (>=0.8), mixed, non-bible (<0.3)
    examples = defaultdict(list)
    for folder in CONDITIONS:
        for line in open(Path(args.runs) / folder / "responses.jsonl", encoding="utf-8"):
            r = json.loads(line)
            prompt_grams = {tuple(w) for w in zip(*[norm_words(r["prompt"])[i:] for i in range(5)])}
            for span in re.findall(r"[\"“]([^\"“”]{20,400})[\"”]", r["final"]):
                w = norm_words(span)
                g = [tuple(w[i:i + 5]) for i in range(len(w) - 4)]
                if len(w) < 6 or not g or sum(x in prompt_grams for x in g) / len(g) > 0.5:
                    continue
                cov = sum(x in grams for x in g) / len(g)
                s = stats[(r["model"], folder)]
                s[0] += 1
                s[1 if cov >= 0.8 else 3 if cov < 0.3 else 2] += 1
                if cov < 0.3 and re.search(r"\b(says?|said|wrote|verse|[1-3]?\s?[A-Z][a-z]+ \d+:\d+)\b",
                                           r["final"][max(0, r["final"].find(span) - 80):r["final"].find(span)]):
                    examples[r["model"]].append({"id": r["id"], "condition": folder, "span": span[:160]})
    print("model     condition         spans  bible>=0.8  mixed  non-bible<0.3")
    for (m, c), s in sorted(stats.items()):
        print(f"{m:9s} {c:17s} {s[0]:5d} {s[1]:10d} {s[2]:6d} {s[3]:13d}")
    out = Path(args.runs) / "quote_check.json"
    out.write_text(json.dumps({"stats": {f"{m}|{c}": s for (m, c), s in stats.items()},
                               "attributed_non_bible_examples": examples}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {out}")


LABEL = re.compile(r"^\s*((?:[1-3] )?[A-Z][a-z]+(?: of [A-Z][a-z]+)?) (\d+)(?::(\d+)(?:-(\d+))?)? \((ESV|NKJV)\)\n(.*?)(?:\n\n|$)", re.S)


def mechanism(args):
    """For one model's replies: does the analysis open with a recalled Scripture scene, is the scene
    exact text of the cited verses, and does Scripture reach the final answer where it should not?"""
    codes = ("Gen Exo Lev Num Deu Jos Jdg Rut 1Sa 2Sa 1Ki 2Ki 1Ch 2Ch Ezr Neh Est Job Psa Pro Ecc Sol Isa Jer Lam "
             "Eze Dan Hos Joe Amo Oba Jon Mic Nah Hab Zep Hag Zec Mal Mat Mar Luk Joh Act Rom 1Co 2Co Gal Eph Phi Col "
             "1Th 2Th 1Ti 2Ti Tit Phm Heb Jam 1Pe 2Pe 1Jo 2Jo 3Jo Jud Rev").split()
    full = ("Genesis Exodus Leviticus Numbers Deuteronomy Joshua Judges Ruth 1_Samuel 2_Samuel 1_Kings 2_Kings "
            "1_Chronicles 2_Chronicles Ezra Nehemiah Esther Job Psalm Proverbs Ecclesiastes Song_of_Solomon Isaiah "
            "Jeremiah Lamentations Ezekiel Daniel Hosea Joel Amos Obadiah Jonah Micah Nahum Habakkuk Zephaniah Haggai "
            "Zechariah Malachi Matthew Mark Luke John Acts Romans 1_Corinthians 2_Corinthians Galatians Ephesians "
            "Philippians Colossians 1_Thessalonians 2_Thessalonians 1_Timothy 2_Timothy Titus Philemon Hebrews James "
            "1_Peter 2_Peter 1_John 2_John 3_John Jude Revelation").split()
    names = {f.replace("_", " "): c for f, c in zip(full, codes)} | {"Psalms": "Psa", "Song of Songs": "Sol"}
    bible = {}
    for tr in ("ESV", "NKJV"):
        for line in open(HERE / "data" / f"{tr}_bible_segments.jsonl", encoding="utf-8"):
            r = json.loads(line)
            bible[(tr, r["id"])] = r["text"]
    rows = [json.loads(l) for f in args.files for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["model"] == args.model]
    stats = defaultdict(int)
    refs = defaultdict(int)
    for r in rows:
        stats["replies"] += 1
        stats["complete"] += r["complete"]
        m = LABEL.match(r["analysis"].replace("<|channel|>analysis<|message|>", "", 1))
        if not m:
            continue
        stats["opens_with_label"] += 1
        book, ch, v1, v2, tr, text = m.groups()
        refs[f"{book} {ch}" + (f":{v1}" if v1 else "") + (f"-{v2}" if v2 else "")] += 1
        code = names.get(book)
        verses = ([bible.get((tr, f"{code}.{ch}.{v}")) for v in range(int(v1), int(v2 or v1) + 1)] if code and v1 else [None])
        if None in verses:
            stats["label_unknown_book_or_bad_range"] += 1
            continue
        exact = " ".join(verses)
        stats["recall_exact"] += text.strip() == exact
        w, e = norm_words(text), norm_words(exact)
        g = {tuple(e[i:i + 5]) for i in range(len(e) - 4)}
        cov = sum(tuple(w[i:i + 5]) in g for i in range(len(w) - 4)) / max(1, len(w) - 4)
        stats["recall_cov>=0.9"] += cov >= 0.9
        if r["id"].startswith(("tech", "secular", "gsm8k")) and re.search(r"\b(Matthew|Mark|Luke|John|Acts) \d+:\d+", r["final"]):
            stats["scripture_ref_in_secular_final"] += 1
    print(json.dumps(dict(stats), indent=1))
    print("most recalled:", sorted(refs.items(), key=lambda x: -x[1])[:12])


FIELDS = ("coherent", "focus", "correct", "useful", "misquote", "shape")


def score(args):
    """Average the reviewers' scores per reply, then per model and condition. Best and worst picks and
    forbidden-behaviour flags are counted per reviewer (so two reviewers can each flag a reply)."""
    review = Path(args.review)
    key = json.loads((review / "key.json").read_text(encoding="utf-8"))
    scores = json.loads((review / "scores.json").read_text(encoding="utf-8"))
    per_reply = defaultdict(lambda: defaultdict(list))  # (probe, label) -> field -> [reviewer values]
    flags = defaultdict(lambda: defaultdict(int))
    picks = defaultdict(lambda: [0, 0])
    for rv in scores:
        for p in rv["probes"]:
            k = key[p["id"]]
            for r in p["replies"]:
                for f in FIELDS:
                    if r.get(f) is not None:
                        per_reply[(p["id"], r["label"])][f].append(r[f])
                cell = (k[r["label"]]["model"], k[r["label"]]["condition"])
                for fl in r["forbidden"]:
                    flags[cell][fl] += 1
            for i, field in enumerate(("best", "worst")):
                for lab in p[field]:
                    picks[(k[lab]["model"], k[lab]["condition"])][i] += 1
    agg = defaultdict(lambda: defaultdict(list))
    agree = defaultdict(lambda: [0, 0])
    for (pid, lab), vals in per_reply.items():
        who = key[pid][lab]
        for f, v in vals.items():
            agg[(who["model"], who["condition"])][f].append(sum(v) / len(v))
            if len(v) == 2:
                agree[f][0] += v[0] == v[1]
                agree[f][1] += 1
    print("model     condition   n  " + " ".join(f"{c:>8s}" for c in FIELDS) + "  best worst  forbidden flags")
    table = {}
    for cell in sorted(agg):
        a = agg[cell]
        means = {c: (sum(a[c]) / len(a[c]) if a[c] else None) for c in FIELDS}
        table[f"{cell[0]}|{cell[1]}"] = {"n": len(a["focus"]), **means, "best": picks[cell][0],
                                         "worst": picks[cell][1], "forbidden": dict(flags[cell])}
        vals = " ".join(f"{means[c]:8.2f}" if means[c] is not None else f"{'-':>8s}" for c in FIELDS)
        print(f"{cell[0]:9s} {cell[1]:10s} {len(a['focus']):3d} {vals}  {picks[cell][0]:4d} {picks[cell][1]:5d}  "
              f"{dict(flags[cell])}")
    print("exact reviewer agreement: " + ", ".join(f"{f} {a / n:.0%} (n={n})" for f, (a, n) in agree.items() if n))
    (review / "summary.json").write_text(json.dumps({"cells": table, "agreement": {f: a / n for f, (a, n) in agree.items() if n}},
                                                    indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--runs", default="runs/audit"); p.add_argument("--out", required=True)
    p.add_argument("--source", action="append", help="FILE:MODEL:CONDITION (repeatable) instead of the audit set")
    q = sub.add_parser("quotes"); q.add_argument("--runs", required=True)
    s = sub.add_parser("score"); s.add_argument("--review", required=True)
    m = sub.add_parser("mechanism"); m.add_argument("--model", required=True); m.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"prep": prep, "quotes": quotes, "score": score, "mechanism": mechanism}[a.cmd](a)
