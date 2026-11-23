"""rd0, the scroll-open test (Clayton, 22 Oct 2026; no training). Luke 4:17: he unrolled the scroll and found the
place before he read. When the place is found for it, does the model read, or recite what it does not hold?

The 33 unseen chapters in Cut-1 form, with the scroll in the turn:
1. the chapter in the BSB (as rm1's unseen prompts already carry it);
2. cross-references in the BSB from the other Testament, from OpenBible.info's index (CC BY; links in both
   directions, with votes above 0), at most 15 per chapter by votes, plus every other-Testament verse that shares
   distinctive wording with a verse of the chapter in the BSB (three word 5-grams, or two where the index links those
   two verses; one of them in at most four verses of the Bible), so that explicit quotations are never dropped;
   KNOWN, the explicit New Testament quotations of these chapters, checks that none is missing;
3. the rule, after the question.
The exemplar arm puts three same-genre corpus posts before the turn as earlier turns, never on the same chapter.

    python rd0.py build     # -> runs/rd0/scroll_prompts.jsonl, exemplar_prompts.jsonl, scrolls.json, build.md
    python rd0.py score     # off-scroll references, grounding and the plumb line, per arm -> runs/rd0/score.md
    python rd0.py reviews --out runs/rd0/review_t0   # via_scroll and no_claim per arm, beside review.py rubric
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from bible import CODES, FULL, STORED, verses

HERE = Path(__file__).parent
RUN = HERE / "runs" / "rd0"
XREF = HERE / "data" / "OpenBible_xref_source" / "cross_references.txt"
SEED = 20261022
CAP = 15  # cross-references per chapter, by the index's votes
SPAN = 5  # verses at most in one cross-reference
RULE = ("Use only the Scripture provided above. Quote only from it, with references. "
        "If a passage you want is not provided, say so and do not quote it.")
OB_CODES = ("Gen Exod Lev Num Deut Josh Judg Ruth 1Sam 2Sam 1Kgs 2Kgs 1Chr 2Chr Ezra Neh Esth Job Ps Prov Eccl Song "
            "Isa Jer Lam Ezek Dan Hos Joel Amos Obad Jonah Mic Nah Hab Zeph Hag Zech Mal Matt Mark Luke John Acts Rom "
            "1Cor 2Cor Gal Eph Phil Col 1Thess 2Thess 1Tim 2Tim Titus Phlm Heb Jas 1Pet 2Pet 1John 2John 3John Jude "
            "Rev").split()
OB = dict(zip(OB_CODES, CODES))
OT = set(CODES[:39])
RARE = 4  # a distinctive 5-gram occurs in at most this many verses of the BSB
KNOWN = {  # explicit quotations between these chapters and the other Testament (the brief's four among them)
    "Psa.14": ["Rom.3.10", "Rom.3.11", "Rom.3.12"], "Psa.16": ["Act.2.25", "Act.2.26", "Act.2.27", "Act.2.28", "Act.13.35"],
    "Psa.18": ["Rom.15.9"], "Psa.19": ["Rom.10.18"],
    "Psa.22": ["Mat.27.46", "Mar.15.34", "Joh.19.24", "Heb.2.12", "Mat.27.43"], "Psa.35": ["Joh.15.25"],
    "Psa.36": ["Rom.3.18"], "Psa.37": ["Mat.5.5"], "Psa.40": ["Heb.10.5", "Heb.10.6", "Heb.10.7"],
    "Psa.41": ["Joh.13.18"], "Gen.22": ["Heb.6.14", "Act.3.25"], "Exo.12": ["Joh.19.36"],
    "Isa.53": ["Joh.12.38", "Rom.10.16", "Mat.8.17", "Act.8.32", "Act.8.33", "1Pe.2.22", "1Pe.2.24", "Luk.22.37"],
    "Jon.1": ["Mat.12.40"], "Mar.15": ["Psa.22.1", "Psa.22.18"]}
GENRE = {"Psa": "psalms", "Isa": "psalms", "Rut": "narrative", "Jon": "narrative", "Gen": "narrative",
         "Exo": "temple", "Lev": "temple", "Mar": "gospel", "Luk": "gospel"}


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def ob_ref(s):
    """'Matt.27.46' or 'Acts.8.32-Acts.8.33' -> (code, chapter, first verse, last verse), or None across chapters."""
    a, _, b = s.partition("-")
    bk, c, v = a.split(".")
    v2 = int(b.split(".")[2]) if b and b.split(".")[:2] == [bk, c] else int(v)
    return (OB[bk], int(c), int(v), v2) if not b or b.split(".")[:2] == [bk, c] else None


def cite(code, c, v1, v2):
    return f"{FULL[code]} {c}:{v1}" + (f"-{v2}" if v2 != v1 else "")


def grams(text, n=5):
    from plumb import lwords
    w = lwords(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def scroll(chapter, index, bsb, gram_verses):
    """The cross-references for one chapter: [(code, c, v1, v2, (anchor first, anchor last), votes, why)], in the
    chapter's order. The anchor is the verse or verses of the chapter that the index (or the matcher) ties it to."""
    code, c = chapter.split(".")
    c = int(c)
    other = (lambda k: k not in OT) if code in OT else (lambda k: k in OT)
    inside = lambda span, vid: f"{span[0]}.{span[1]}" == vid.rsplit(".", 1)[0] and span[2] <= int(vid.rsplit(".", 1)[1]) <= span[3]
    rows = [(a1, a2, b, votes) for a1, a2, b, votes in index[(code, c)] if other(b[0]) and votes > 0]  # 0 or fewer: dropped
    best = {}
    for a1, a2, b, votes in rows:
        span = (b[0], b[1], b[2], min(b[3], b[2] + SPAN - 1))
        if span not in best or votes > best[span][1]:
            best[span] = ((a1, a2), votes)
    picked = []
    for span, (a, votes) in sorted(best.items(), key=lambda kv: -kv[1][1]):
        if len(picked) == CAP:
            break
        if not any(p[0] == span[0] and p[1] == span[1] and p[2] <= span[3] and span[2] <= p[3] for p in picked):  # no overlap
            picked.append(span + (a, votes, "index"))
    covered = lambda vid: any(inside(p, vid) for p in picked)
    adds = []
    for vid, t in bsb.items():  # explicit quotations: other-Testament verses sharing distinctive wording with this verse
        if not vid.startswith(f"{code}.{c}."):
            continue
        v = int(vid.rsplit(".", 1)[1])
        shared = defaultdict(set)
        for x in grams(t):
            for h in gram_verses.get(x, ()):
                if other(h.split(".")[0]):
                    shared[h].add(x)
        linked = lambda h: any(a1 <= v <= a2 and inside(b, h) for a1, a2, b, _ in rows)  # the index links these two verses
        adds += [(h, (v, v), "5-gram") for h, gs in sorted(shared.items())
                 if (len(gs) >= 3 or len(gs) == 2 and linked(h)) and min(len(gram_verses[x]) for x in gs) <= RARE]
    for h in KNOWN.get(chapter, []):
        a = next((a for s, (a, _) in sorted(best.items(), key=lambda kv: -kv[1][1]) if inside(s, h)), None)
        if a:
            adds.append((h, a, "known quotation"))
    for hit, a, why in adds:
        if not covered(hit) and hit in bsb:
            k, hc, hv = hit.split(".")
            votes = max([v for s, (_, v) in best.items() if inside(s, hit)], default=0)
            picked.append((k, int(hc), int(hv), int(hv), a, votes, why))
    return sorted(picked, key=lambda p: (p[4], CODES.index(p[0]), p[1], p[2]))


def scroll_text(chapter, refs, bsb):
    code, ch = chapter.split(".")
    head = "New Testament cross-references (BSB)" if code in OT else "Old Testament cross-references (BSB)"
    out = [head]
    for k, c, v1, v2, (a1, a2), _, _ in refs:
        vs = [(v, bsb[f"{k}.{c}.{v}"]) for v in range(v1, v2 + 1) if f"{k}.{c}.{v}" in bsb]
        body = vs[0][1] if len(vs) == 1 else " ".join(f"{v} {t}" for v, t in vs)
        out.append(f"{cite(k, c, vs[0][0], vs[-1][0])} (on {cite(code, ch, a1, a2)})\n{body}")
    return "\n\n".join(out)


def tomb_post(post, bsb):
    """The Holy Saturday post's four burial accounts, which name their reference after the block, re-rendered in the
    BSB under the author's own reference (the Lessons rule; lessons.py reads only a reference before the block)."""
    names = {"Matt": "Mat", "Mark": "Mar", "Luke": "Luk", "John": "Joh"}
    def sub(m):
        k, c, v1, v2 = names[m.group(2)], int(m.group(3)), int(m.group(4)), int(m.group(5))
        return " ".join(f"{v} {bsb[f'{k}.{c}.{v}']}" for v in range(v1, v2 + 1)) + f"\n({m.group(2)} {c}:{v1}-{v2})"
    return re.sub(r"((?:^\d+ [^\n]+\n(?:\n(?=\d+ ))?)+)\((Matt|Mark|Luke|John) (\d+):(\d+)\s*-\s*(\d+)\)", sub, post,
                  flags=re.M)


def exemplars(bsb):
    """Three corpus posts per genre (the rm1 corpus, with its corrections), each passing the plumb line:
    {genre: [(file, user, post)]}."""
    from plumb import measure
    from rm1 import IN, RUN as RM1
    posts = {p["file"]: p for p in jl(IN / "corpus.jsonl")}
    corr = json.loads((RM1 / "corrections.json").read_text(encoding="utf-8"))
    for key, edits in corr.items():
        file, field, *_ = key.split(":")
        for old, new in edits:
            if field in ("post", "lessons", "lesson"):
                posts[file]["post"] = posts[file]["post"].replace(old, new)
    def turn(p):
        post = re.sub(r"\A.{0,200}?\bby Peter Sze[^\n]*\n+", "", p["post"], count=1, flags=re.S)
        if p["chapter"] is None:  # the Holy Saturday post: no chapter of its own
            return "What does the burial of Jesus teach us about Christ?", tomb_post(post, bsb)
        return f"What does {p['ref']} teach us about Christ?\n\n{p['turn_text']}", post
    def ok(p):
        user, post = turn(p)
        return not measure(post, user)["fails"]
    pools = {"psalms": [f for f, p in posts.items() if (p["chapter"] or "").startswith("Psa.")],
             "narrative": [f for f, p in posts.items() if (p["chapter"] or "").split(".")[0] in ("Est", "1Ki", "2Ki")
                           and p["chapter"] not in ("1Ki.6", "1Ki.7", "1Ki.8")]}
    out, report = {}, {}
    for g, files in pools.items():
        files = sorted(files)
        random.Random(f"{SEED}:{g}").shuffle(files)
        good = [f for f in files if ok(posts[f])][:3]
        out[g] = good
        report[g] = {"pool": len(files), "failing before the three chosen": [f for f in files[:files.index(good[-1])] if f not in good]}
    out["temple"] = sorted(f for f, p in posts.items() if p["chapter"] in ("1Ki.6", "1Ki.7", "1Ki.8"))
    tomb = [f for f, p in posts.items() if "quiet-day-at-the-tomb" in f]
    out["gospel"] = tomb + out["narrative"][:2]  # one Holy Saturday post; two narrative posts make the three
    for g in ("temple", "gospel"):
        report[g] = {"plumb fails": {f: [x[2:] for x in measure(turn(posts[f])[1], turn(posts[f])[0])["fails"]]
                                     for f in out[g]}}
    return {g: [(f,) + turn(posts[f]) for f in fs] for g, fs in out.items()}, report


def build(args):
    RUN.mkdir(parents=True, exist_ok=True)
    bsb = verses(STORED)
    index = defaultdict(list)  # (code, chapter) -> [(anchor first, anchor last, other passage, votes)], both directions
    for line in open(XREF, encoding="utf-8"):
        if line.startswith("From"):
            continue
        a, b, votes = line.rstrip("\n").split("\t")
        ra, rb = ob_ref(a), ob_ref(b)
        if ra and rb:
            index[ra[:2]].append((ra[2], ra[3], rb, int(votes)))
            index[rb[:2]].append((rb[2], rb[3], ra, int(votes)))  # a link to a range counts for the range's chapter too
    gram_verses = defaultdict(set)
    for vid, t in bsb.items():
        for g in grams(t):
            gram_verses[g].add(vid)
    unseen = jl(HERE / "runs" / "rm1" / "unseen_prompts.jsonl")
    ex, ex_report = exemplars(bsb)
    rows, rows_ex, record, lines = [], [], {}, ["# rd0 scrolls", ""]
    for r in unseen:
        chapter = r["id"].removeprefix("unseen_")
        refs = scroll(chapter, index, bsb, gram_verses)
        prompt = f"{r['prompt']}\n\n{scroll_text(chapter, refs, bsb)}\n\n{RULE}"
        rows.append({"id": r["id"], "prompt": prompt})
        g = GENRE[chapter.split(".")[0]]
        ctx = [m for f, user, post in ex[g] for m in ({"role": "user", "content": user}, {"role": "assistant", "content": post})]
        rows_ex.append({"id": r["id"], "prompt": prompt, "context": ctx})
        record[r["id"]] = {"refs": [cite(k, c, v1, v2) for k, c, v1, v2, *_ in refs],
                           "verses": [f"{k}.{c}.{v}" for k, c, v1, v2, *_ in refs for v in range(v1, v2 + 1)],
                           "exemplars": [f for f, *_ in ex[g]]}
        n5 = [p for p in refs if p[6] != "index"]
        missing = [v for v in KNOWN.get(chapter, []) if v not in record[r["id"]]["verses"]]
        lines.append(f"- **{chapter}:** {len(refs)} cross-references ({len(refs) - len(n5)} by votes"
                     + (f"; added: {', '.join(cite(*p[:4]) + ' (' + p[6] + ')' for p in n5)}" if n5 else "")
                     + (f"; known quotations on it: {len(KNOWN[chapter]) - len(missing)} of {len(KNOWN[chapter])}"
                        if chapter in KNOWN else "") + (f", MISSING {missing}" if missing else "") + ")")
    for name, rs in (("scroll_prompts.jsonl", rows), ("exemplar_prompts.jsonl", rows_ex)):
        with open(RUN / name, "w", encoding="utf-8", newline="\n") as f:
            for x in rs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (RUN / "scrolls.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    lines += ["", "Exemplars per genre: " + json.dumps({g: [f for f, *_ in v] for g, v in ex.items()}, indent=1),
              "", "Exemplar report: " + json.dumps(ex_report, ensure_ascii=False, indent=1)]
    (RUN / "build.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


ARMS = {  # arm -> {temperature: (responses, model)}; sample 0 of each
    "base bare": {"T=0": ("runs/rd0/bare_t0", "base20"), "T=1": ("runs/rm1/unseen_baselines", "base20")},
    "base + scroll": {"T=0": ("runs/rd0/scroll_t0", "base20"), "T=1": ("runs/rd0/scroll_t1", "base20")},
    "base + scroll + exemplars": {"T=0": ("runs/rd0/ex_t0", "base20"), "T=1": ("runs/rd0/ex_t1", "base20")},
    "rm1r + scroll": {"T=0": ("runs/rd0/scroll_t0", "rm1r"), "T=1": ("runs/rd0/scroll_t1", "rm1r")},
    "R38 + scroll": {"T=0": ("runs/rd0/scroll_t0", "r38"), "T=1": ("runs/rd0/scroll_t1", "r38")},
    "rm1r bare": {"T=1": ("runs/rm1r/eval_unseen", "rm1r")},
    "R38 bare": {"T=0": ("runs/rm1/unseen_baselines", "r38")},
}
NOT_PROVIDED = re.compile(r"\bnot (?:been |yet )?(?:provided|included|supplied|given|shown|listed|quoted|in the (?:scroll|"
                          r"text|passages?|scriptures?) (?:provided|above|given))|\bisn['’]t (?:provided|included)|"
                          r"\b(?:outside|beyond) (?:the|what) (?:scroll|provided|supplied|was provided|passages? provided)",
                          re.I)


def arm_rows(root=RUN):
    """Sample 0 of every arm; with root, an earlier build's replies where it has them (runs/rd0/build1)."""
    out = {}
    for arm, temps in ARMS.items():
        for t, (folder, model) in temps.items():
            path = HERE / folder.replace("runs/rd0", str(root.relative_to(HERE)).replace("\\", "/")) / "responses.jsonl"
            path = path if path.exists() else HERE / folder / "responses.jsonl"
            if path.exists():
                out[(arm, t)] = {r["id"]: r for r in jl(path) if r["model"] == model and r["sample"] == 0}
    return out


def measure_reply(final, question, chapter, rec, bsb):
    """One reply against its scroll: references on and off it, references marked as not provided, quotations held
    word for word by the scroll, and the plumb line's own failures (measured against the question alone)."""
    from plumb import exact, lwords, measure, ref_parts
    code, c = chapter.split(".")
    on_verses = {v for v in bsb if v.startswith(f"{code}.{c}.")} | set(rec["verses"])
    on_chapters = {tuple(v.split(".")[:2]) for v in on_verses}
    order = lambda v: (CODES.index(v.split(".")[0]), int(v.split(".")[1]), int(v.split(".")[2]))
    runs = {"scroll": [lwords(bsb[v]) for v in sorted(on_verses, key=order) if v in bsb]}  # in canonical order
    m = measure(final, question)
    off, marked, on = [], [], []
    for s, e, ref, label, have in m["refs"]:
        k, ch, a, b = ref_parts(ref)
        lo, hi = (a, b or a) if a is None or (b or a) >= a else (b, a)  # a reversed range ("Luke 24:28-27") as written
        here = (k, str(ch)) in on_chapters if a is None else any(f"{k}.{ch}.{v}" in on_verses for v in range(lo, hi + 1))
        lo = max(final.rfind(x, 0, s) for x in ".!?\n") + 1
        hi = min([i for i in (final.find(x, e) for x in ".!?\n") if i >= 0] or [len(final)])
        (on if here else marked if NOT_PROVIDED.search(final[lo:hi]) else off).append(ref)
    unheld = Counter()  # a quotation not word for word on the scroll: the divine name's case only; another witness's
    for q in m["quotes"]:  # words (or a verse off the scroll); or no witness's words at all
        t = final[q["start"] + 1:q["end"] - 1]
        if not exact(t, runs):
            unheld["case only" if exact(t, runs, folded=True) else "a witness's words" if q["matched"] else "no witness"] += 1
    return {"refs": len(m["refs"]), "on": len(on), "off": off, "marked": marked, "quotes": len(m["quotes"]),
            "unheld": sum(unheld.values()), "unheld_kind": dict(unheld), "plumb": [f"{f[2]} {f[3]}" for f in m["fails"]],
            "chars": len(final)}


def score(args):
    root = HERE / args.root
    bsb, scrolls = verses(STORED), json.loads((root / "scrolls.json").read_text(encoding="utf-8"))
    per, out = {}, {}
    for (arm, t), rows in arm_rows(root).items():
        got = {pid: measure_reply(r["final"], r["prompt"].split("\n\n")[0], pid.removeprefix("unseen_"), scrolls[pid], bsb)
               | {"complete": r["complete"]} for pid, r in rows.items()}
        per[f"{arm} | {t}"] = got
        n = len(got)
        v = list(got.values())
        answered = [x for x in v if x["chars"]]  # a final answer, even one cut off at the cap
        out[(arm, t)] = {
            "n": n, "complete": sum(x["complete"] for x in v), "with a final answer": len(answered),
            "off per reply": sum(len(x["off"]) for x in v) / n,
            "off per answered reply": sum(len(x["off"]) for x in answered) / max(1, len(answered)),
            "replies with off-scroll refs": sum(bool(x["off"]) for x in v),
            "refs per reply": sum(x["refs"] for x in v) / n, "marked not provided": sum(len(x["marked"]) for x in v),
            "quotes": sum(x["quotes"] for x in v), "quotes not on the scroll": sum(x["unheld"] for x in v),
            "plumb-failing replies": sum(bool(x["plumb"]) for x in v),
            "grounding-failing replies": sum(bool(x["off"] or x["unheld"]) for x in v),
            "failing either": sum(bool(x["plumb"] or x["off"] or x["unheld"]) for x in v),
            "mean final chars": sum(x["chars"] for x in v) / n,
            "unheld kinds": dict(sum((Counter(x["unheld_kind"]) for x in v), Counter())),
            "plumb fail reasons": dict(Counter(f.split()[0] for x in v for f in x["plumb"]).most_common()),
            "off-scroll refs": dict(Counter(r for x in v for r in x["off"]).most_common(12))}
    cols = ["n", "complete", "with a final answer", "off per reply", "off per answered reply",
            "replies with off-scroll refs", "refs per reply", "marked not provided",
            "quotes", "quotes not on the scroll", "plumb-failing replies", "grounding-failing replies", "failing either",
            "mean final chars"]
    lines = ["| Arm | T | " + " | ".join(cols) + " |", "|---|---|" + "---:|" * len(cols)]
    for (arm, t), x in out.items():
        lines.append(f"| {arm} | {t} | " + " | ".join(f"{x[c]:.2f}" if isinstance(x[c], float) and c != "mean final chars"
                                                       else f"{x[c]:.0f}" for c in cols) + " |")
    lines += ["", "Quotations not word for word on the scroll, by kind; the plumb line's failures by reason; the most "
              "frequent off-scroll references:", ""]
    lines += [f"- **{arm} {t}:** {x['unheld kinds']}; {x['plumb fail reasons']}; {x['off-scroll refs']}"
              for (arm, t), x in out.items()]
    text = "\n".join(lines)
    print(text)
    (root / "score.md").write_text(text + "\n", encoding="utf-8")
    (root / "score.json").write_text(json.dumps({"arms": {f"{a} | {t}": x for (a, t), x in out.items()}, "replies": per},
                                               ensure_ascii=False, indent=1), encoding="utf-8")


def reviews(args):
    """The rd0 review's extra fields per arm, beside review.py rubric: via_scroll (the Christ-turn made through a
    supplied cross-reference) and no_claim (a Scripture answer with no checkable claim, counted as D4)."""
    out = Path(args.out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    res = json.loads((out / "output.json").read_text(encoding="utf-8"))
    tally = defaultdict(Counter)
    for batch in res:
        for reviewer in batch:
            for x in (reviewer or {}).get("results", []):
                for lab, m in key[x["probe"]].items():
                    tally[m]["reviews"] += 1
                    tally[m]["via_scroll"] += bool(x[lab].get("via_scroll"))
                    tally[m]["no_claim"] += bool(x[lab].get("no_claim"))
    for m, c in sorted(tally.items()):
        print(f"{m}: via_scroll {c['via_scroll']}/{c['reviews']}, no checkable claim {c['no_claim']}/{c['reviews']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("score").add_argument("--root", default="runs/rd0", help="a build's folder, e.g. runs/rd0/build1")
    sub.add_parser("reviews").add_argument("--out", required=True)
    args = ap.parse_args()
    {"build": build, "score": score, "reviews": reviews}[args.cmd](args)


if __name__ == "__main__":
    main()
