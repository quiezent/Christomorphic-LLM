"""Build the first-movement training set: Scripture's own requests, answered by Scripture's own scenes.

Every example is exact ESV or NKJV text. The user turn is the words a person speaks to Jesus or an
apostle in the Gospels or Acts. The training target, in the model's private analysis channel, is the
whole scene recalled with its reference: who asked, and how the Lord (or the apostle) answered.

    python build_data.py candidates   # regex candidates -> runs/data/candidates.json
    python build_data.py build        # kept refs (refs.json) -> runs/data/train.jsonl
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "runs" / "data"
BOOKS = {"Mat": "Matthew", "Mar": "Mark", "Luk": "Luke", "Joh": "John", "Act": "Acts"}
OPEN, CLOSE = "“", "”"
REPLY = re.compile(r"^(?:(?:And|But|Then|So|Now)\s+)?(?:Jesus|he|He|the Lord|Peter|Paul)\b[^“]{0,50}"
                   r"\b(said|answered|replied|declared)\b[^“]{0,60}“")
VOCATIVE = re.compile(r"^(?:(?:Good |O )?(?:Lord|Teacher|Rabbi|Master|Sir|Son of David|Jesus)"
                      r"(?:, (?:Son of David|Master|Lord))?[,!]\s*)+")


def verses(translation):
    rows = [json.loads(l) for l in open(HERE / "data" / f"{translation}_bible_segments.jsonl", encoding="utf-8")]
    return {r["id"]: r for r in rows if r["book"] in BOOKS}


def speech(text):
    """The quoted words of a verse, first opening mark to last closing mark, as an exact substring."""
    start, end = text.find(OPEN), text.rfind(CLOSE)
    if start < 0:
        return ""
    return text[start + 1:end] if end > start else text[start + 1:]


def reply_span(vs, book, ch, v):
    """Reply verse ids from v, continuing while the reply's quotation is still open (max 8 verses)."""
    ids, depth = [], 0
    for k in range(v, v + 8):
        r = vs.get(f"{book}.{ch}.{k}")
        if r is None:
            break
        ids.append(r["id"])
        depth += r["text"].count(OPEN) - r["text"].count(CLOSE)
        if depth <= 0:
            break
    return ids


def candidates():
    esv = verses("ESV")
    out = []
    for vid, q in esv.items():
        b, c, v = q["book"], q["chapter"], q["verse"]
        a = esv.get(f"{b}.{c}.{v + 1}")
        if a is None or OPEN not in q["text"] or not REPLY.search(a["text"]):
            continue
        if re.search(r"\b(Jesus|he|He)\b[^“]{0,40}\b(said|saying|answered|replied)\b", q["text"].split(OPEN)[0]):
            continue
        out.append({"ask": vid, "reply": reply_span(esv, b, c, v + 1), "ask_text": q["text"],
                    "reply_text": " ".join(esv[i]["text"] for i in reply_span(esv, b, c, v + 1))})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} candidates -> {OUT / 'candidates.json'}")


def label(ids):
    b, c, v1 = ids[0].split(".")
    v2 = ids[-1].split(".")[2]
    return f"{BOOKS[b]} {c}:{v1}" + (f"-{v2}" if v2 != v1 else "")


def build():
    keep = json.loads((HERE / "refs.json").read_text(encoding="utf-8"))["pairs"]
    rows = []
    for tr in ("ESV", "NKJV"):
        vs = verses(tr)
        for p in keep:
            b, c, v = p["ask"].split(".")
            ask = vs.get(p["ask"])
            span = [p["ask"]] + [i for i in reply_span(vs, b, int(c), int(v) + 1)]
            if ask is None or len(span) < 2 or OPEN not in ask["text"]:
                print(f"skip {tr} {p['ask']}: missing or no speech")
                continue
            user = VOCATIVE.sub("", speech(ask["text"])).strip()
            user = user[:1].upper() + user[1:]
            scene = " ".join(vs[i]["text"] for i in span)
            rows.append({"ref": label(span), "translation": tr, "user": user,
                         "analysis": f"{label(span)} ({tr})\n{scene}"})
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} examples from {len(keep)} pairs -> {OUT / 'train.jsonl'}")


if __name__ == "__main__":
    {"candidates": candidates, "build": build}[sys.argv[1]]()
