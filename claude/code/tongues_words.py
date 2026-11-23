"""Form B at the word (15-21 Nov 2026): which original word each English phrase renders, as places in the two texts.

Why. On base, a verse's own original already stands first among its chapter's originals for 79% (Hebrew, consonants) to 97%
(Greek) of verses (`tongues.py decide`), so the verse-level loss has little left to teach. Clayton also said yes to "the same word
in the tongue it was written in" as the relation to mirror first. So the pairing is taken down to the word: the publisher's
word-by-word table (data/orig/BSB_tables.jsonl) says which English phrase renders which Hebrew, Aramaic or Greek word of the verse.
That table, with its Strong's numbers, is a judgment from outside the text. It was used in round 3 before Clayton had been asked
about it for Form B. Asked on 22 Nov 2026 whether the pairing may come from it, Clayton answered: "Yes, allowed".

What this file gives (CPU): for every verse, the original verse in the chosen form with the character span of each written word,
and for each English phrase of words_probe.py's table the span of its content word and the index of the original word it renders.

    python tongues_words.py check        # coverage of the join, and a few verses printed
"""
import difflib
import io
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from tongues import ACCENTS, HERE, POINTS, config, jl, order_key

MARKS_IN = re.compile("[־׀׃׆\\[\\]]")


def plain_greek(s):
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def greek_key(s):
    return re.sub(r"[^\w]", "", plain_greek(s).lower()).replace("ς", "σ")


def hebrew_key(strong):
    m = re.search(r"\d+", (strong or "").split("/")[-1])
    return m[0] if m else ""


def original_words(row, lang, cfg):
    """The verse in the chosen form, the span of each written word in it, and a key for each word (to join it to the table)."""
    pieces, keys, lex = [], [], []
    if lang == "H":
        pointed = cfg["hebrew"] == "points"
        for w in row["words"]:
            if w.get("type") == "qere":
                continue
            f = ACCENTS.sub("", w["form"])
            f = MARKS_IN.sub("", f if pointed else POINTS.sub("", f)).strip()
            if f:
                pieces.append(f)
                keys.append(hebrew_key(w.get("strong")))
                lex.append("H" + hebrew_key(w.get("strong")))
    else:
        printed = cfg["greek"] == "printed"
        for w in row["words"]:
            f = w["form"] if printed else plain_greek(w["form"])
            pieces.append(f)
            keys.append(greek_key(w["form"]))
            lex.append("G" + (w.get("lemma") or greek_key(w["form"])))
    text, spans, at = "", [], 0
    if lang == "H":
        for f in pieces:
            spans.append((at, at + len(f)))
            at += len(f) + 1
        text = " ".join(pieces)
    else:  # the printed verse keeps its punctuation: find each word in turn
        text = row["text"] if cfg["greek"] == "printed" else plain_greek(row["text"])
        for f in pieces:
            i = text.find(f, at)
            if i < 0:
                return None
            spans.append((i, i + len(f)))
            at = i + len(f)
    return {"text": text, "spans": spans, "keys": keys, "lex": lex}


def build(cfg=None):
    """{verse: {"orig": text, "spans": [...], "pairs": [(english content-word span, original word index, lexicon number, english key, part of speech, early)]}}"""
    cfg = cfg or config()
    table = {r["id"]: r["words"] for r in jl(HERE / "data" / "orig" / "BSB_tables.jsonl")}
    phrases = defaultdict(list)
    for r in jl(HERE / "runs" / "words" / "phrases.jsonl"):
        phrases[r["v"]].append(r)
    out, stats = {}, Counter()
    for file, lang in (("WLC.jsonl", "H"), ("SBLGNT.jsonl", "G")):
        for row in jl(HERE / "data" / "orig" / file):
            v = row["id"]
            if v not in table or len(row.get("refs", [])) != 1 or not row["words"]:
                continue
            ow = original_words(row, lang, cfg)
            if ow is None or not ow["spans"]:
                stats["verse not joined"] += 1
                continue
            tw = table[v]
            tkeys = [hebrew_key(w.get("strong")) if lang == "H" else greek_key(w.get("form", "")) for w in tw]
            sm = difflib.SequenceMatcher(None, tkeys, ow["keys"], autojunk=False)
            to_orig = {}
            for a, b, n in sm.get_matching_blocks():
                for k in range(n):
                    to_orig[a + k] = b + k
            used, pairs = set(), []
            norm = lambda s: re.sub(r"[^a-z]", "", (s or "").lower())
            for p in phrases.get(v, []):
                cands = [i for i, w in enumerate(tw) if i not in used and w.get("strong") == p["strong"]]
                exact = [i for i in cands if norm(tw[i].get("gloss")) == norm(p["phrase"])]
                pick = exact[0] if exact else (cands[0] if len(cands) == 1 else None)
                stats["phrases"] += 1
                if pick is None or pick not in to_orig:
                    stats["phrase not joined"] += 1
                    continue
                used.add(pick)
                pairs.append((p["word"], to_orig[pick], p["strong"], p["key"], p["pos"], p["early"]))
            out[v] = {"lang": lang, "orig": ow["text"], "spans": ow["spans"], "lex": ow["lex"], "pairs": pairs}
            stats["verses"] += 1
            stats["pairs"] += len(pairs)
            stats["original words"] += len(ow["spans"])
    return out, dict(stats)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    from tongues import texts, original
    cfg = config()
    W, stats = build(cfg)
    print(json.dumps(stats, indent=1))
    T = texts()
    same = sum(W[v]["orig"] == original(T[v], cfg) for v in W if v in T)
    print("verses whose text here equals tongues.py's own:", same, "of", sum(v in T for v in W))
    for v in ("Isa.11.2", "Rom.12.2", "Gen.1.2"):
        w = W[v]
        print("\n" + v, "|", w["orig"])
        en = T[v]["en"]
        for (s, e), k, strong, key, pos, early in w["pairs"]:
            a, b = w["spans"][k]
            print(f"   {en[s:e]:16s} -> {w['orig'][a:b]:14s} {strong} {pos}{' (early)' if early else ''}")
    n = Counter(len(W[v]["pairs"]) for v in W)
    print("\npairs a verse: mean", round(sum(k * c for k, c in n.items()) / sum(n.values()), 2), "| verses with 3 or more:", sum(c for k, c in n.items() if k >= 3))


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] == "check"):
    main()


# ---------------------------------------------------------------- sizing at the word, on base (forward passes only)

def last_token(offsets, start, end):
    """The last token that covers any of the characters start..end (the start token has offsets 0,0 and is never chosen)."""
    hit = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    return hit[-1] if hit else None


def word_states(model, tok, strings, wanted, layers, budget=1536):
    """For each text i, the states at the token positions wanted[i] (lists of character spans), at the given layers.
    Returns [(text index, span index, token position, tokens in the text)], and an array [n, layers, dim] float16."""
    import numpy as np
    import torch
    encs = [tok(s, add_special_tokens=True, return_offsets_mapping=True) for s in strings]
    order = sorted(range(len(strings)), key=lambda i: len(encs[i]["input_ids"]))
    index, chunks = [], []
    b = 0
    while b < len(order):
        n = 1
        while b + n < len(order) and (n + 1) * len(encs[order[b + n]]["input_ids"]) <= budget:
            n += 1
        idx = order[b:b + n]
        width = max(len(encs[i]["input_ids"]) for i in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        rows, cols = [], []
        for j, i in enumerate(idx):
            L = len(encs[i]["input_ids"])
            ids[j, :L] = torch.tensor(encs[i]["input_ids"])
            mask[j, :L] = 1
            for k, (s, e) in enumerate(wanted[i]):
                t = last_token(encs[i]["offset_mapping"], s, e)
                if t is not None:
                    rows.append(j)
                    cols.append(t)
                    index.append((i, k, t, L))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model(input_ids=ids.to(0), attention_mask=mask.to(0), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        if rows:
            r, c = torch.tensor(rows, device=0), torch.tensor(cols, device=0)
            chunks.append(torch.stack([hs[l][r, c].float() for l in layers], 1).cpu().numpy().astype(np.float16))
        b += n
    return index, (np.concatenate(chunks) if chunks else np.zeros((0, len(layers), 1), dtype=np.float16))


def wplaces(tag, verses, adapter=None, scale=1.0, layers=None, limit=0):
    """The places of English content words and of every original word of the given verses, for base or an adapter at a strength."""
    import numpy as np
    import torch
    from measure_base import load_model
    from probe import MODELS
    from tongues import KEY, LAYERS, OUT, texts
    cfg = config()
    layers = list(layers or LAYERS)
    W, _ = build(cfg)
    T = texts()
    vs = [v for v in verses if v in W and v in T and len(W[v]["pairs"]) >= 3]
    if limit:
        vs = vs[:limit]
    model_id, revision = MODELS[KEY]
    model, tok = load_model(model_id, revision, True, str(adapter) if adapter else None, cpu_words=True)
    if adapter:
        for m in model.modules():                       # the strength, set outright so that it cannot compound
            if isinstance(getattr(m, "scaling", None), dict):
                for k in m.scaling:
                    m.scaling[k] = scale * m.lora_alpha[k] / m.r[k]
    torch.cuda.reset_peak_memory_stats()
    ie, E = word_states(model, tok, [T[v]["en"] for v in vs], [[p[0] for p in W[v]["pairs"]] for v in vs], layers)
    io_, O = word_states(model, tok, [W[v]["orig"] for v in vs], [W[v]["spans"] for v in vs], layers)
    (OUT / tag).mkdir(parents=True, exist_ok=True)
    np.save(OUT / tag / "english.npy", E)
    np.save(OUT / tag / "original.npy", O)
    (OUT / tag / "index.json").write_text(json.dumps({"verses": vs, "english": ie, "original": io_, "layers": layers, "config": cfg,
                                                      "adapter": str(adapter) if adapter else None, "scale": scale}), encoding="utf-8")
    print(f"{tag}: {len(vs)} verses, {len(ie)} English words, {len(io_)} original words, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)


def wsize(limit=0):
    """Base only: the sample of training chapters `size` read."""
    from tongues import OUT
    wplaces("wsize", json.loads((OUT / "size" / "en.json").read_text(encoding="utf-8")), limit=limit)


def wread(tag="wsize"):
    """Within a verse: is the original word nearest an English word's place the one it renders? Beside it, what the place in the
    verse alone would give."""
    import numpy as np
    from tongues import OUT
    cfg = config()
    W, _ = build(cfg)
    d = json.loads((OUT / tag / "index.json").read_text(encoding="utf-8"))
    E = np.load(OUT / tag / "english.npy").astype(np.float32)
    O = np.load(OUT / tag / "original.npy").astype(np.float32)
    vs = d["verses"]
    lang = np.array([W[vs[i]]["lang"] for i, _, _, _ in d["original"]])
    elang = np.array([W[vs[i]]["lang"] for i, _, _, _ in d["english"]])
    of_verse = defaultdict(list)
    for n, (i, k, t, L) in enumerate(d["original"]):
        of_verse[i].append((k, n, t / L))
    rows = []
    for li, layer in enumerate(d["layers"]):
        e, o = E[:, li].copy(), O[:, li].copy()
        for g in "HG":
            e[elang == g] -= e[elang == g].mean(0)
            o[lang == g] -= o[lang == g].mean(0)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        o /= np.linalg.norm(o, axis=1, keepdims=True)
        res = defaultdict(list)
        for n, (i, k, t, L) in enumerate(d["english"]):
            target = W[vs[i]]["pairs"][k][1]
            cand = of_verse[i]
            ks = [c[0] for c in cand]
            if target not in ks or len(cand) < 4:
                continue
            sims = o[[c[1] for c in cand]] @ e[n]
            own = sims[ks.index(target)]
            rank = 1 + (sims > own).sum()
            pos = np.abs(np.array([c[2] for c in cand]) - t / L)
            prank = 1 + (pos < pos[ks.index(target)]).sum()
            g = W[vs[i]]["lang"]
            res[g].append((rank == 1, 1 / rank, prank == 1, 1 / len(cand)))
        for g, x in res.items():
            x = np.array(x, float)
            rows.append({"layer": layer, "tongue": g, "words": len(x), "first": float(x[:, 0].mean()), "mrr": float(x[:, 1].mean()),
                         "by place in the verse alone": float(x[:, 2].mean()), "chance": float(x[:, 3].mean())})
    (OUT / tag / "read.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{'layer':>5s} {'tongue':6s} {'words':>6s} {'own word nearest':>17s} {'MRR':>6s} {'by place alone':>15s} {'chance':>7s}")
    for r in rows:
        print(f"{r['layer']:5d} {r['tongue']:6s} {r['words']:6d} {r['first']:17.3f} {r['mrr']:6.3f} {r['by place in the verse alone']:15.3f} {r['chance']:7.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("wsize", "wread"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    from measure_base import in_venv, keep_running
    in_venv()
    if sys.argv[1] == "wsize":
        keep_running()
        wsize(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        wread()
