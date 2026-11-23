"""Form B at the word: each English word of the BSB drawn to the Hebrew, Aramaic or Greek word it renders (Clayton, 15 Nov 2026:
"Step 2 is yes too", "Form B"; and on what geometric means, "yes to all 3": the loss may fall on where the model places its
words with no loss on the next word).

WHY THE WORD AND NOT THE VERSE (measured on base, 21 Nov 2026, `size` and `decide`, tongues_words.py `wsize`). The proposal paired
whole verses. Base already puts a verse's own original first among its chapter's originals for 79% of Hebrew verses and 96% of
Greek, so that loss had little to teach. At the word, base's nearest original word in the verse is the English word's own for 72%
(Hebrew) and 78% (Greek); place in the verse alone gives 36% and 32%.

WHAT IS TRAINED. The model reads each verse twice, each time alone after its own start token: the exact BSB verse, and the same
verse in the tongue it was written in (WLC, SBLGNT). The publisher's word-by-word table says which English phrase renders which
original word (tongues_words.py). The loss falls on where the model places words at one middle layer: each English content word
must stand nearer the original word it renders than any other original word in the step. Other occurrences of that same original
word are neither drawn nor held apart. (In the true arm the one word the false arm is drawn to is also left out of what the
English word is held apart from: one other word of the verse.)
- Only the English is drawn. The original is read by base, with the adapter off and without gradients: it is the mark and does
  not move. With both sides free the loss is met as well by moving the originals onto the English.
- No output word is a target, so nothing is trained to recite. The forward pass stops at the trained layer, only the blocks
  below it carry LoRA, and everything above keeps base's weights.

THE TEXT FORMS (Clayton, 15 Nov: "You'll have to make a decision on this. I don't know Hebrew but you do."). Hebrew: the
consonants as written (the ketiv; the margin's qere, the vowel points and the accents are the Masoretes' reading and are left
out). This tokenizer reads pointed Hebrew one letter and one mark at a time (112 to 138 tokens a verse) and unpointed Hebrew in
word pieces (29, the same as the English). Base ranks pointed Hebrew a little better (88% against 79% at the verse), but too few
pointed verses fit in a step. Greek: without accents and breathings (46 tokens against 79; base ranks the two alike).
Left out: a psalm's first verse where the BSB prints a title, and Greek rows the edition divides differently.

THE ARMS, each trained from base on the same steps in the same order, two runs each:
  true      each English word with the original word it renders.
  false     each English word with another word of its own verse (fixed by the seed). The word's own original is left out of
            what it is held apart from, so the arm is not trained against the truth. "The same Bible with its links scrambled."
TinyStories has no original and is left out. The KJV has no word table, so there is no English-with-English arm.
What the false arm is and is not (review, 21 Nov): its target is drawn afresh for each occurrence and each verse is read once, so
there is nothing in it to learn (its target stayed nearest for 6 to 7% of words through training; chance within a verse is 8%). It is
a random pull within the verse, not an equally learnable false rule, and it moves the places nearly three times as far as the true
arm. Both false runs share one pairing and differ in the order of steps and the start of the LoRA weights only.

THE DOSE. 800 steps of about 19 verses: one pass over 14,321 verses (46% of the Bible), 97,659 English words. 30% of chapters
are kept unread as whole chapters, among them Isaiah 11, Genesis 3, Psalm 110, Isaiah 53, Matthew 4, Luke 4 and 24, John 1,
Romans 12 and Philippians 2. Rate 1e-4 with 30 warm-up steps, LoRA rank 16, temperature 0.05, layer 17 of 34.

THE EXACT-WORDS RECIPE (--exact 0.1; added 21 Nov 2026 after the first place-only run, before any adapter of this recipe existed).
The first run learned the pairing and lost its speech: at full strength unread English verses fell by 220 nats each and ordinary
prose by 491; only at a quarter strength did speech hold. Clayton had allowed for this on 15 Nov: "if speech is harmed a small
loss on the exact words may be added beside it". So, beside the place loss, the next word of the exact BSB verse is scored at
384 token positions a step, weight 0.1. The whole model is run; LoRA still sits only on the blocks below the trained layer, so the
upper blocks keep base's weights and the lower ones must leave them something they can read. Same data, steps, arms and seeds.
- The weight was chosen from two 100-step trials (0.1 and 1.0), both thrown away. Both kept unread English verses at or above
  base. 0.1 kept more of the pairing (12 points against 9) and is the small loss that was allowed.
- The readings are the same five, by the same rules, at full and at half strength. This recipe trains the next word on the verses
  it reads, so the share of read verses finished word for word is shown beside the letter: it is how reciting would show.

THE READINGS, fixed on 21 Nov 2026 before any adapter existed (the same text is runs/tongues/READING.md). All on verses no step
read; an unread verse with a twin among the read verses is not counted. Each arm minus base, 95% intervals in which the verses
of one chapter may move together.
1. The pairing was learned and carries (`places`, `read`). On 2,500 unread verses from the kept chapters: the share of English
   words whose nearest original word in the verse is the one it renders, the original read by base (the mark). True minus base
   above zero in both runs. This shows that a word list between the tongues was learned and carries. A lexicon with no context
   would pass it: it shows nothing about reading in context. Beside it: the same with the original read by the arm.
2. The English alone, the result that would count (words_probe.py). Held apart: one English word standing for two different
   original words must stand further apart than the same English word standing for one. Yes only if true minus base is above
   zero in both runs, and true stands above both false runs by more than the two false runs differ from each other. Drawn
   together (two English renderings of one original word stand nearer) is reported beside it as the synonym-list half.
3. Which tongue moved: how far each English word's place and each original word's place stand from where base had them. The
   adapter is on whatever the model reads, so the originals can still move. If they moved as far as the English, that is said.
4. Harm. Every reading is taken at the highest of full, three-quarters, half and a quarter strength at which the letter holds.
   The letter (pairs_probe.py fidelity): the unread verses base holds best, the unread verses base quotes and 300 at random
   must not be shown below base. The originals, and ordinary prose (tongues_harm.py): 300 unread Hebrew verses, 300 unread
   Greek and 300 TinyStories passages must not be shown below base. Replies (chat.py), 4 draws against base: invented words in
   God's or Jesus' mouth, Hebrew or Greek letters in an English reply, a claim to be Christ, no one sent to the Church or a
   professional where base sends them.
5. Shown, not read: the quarter and half checkpoints, round 2's R1 and R2 sets, who is speaking (speakers_probe.py).
A negative means the model remains closer to the pattern of this world. It is never evidence against Scripture.

CORRECTIONS after an independent review (21 Nov 2026, when the first two adapters had been read; none changes a figure):
- Reading 2's intervals resample English words (held apart) and original words (drawn together), not chapters as stated above.
  Recomputed with chapters moving together they are the same to the third decimal.
- Held apart as drawn partly reads the book: its near pairs share a book 24% of the time and its far pairs 9%. Beside the fixed
  reading words_probe.py now gives the triples in which neither pair stands within one book, and `--single` (phrases of one content
  word). The second clause of reading 2 (true above both false runs) is judged on all of them and said so.
- The letter: letter_held is summed over the odd tokens of the rest of the verse; the quoted set is 20 verses in 14 chapters (of
  base's 51, Form B's steps read 26); letter_random is drawn after the 300 of letter_held are taken out.
- "Moved" by 1 - cosine ignores length. `read` also prints the distance moved over the length of base's state.

    python tongues.py data                       # the split, the steps, the false pairing -> runs/tongues/data.json (CPU)
    python tongues.py size                       # base only: a sample in every text form -> runs/tongues/size/ (card)
    python tongues.py decide                     # the table the text forms were chosen from (CPU)
    python tongues.py train --arm true --seed 1  # -> runs/train/ministral3_8b_tongues_true/ (card, about half an hour)
    python tongues.py places --arm base          # word places of the unread sample in both tongues -> runs/tongues/places_<name>/ (card)
    python tongues.py read                       # reading 1 and which tongue moved, for every arm that has places (CPU)
"""
import argparse
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bible import CODES, verses
from measure_base import in_venv, keep_running, load_model, text_config
from probe import MODELS

HERE = Path(__file__).parent
OUT = HERE / "runs" / "tongues"
TRAIN = HERE / "runs" / "train"
STUDY = HERE / "runs" / "design_study_2026-11-15"
KEY = "ministral3_8b"
SEED = 20261115
HELD = 0.30                # share of chapters kept unread, as whole chapters
NAMED = ("Isa.11", "Gen.3", "Psa.110", "Isa.53", "Mat.4", "Luk.4", "Luk.24", "Joh.1", "Rom.12", "Phi.2")  # kept unread by name
LAYERS = (8, 17, 25)       # where `size` looks
BUDGET_EN = 1200           # padded tokens of English a training step (the side that carries gradients)
BUDGET_OR = 2400           # padded tokens of the original a step (read without gradients)
MAX_VERSES = 24            # verses a step
MIN_STEP = 6               # verses: fewer leave too little to hold apart
MAX_STEPS = 800
WARMUP = 30
RANK = 16
LR = 1e-4
TEMP = 0.05
EXACT_TOKENS = 384         # the exact-words recipe: token positions of the English verses scored a step
ARMS = ("true", "false")
DEFAULT = {"hebrew": "consonants", "greek": "plain", "pool": "mean", "layer": 17}

ACCENTS = re.compile("[֑-ֽ֯]")                                   # cantillation and meteg
POINTS = re.compile("[ְ-ׇּֿׁׂׅׄ]")      # vowel points, dagesh, rafe, shin and sin dots
MARKS = re.compile("[־׀׃׆\\[\\]]")                           # maqqef, paseq, sof pasuq, nun hafukha, brackets


def jl(path):
    for line in open(path, encoding="utf-8"):
        yield json.loads(line)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def hebrew(row, form):
    """A WLC row in one of two forms. consonants: the words as written (ketiv), no points or accents, maqqef as a space.
    points: the printed verse without accents and without the bracketed qere."""
    if form == "consonants":
        words = [POINTS.sub("", ACCENTS.sub("", w["form"])) for w in row["words"] if w.get("type") != "qere"]
        return re.sub(r"\s+", " ", MARKS.sub(" ", " ".join(words))).strip()
    text = re.sub(r"\s*\[[^\]]*\]\s*", " ", row["text"])
    return re.sub(r"\s+", " ", ACCENTS.sub("", text)).strip()


def greek(row, form):
    """An SBLGNT row as printed, or plain: without accents, breathings, iota subscript or diaeresis (the letters as written)."""
    if form == "printed":
        return row["text"]
    d = unicodedata.normalize("NFD", row["text"])
    return unicodedata.normalize("NFC", "".join(c for c in d if unicodedata.category(c) != "Mn"))


def texts():
    """Every verse that has exactly one row in its original tongue: its English (BSB, KJV) and each form of the original.
    Left out: a psalm's first verse where the BSB prints a title (the Hebrew row holds the title and the English text does not),
    and Greek rows the edition divides differently from the BSB (three or more words apart from the publisher's own Greek)."""
    bsb, kjv = verses("BSB"), verses("KJV")
    titled = {r["id"] for r in jl(HERE / "data" / "BSB_bible_segments.jsonl") if r.get("title")}
    table_words = {r["id"]: len(r["words"]) for r in jl(HERE / "data" / "orig" / "BSB_tables.jsonl")}
    out = {}
    for file, lang in (("WLC.jsonl", "H"), ("SBLGNT.jsonl", "G")):
        for r in jl(HERE / "data" / "orig" / file):
            v = r["id"]
            if v not in bsb or len(r.get("refs", [])) != 1 or not r["words"] or v in titled:
                continue
            if lang == "G" and abs(len(r["words"]) - table_words.get(v, len(r["words"]))) >= 3:
                continue
            forms = {"consonants": hebrew(r, "consonants"), "points": hebrew(r, "points")} if lang == "H" else \
                    {"plain": greek(r, "plain"), "printed": greek(r, "printed")}
            if not all(forms.values()):
                continue
            out[v] = {"lang": lang, "en": bsb[v], "kjv": kjv.get(v), **forms}
    return out


def order_key(v):
    b, c, n = v.split(".")
    return CODES.index(b), int(c), int(n)


def chapter(v):
    return v.rsplit(".", 1)[0]


def config():
    p = OUT / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else dict(DEFAULT)


def original(t, cfg):
    return t[cfg["hebrew"]] if t["lang"] == "H" else t[cfg["greek"]]


def tokenizer():
    from transformers import AutoTokenizer
    model_id, revision = MODELS[KEY]
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def texts_all_ids():
    """Every BSB verse id (the split is drawn over all chapters, so it does not move when texts() leaves a verse out)."""
    return list(verses("BSB"))


def data(args):
    """The split, the steps and the false pairing, at the word. The split depends only on the seed."""
    import tongues_words
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config()
    T = texts()
    W, wstats = tongues_words.build(cfg)
    tok = tokenizer()
    ids = sorted((v for v in T if v in W and W[v]["pairs"]), key=order_key)
    chapters = defaultdict(list)
    for v in ids:
        chapters[chapter(v)].append(v)
    names = sorted(chapters, key=lambda c: order_key(c + ".1"))
    all_names = sorted({chapter(v) for v in texts_all_ids()}, key=lambda c: order_key(c + ".1"))
    rng = random.Random(f"{SEED} split")
    held = set(rng.sample(all_names, round(HELD * len(all_names)))) | set(NAMED)
    n_tok = lambda s: len(tok(s, add_special_tokens=True)["input_ids"])
    length = {v: (n_tok(T[v]["en"]), n_tok(W[v]["orig"])) for v in ids}
    train_ch = [c for c in names if c not in held]
    random.Random(f"{SEED} chapters").shuffle(train_ch)
    steps = []
    for c in train_ch:
        run, w_en, w_or, runs = [], 0, 0, []
        for v in chapters[c]:
            e, o = length[v]
            if run and ((len(run) + 1) * max(w_en, e) > BUDGET_EN or (len(run) + 1) * max(w_or, o) > BUDGET_OR or len(run) >= MAX_VERSES):
                runs.append(run)
                run, w_en, w_or = [], 0, 0
            run.append(v)
            w_en, w_or = max(w_en, e), max(w_or, o)
        if run:
            runs.append(run)
        if len(runs) > 1 and len(runs[-1]) < MIN_STEP:
            runs[-2:] = [runs[-2] + runs[-1]]
        steps += [{"chapter": c, "verses": r} for r in runs if len(r) >= MIN_STEP]
        if len(steps) >= MAX_STEPS:
            break
    steps = steps[:MAX_STEPS]
    read = {v for s in steps for v in s["verses"]}
    # the false pairing: for each English word, another word of its own verse that is not the same original word
    false = {}
    for v in sorted(read, key=order_key):
        prng = random.Random(f"{SEED} false {v}")
        lex = W[v]["lex"]
        row = []
        for span, k, strong, key, pos, early in W[v]["pairs"]:
            cands = [i for i in range(len(lex)) if lex[i] != lex[k]]
            row.append(prng.choice(cands) if cands else -1)
        false[v] = row
    # unread: every verse no step read; a verse with a twin among the read verses is not counted
    def grams(v):
        w = [W[v]["orig"][a:b] for a, b in W[v]["spans"]]
        return {tuple(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
    seen = set()
    for v in read:
        seen |= grams(v)
    read_text = {T[v]["en"] for v in read}
    twins = defaultdict(set)
    for r in jl(STUDY / "relations" / "relations.jsonl"):
        if r["layer"] in ("c parallel passage", "a quotation"):
            twins[r["a"]].add(r["b"])
            twins[r["b"]].add(r["a"])
    unread_all = [v for v in ids if v not in read]
    leaky = {v for v in unread_all if grams(v) & seen or T[v]["en"] in read_text or twins.get(v, set()) & read}
    unread = [v for v in unread_all if v not in leaky]
    held_unread = [v for v in unread if chapter(v) in held]
    ok = [v for v in held_unread if len(W[v]["pairs"]) >= 3]
    sample = sorted(random.Random(f"{SEED} sample").sample(ok, min(2500, len(ok))), key=order_key)
    stats = {"verses joined at the word": len(ids), "chapters kept unread": len(held), "steps": len(steps), "verses read": len(read),
             "verses read, Hebrew": sum(T[v]["lang"] == "H" for v in read), "verses read, Greek": sum(T[v]["lang"] == "G" for v in read),
             "English words trained on": sum(sum(k >= 0 for k in false[v]) for v in read),
             "verses a step, median": float(np.median([len(s["verses"]) for s in steps])),
             "training chapters reached": len({s["chapter"] for s in steps}), "of training chapters": len(train_ch),
             "unread verses": len(unread_all), "of them with a twin among the read (not counted)": len(leaky),
             "unread in kept chapters, counted": len(held_unread), "sample read for the readings": len(sample), "join": wstats}
    out = {"seed": SEED, "config": cfg, "held_chapters": sorted(held, key=lambda c: order_key(c + ".1")), "steps": steps, "false": false,
           "unread": unread, "unread_sample": sample, "stats": stats, "bsb_sha": sha(HERE / "data" / "BSB_bible_segments.jsonl")}
    (OUT / "data.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (OUT / "unread.json").write_text(json.dumps(unread), encoding="utf-8")
    (OUT / "read_verses.json").write_text(json.dumps(sorted(read, key=order_key)), encoding="utf-8")
    print(json.dumps(stats, indent=1))


# ---------------------------------------------------------------- the model

def decoder(model):
    """The list of decoder blocks of the text model, and the module that holds it."""
    import torch
    n = text_config(model).num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers") and not re.search("vision|audio", name):
            return model.get_submodule(name.rsplit(".", 1)[0]), mod, name
    raise SystemExit("the decoder blocks were not found")


def truncate(model, layer):
    """Stop the forward pass at `layer`: the blocks above it are dropped and the closing norm is removed, so hidden_states[-1]
    is the state after block layer-1, which is hidden_states[layer] of the whole model (checked on the card in `train`)."""
    import torch
    import gc
    parent, blocks, name = decoder(model)
    parent.layers = torch.nn.ModuleList(list(blocks)[:layer])
    parent.norm = torch.nn.Identity()      # the text model ends with a norm; without this the last state returned is the normed one
    del blocks
    gc.collect()                           # a dropped 4-bit block is held by its own weight until a full collection
    torch.cuda.empty_cache()
    return name


def pooled(h, enc, tok, how):
    """A verse's place: the mean of its tokens' states, or the state at its last token. The start token is not part of the verse."""
    import torch
    mask = enc["attention_mask"].clone()
    if tok.bos_token_id is not None:
        mask[enc["input_ids"] == tok.bos_token_id] = 0
    if how == "mean":
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
    if tok.padding_side == "left":
        last = torch.full((h.shape[0],), h.shape[1] - 1, device=h.device)
    else:
        last = enc["attention_mask"].sum(1) - 1
    return h[torch.arange(h.shape[0], device=h.device), last]


def states(model, tok, batch, layers=None):
    """hidden states of a batch of texts, each alone after the start token. layers=None: the last state the model computes."""
    enc = tok(batch, add_special_tokens=True, padding=True, return_tensors="pt").to(0)
    hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
    return enc, (hs[-1] if layers is None else [hs[l] for l in layers])


def embed(model, tok, strings, layers, budget=1536):
    """[n, layers, 2, dim] float16: each text's place at each layer, pooled both ways (0 mean, 1 last). layers=None after truncate."""
    import torch
    n_tok = [len(tok(s, add_special_tokens=True)["input_ids"]) for s in strings]
    order = sorted(range(len(strings)), key=lambda i: n_tok[i])
    dim = text_config(model).hidden_size
    out = np.zeros((len(strings), 1 if layers is None else len(layers), 2, dim), dtype=np.float16)
    b = 0
    while b < len(order):
        n = 1
        while b + n < len(order) and (n + 1) * n_tok[order[b + n]] <= budget:
            n += 1
        idx = order[b:b + n]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            enc, hs = states(model, tok, [strings[i] for i in idx], layers)
        for j, h in enumerate([hs] if layers is None else hs):
            h = h.float()
            out[idx, j, 0] = pooled(h, enc, tok, "mean").cpu().numpy().astype(np.float16)
            out[idx, j, 1] = pooled(h, enc, tok, "last").cpu().numpy().astype(np.float16)
        b += n
    return out


def size(args):
    """Base only, forward passes: a sample of training chapters in English (BSB, KJV) and in every form of the original, at three
    layers, pooled both ways. Nothing is trained. The unread chapters are not looked at."""
    import torch
    d = json.loads((OUT / "data.json").read_text(encoding="utf-8"))
    T = texts()
    held = set(d["held_chapters"])
    chapters = defaultdict(list)
    for v in sorted(T, key=order_key):
        if chapter(v) not in held and T[v]["kjv"]:
            chapters[chapter(v)].append(v)
    rng = random.Random(f"{SEED} size")
    names = sorted(chapters, key=lambda c: order_key(c + ".1"))
    rng.shuffle(names)
    want, got, pick = {"H": args.hebrew, "G": args.greek}, Counter(), []
    for c in names:
        lang = T[chapters[c][0]]["lang"]
        if got[lang] < want[lang]:
            pick += chapters[c]
            got[lang] += len(chapters[c])
    if args.limit:
        pick = pick[:args.limit]
    model_id, revision = MODELS[KEY]
    model, tok = load_model(model_id, revision, True, cpu_words=True)
    torch.cuda.reset_peak_memory_stats()
    (OUT / "size").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    forms = [("en", pick), ("kjv", pick), ("consonants", [v for v in pick if T[v]["lang"] == "H"]), ("points", [v for v in pick if T[v]["lang"] == "H"]),
             ("plain", [v for v in pick if T[v]["lang"] == "G"]), ("printed", [v for v in pick if T[v]["lang"] == "G"])]
    for name, vs in forms:
        arr = embed(model, tok, [T[v][name] for v in vs], LAYERS, budget=args.budget)
        np.save(OUT / "size" / f"{name}.npy", arr)
        (OUT / "size" / f"{name}.json").write_text(json.dumps(vs), encoding="utf-8")
        print(f"size {name}: {len(vs)} verses, {time.time() - t0:.0f} s, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)


def ranks(a, b, groups):
    """For each row i of a, the rank (1 = nearest) of b[i] among the rows of b in i's group, by cosine. a and b are centred by the caller."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    out = np.zeros(len(a))
    for g in groups:
        g = np.asarray(g)
        s = a[g] @ b[g].T
        own = np.diag(s)
        out[g] = 1 + (s > own[:, None]).sum(1) + 0.5 * ((s == own[:, None]).sum(1) - 1)
    return out


def by_chapter(vs):
    g = defaultdict(list)
    for i, v in enumerate(vs):
        g[chapter(v)].append(i)
    return [x for x in g.values() if len(x) >= 2]


def decide(args):
    """Which text forms, pooling and layer base already places best, on the sample `size` read. Printed in full; the choice is
    written to config.json. Beside each figure: what the verse's length alone would give."""
    T = texts()
    tok = tokenizer()
    S = OUT / "size"
    load = lambda name: (np.load(S / f"{name}.npy").astype(np.float32), json.loads((S / f"{name}.json").read_text(encoding="utf-8")))
    en, en_ids = load("en")
    row = {v: i for i, v in enumerate(en_ids)}
    table, best = [], {}
    for lang, forms in (("H", ("consonants", "points")), ("G", ("plain", "printed"))):
        for form in forms + ("kjv",):
            arr, vs = load(form)
            keep = [i for i, v in enumerate(vs) if v in T and (form != "kjv" or T[v]["lang"] == lang)]   # texts() has since left some verses out
            arr, vs = arr[keep], [vs[i] for i in keep]
            e = en[[row[v] for v in vs]]
            groups = by_chapter(vs)
            n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
            le, lo = np.array([n_tok(T[v]["en"]) for v in vs], float), np.array([n_tok(T[v][form]) for v in vs], float)
            z = lambda x: (x - x.mean()) / x.std()
            length_rank = np.zeros(len(vs))
            for g in groups:
                g = np.asarray(g)
                dlt = np.abs(z(le)[g][:, None] - z(lo)[g][None, :])
                own = np.diag(dlt)
                length_rank[g] = 1 + (dlt < own[:, None]).sum(1) + 0.5 * ((dlt == own[:, None]).sum(1) - 1)
            members = [i for g in groups for i in g]
            base_line = {"mrr": float((1 / length_rank[members]).mean()), "first": float((length_rank[members] == 1).mean())}
            for li, layer in enumerate(LAYERS):
                for pi, pool in enumerate(("mean", "last")):
                    a, b = e[:, li, pi], arr[:, li, pi]
                    a, b = a - a.mean(0), b - b.mean(0)
                    r = ranks(a, b, groups)[members]
                    rec = {"lang": lang, "form": form, "layer": layer, "pool": pool, "verses": len(members), "chapters": len(groups),
                           "mrr": float((1 / r).mean()), "first": float((r == 1).mean()), "length_only_mrr": base_line["mrr"],
                           "tokens": float(lo.mean())}
                    table.append(rec)
    print(f"{'tongue':7s} {'form':11s} {'layer':>5s} {'pool':5s} {'verses':>6s}  {'MRR':>6s} {'first':>6s}  {'length alone':>12s} {'tokens':>7s}")
    for r in table:
        print(f"{r['lang']:7s} {r['form']:11s} {r['layer']:5d} {r['pool']:5s} {r['verses']:6d}  {r['mrr']:6.3f} {r['first']:6.3f}  {r['length_only_mrr']:12.3f} {r['tokens']:7.1f}")
    (OUT / "size_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print("\nThe choice is made by the researcher from this table and written with --write hebrew=... greek=... pool=... layer=...")
    if args.write:
        cfg = dict(DEFAULT)
        for kv in args.write:
            k, v = kv.split("=")
            cfg[k] = int(v) if k == "layer" else v
        (OUT / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        print("written:", cfg)


def name_of(arm, seed, exact=0.0):
    """Folder name of a run. exact > 0: the recipe with the small loss on the exact words beside the place loss ("_w")."""
    return f"{KEY}_tongues_{arm}{'_w' if exact else ''}{'' if seed == 1 else f'_s{seed}'}"


def train(args):
    """The word-level loss. English is read with gradients through the adapter; the original is read by base, with the adapter
    off and without gradients (the original is the mark and does not move; the English is drawn to it).
    For each English content word: its place must stand nearer the place of the original word it renders than any other original
    word in the step. Other occurrences of that same original word are neither drawn nor held apart."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    import tongues_words
    from tongues_words import last_token

    torch.cuda.set_per_process_memory_fraction(0.98)
    cfg = config()
    d = json.loads((OUT / "data.json").read_text(encoding="utf-8"))
    assert d["config"] == cfg, "data.json was built for another choice of text forms: run `data` again"
    T = texts()
    W, _ = tongues_words.build(cfg)
    L = cfg["layer"]
    out = TRAIN / (name_of(args.arm, args.seed, args.exact) + (f"_limit{args.limit}" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    seed = SEED + args.seed - 1
    steps = list(d["steps"])
    random.Random(seed).shuffle(steps)            # the same order for every arm of one seed
    if args.limit:
        steps = steps[:args.limit]
    torch.manual_seed(seed)
    model_id, revision = MODELS[KEY]
    model, tok = load_model(model_id, revision, True, cpu_words=True)
    for p in model.parameters():
        p.requires_grad = False
    probe = tok([T[v]["en"] for v in steps[0]["verses"][:6]], add_special_tokens=True, padding=True, return_tensors="pt").to(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"], output_hidden_states=True, use_cache=False, logits_to_keep=1)
    head = model.get_output_embeddings()
    if args.exact:
        # pre-flight: the output layer on the last state returned must give the model's own logits (the last state is the normed one)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            mine = head(o.hidden_states[-1][:, -1:]).float()
        gap = float((mine - o.logits.float()).abs().max() / o.logits.float().abs().max())
        print(f"pre-flight: the output layer on the last state against the model's logits, largest difference {gap:.2e} of the largest value", flush=True)
        assert gap < 1e-2, "the last hidden state is not what the output layer reads"
        del mine
    else:
        # pre-flight: the truncated pass must give the whole model's state at the trained layer
        whole = o.hidden_states[L].float()
        truncate(model, L)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cut = model(input_ids=probe["input_ids"], attention_mask=probe["attention_mask"], output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states[-1].float()
        gap = float((whole - cut).abs().max() / whole.abs().max())
        print(f"pre-flight: truncated against whole, largest difference {gap:.2e} of the largest value; card holds {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        assert gap < 1e-3, "the truncated pass does not give the whole model's state at the trained layer"
        del whole, cut
    del o
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    if hasattr(text_config(model), "use_cache"):
        text_config(model).use_cache = False
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                             layers_to_transform=list(range(L)), layers_pattern="layers", task_type="FEATURE_EXTRACTION"))
    n_lora = sum("lora_A" in n for n, _ in model.named_parameters())
    assert n_lora == 7 * L, f"LoRA sits on {n_lora} layers, not {7 * L}"
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    marks = {max(1, len(steps) // 4): "adapter_25", max(2, len(steps) // 2): "adapter_50"} if len(steps) >= 8 else {}
    log, t0, tokens, words, stop_note = [], time.time(), 0, 0, None
    acc = Counter()

    pick = torch.Generator().manual_seed(seed)

    def forward(strings):
        """The places at the trained layer; with the exact-words recipe also the last state, which the output layer reads."""
        enc = tok(strings, add_special_tokens=True, padding=True, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping").tolist()
        enc = enc.to(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        return enc, offs, (hs[L] if args.exact else hs[-1]), (hs[-1] if args.exact else None)
    for n, s in enumerate(steps):
        vs = s["verses"]
        with torch.no_grad(), model.disable_adapter():        # the mark does not move: the original as base reads it
            enc_o, off_o, ho, _ = forward([W[v]["orig"] for v in vs])
        o_rows, o_cols, o_lex, o_at = [], [], [], {}
        for j, v in enumerate(vs):
            for k, (a, b) in enumerate(W[v]["spans"]):
                t = last_token(off_o[j], a, b)
                if t is not None:
                    o_at[(j, k)] = len(o_rows)
                    o_rows.append(j)
                    o_cols.append(t)
                    o_lex.append(W[v]["lex"][k])
        enc_e, off_e, he, last = forward([T[v]["en"] for v in vs])
        e_rows, e_cols, own, other = [], [], [], []
        for j, v in enumerate(vs):
            for (span, k, strong, key, pos, early), kf in zip(W[v]["pairs"], d["false"][v]):
                t = last_token(off_e[j], span[0], span[1])
                if t is None or kf < 0 or (j, k) not in o_at or (j, kf) not in o_at:
                    continue
                e_rows.append(j)
                e_cols.append(t)
                own.append(o_at[(j, k)])
                other.append(o_at[(j, kf)])
        if len(e_rows) < 8:
            continue
        E = he[torch.tensor(e_rows, device=0), torch.tensor(e_cols, device=0)].float()
        O = ho[torch.tensor(o_rows, device=0), torch.tensor(o_cols, device=0)].float()
        E = F.normalize(E - E.mean(0, keepdim=True), dim=-1)
        O = F.normalize(O - O.mean(0, keepdim=True), dim=-1)
        sim = E @ O.T / TEMP
        own_t, other_t = torch.tensor(own, device=0), torch.tensor(other, device=0)
        target = own_t if args.arm == "true" else other_t
        code = {x: i for i, x in enumerate(sorted(set(o_lex)))}
        lex = torch.tensor([code[x] for x in o_lex], device=0)
        col = torch.arange(len(o_lex), device=0)[None, :]
        same_as = lambda t: (lex[None, :] == lex[t][:, None]) & (col != t[:, None])     # other occurrences of that word
        drop = same_as(target)
        if args.arm == "true":
            drop = drop | (col == other_t[:, None])                      # keep the number of candidates the same in both arms
        else:
            drop = drop | (col == own_t[:, None]) | same_as(own_t)       # the false arm is never pushed from the word's own original
        drop = drop & (col != target[:, None])
        masked = sim.masked_fill(drop, float("-inf"))
        place = F.cross_entropy(masked, target)
        loss, said = place, None
        if args.exact:
            # the exact words: the next token of the BSB verse, at a sample of its positions, read by the unchanged upper blocks
            m = enc_e["attention_mask"].clone()
            m[:, 0] = 0                                          # nothing predicts the start token
            at = m.nonzero()
            at = at[torch.randperm(len(at), generator=pick)[:EXACT_TOKENS].to(at.device)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(last[at[:, 0], at[:, 1] - 1])
            said = F.cross_entropy(logits.float(), enc_e["input_ids"][at[:, 0], at[:, 1]])
            loss = place + args.exact * said
        if not torch.isfinite(loss):
            stop_note = f"loss not finite at step {n}"
            break
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        tokens += int(enc_e["attention_mask"].sum()) + int(enc_o["attention_mask"].sum())
        words += len(e_rows)
        acc["n"] += len(e_rows)
        acc["own"] += int((sim.detach().argmax(1) == own_t).sum())
        acc["target"] += int((masked.detach().argmax(1) == target).sum())
        if n % args.log_every == 0:
            rec = {"step": n, "loss": round(place.item(), 4), **({"exact words, nats a token": round(said.item(), 3)} if said is not None else {}), "grad": round(norm, 3), "seconds": round(time.time() - t0), "words": words,
                   "own original nearest": round(acc["own"] / acc["n"], 3), "target nearest": round(acc["target"] / acc["n"], 3),
                   "verses": len(vs), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            log.append(rec)
            print(rec, flush=True)
            acc = Counter()
            torch.cuda.empty_cache()
        if n + 1 in marks:
            model.save_pretrained(out / marks[n + 1])
        if time.time() - t0 > args.max_minutes * 60:
            stop_note = f"time cap reached at step {n} of {len(steps)}"
            break
    model.save_pretrained(out / "adapter")
    summary = {"key": KEY, "arm": f"tongues_{args.arm}", "model": model_id, "revision": revision, "seed": seed, "rank": RANK, "lr": LR, "temperature": TEMP,
               "layer": L, "level": "word", "exact_weight": args.exact, "exact_tokens": EXACT_TOKENS if args.exact else 0, "config": cfg, "steps": len(steps), "english words": words, "tokens": tokens, "lora_layers": n_lora,
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "minutes": round((time.time() - t0) / 60, 1),
               "stop_note": stop_note, "data_sha": sha(OUT / "data.json"), "log": log}
    (out / "train.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("saved", out / "adapter", {k: v for k, v in summary.items() if k != "log"}, flush=True)


def tag_of(args):
    name = "base" if args.arm == "base" else name_of(args.arm, args.seed, args.exact)
    return name + (f"_c{args.checkpoint}" if args.checkpoint else "") + (f"_x{args.scale:g}" if args.scale != 1 else "")


def places(args):
    """The places of English words and original words in a sample of unread verses, at the trained layer, for base or an adapter."""
    import tongues_words
    cfg = config()
    d = json.loads((OUT / "data.json").read_text(encoding="utf-8"))
    adapter = None
    if args.arm != "base":
        folder = TRAIN / name_of(args.arm, args.seed, args.exact)
        t = json.loads((folder / "train.json").read_text(encoding="utf-8"))
        if t["stop_note"]:
            raise SystemExit(f"{folder.name} is void: {t['stop_note']}")
        adapter = folder / ("adapter" if not args.checkpoint else f"adapter_{args.checkpoint}")
    tongues_words.wplaces("places_" + tag_of(args), d["unread_sample"], adapter, args.scale, [cfg["layer"]], args.limit)


def cluster_interval(x, groups):
    """Mean of x with a 95% interval in which the items of one group (a chapter) may move together."""
    x = np.asarray(x, float)
    e = x - x.mean()
    s = np.array([e[g].sum() for g in groups])
    k = len(groups)
    sd = math.sqrt(k / (k - 1) * (s ** 2).sum()) / len(x)
    return float(x.mean()), float(x.mean() - 1.96 * sd), float(x.mean() + 1.96 * sd)


def read(args):
    """Reading 1 at the word, on unread verses: is the original word nearest an English word's place the one it renders? Each arm
    against base on the same words, twice: with the original as base reads it (the mark the English was drawn to) and with the
    original as the arm itself reads it. And which tongue moved: how far each side's places stand from where base had them."""
    import tongues_words
    cfg = config()
    W, _ = tongues_words.build(cfg)
    tags = sorted(p.name[len("places_"):] for p in OUT.glob("places_*") if (p / "index.json").exists())
    assert "base" in tags, "run `places --arm base` first"

    def load(tag):
        d = json.loads((OUT / f"places_{tag}" / "index.json").read_text(encoding="utf-8"))
        E = np.load(OUT / f"places_{tag}" / "english.npy").astype(np.float32)[:, 0]
        O = np.load(OUT / f"places_{tag}" / "original.npy").astype(np.float32)[:, 0]
        vs = d["verses"]
        el = np.array([W[vs[i]]["lang"] for i, _, _, _ in d["english"]])
        ol = np.array([W[vs[i]]["lang"] for i, _, _, _ in d["original"]])
        e, o = E.copy(), O.copy()
        for g in "HG":
            e[el == g] -= e[el == g].mean(0)
            o[ol == g] -= o[ol == g].mean(0)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        o /= np.linalg.norm(o, axis=1, keepdims=True)
        return {"E": E, "O": O, "e": e, "o": o, "vs": vs, "english": d["english"], "original": d["original"], "el": el, "ol": ol}

    def nearest(r, o):
        """1 where the English word's nearest original word in its verse is its own; nan where the word cannot be scored."""
        vs = r["vs"]
        of_verse = defaultdict(list)
        for n, (i, k, t, L) in enumerate(r["original"]):
            of_verse[i].append((k, n))
        first = np.full(len(r["english"]), np.nan)
        for n, (i, k, t, L) in enumerate(r["english"]):
            target = W[vs[i]]["pairs"][k][1]
            ks = [c[0] for c in of_verse[i]]
            if target not in ks or len(ks) < 4:
                continue
            lex = W[vs[i]]["lex"]
            keep = [c for c in of_verse[i] if c[0] == target or lex[c[0]] != lex[target]]    # another occurrence of the same word is not a miss
            sims = o[[c[1] for c in keep]] @ r["e"][n]
            first[n] = float(sims.argmax() == [c[0] for c in keep].index(target))
        return first
    base = load("base")
    base["first"] = nearest(base, base["o"])
    chap = [chapter(base["vs"][i]) for i, _, _, _ in base["english"]]
    res = {}
    cos = lambda a, b: (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print("own word nearest: the share of English words whose nearest original word in the verse is the one it renders.")
    print("'mark' reads the original by base (what the English was drawn to); 'arm' reads the original by the arm itself.\n")
    print(f"{'arm':40s} {'tongue':7s} {'words':>6s} {'mark':>6s} {'minus base':>26s} {'arm':>6s} {'minus base':>26s}   moved: English, original (1 - cosine)   (|arm - base| / |base|)")
    for tag in tags:
        r = base if tag == "base" else load(tag)
        assert r["vs"] == base["vs"] and len(r["english"]) == len(base["english"])
        if tag != "base":
            r["first"] = nearest(r, r["o"])
            r["mark"] = nearest(r, base["o"])
        else:
            r["mark"] = r["first"]
        rec = {}
        for part, sel in (("all", np.ones(len(chap), bool)), ("Hebrew", base["el"] == "H"), ("Greek", base["el"] == "G")):
            ok = sel & ~np.isnan(base["first"]) & ~np.isnan(r["first"]) & ~np.isnan(r["mark"])
            idx = np.where(ok)[0]
            groups = defaultdict(list)
            for pos, i in enumerate(idx):
                groups[chap[i]].append(pos)
            row = {"words": int(len(idx)), "mark": float(r["mark"][idx].mean()), "arm": float(r["first"][idx].mean())}
            if tag != "base":
                row["mark minus base"] = cluster_interval(r["mark"][idx] - base["first"][idx], list(groups.values()))
                row["arm minus base"] = cluster_interval(r["first"][idx] - base["first"][idx], list(groups.values()))
                osel = np.ones(len(base["ol"]), bool) if part == "all" else base["ol"] == part[0]
                row["English moved"] = float((1 - cos(r["E"][sel], base["E"][sel])).mean())
                row["original moved"] = float((1 - cos(r["O"][osel], base["O"][osel])).mean())
                far = lambda a, b: float((np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)).mean())   # 1 - cosine ignores length
                row["English moved, distance"] = far(r["E"][sel], base["E"][sel])
                row["original moved, distance"] = far(r["O"][osel], base["O"][osel])
            rec[part] = row
            f = lambda k: "" if tag == "base" else "{:+.3f} ({:+.3f} to {:+.3f})".format(*row[k])
            mv = "" if tag == "base" else f"   {row['English moved']:.4f}, {row['original moved']:.4f}   {row['English moved, distance']:.3f}, {row['original moved, distance']:.3f}"
            print(f"{tag:40s} {part:7s} {row['words']:6d} {row['mark']:6.3f} {f('mark minus base'):>26s} {row['arm']:6.3f} {f('arm minus base'):>26s}{mv}")
        res[tag] = rec
    (OUT / "read.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def lexicon(args):
    """What a word list with no context scores on reading 1 (CPU; nothing is read by the model). The list is counted from the read
    verses: for each English word, how often it rendered each original word. On the unread sample each English word picks, among its
    verse's original words, the one its list names most often; a word the list does not know picks the nearest by place in the verse."""
    import tongues_words
    cfg = config()
    W, _ = tongues_words.build(cfg)
    d = json.loads((OUT / "data.json").read_text(encoding="utf-8"))
    read_vs = [v for s in d["steps"] for v in s["verses"]]
    count = defaultdict(Counter)
    for v in read_vs:
        for span, k, strong, key, pos, early in W[v]["pairs"]:
            count[key][W[v]["lex"][k]] += 1
    idx = json.loads((OUT / "places_base" / "index.json").read_text(encoding="utf-8"))
    vs = idx["verses"]
    of_verse = defaultdict(list)
    for n, (i, k, t, L) in enumerate(idx["original"]):
        of_verse[i].append((k, t / L))
    res = defaultdict(list)
    for i, k, t, L in idx["english"]:
        v = vs[i]
        span, target, strong, key, pos, early = W[v]["pairs"][k]
        cand = of_verse[i]
        if target not in [c[0] for c in cand] or len(cand) < 4:
            continue
        lex = W[v]["lex"]
        cand = [c for c in cand if c[0] == target or lex[c[0]] != lex[target]]    # as in read(): another occurrence of the same word is not a miss
        known = max((count[key][lex[c[0]]] for c in cand), default=0)
        if known:
            best = [c for c in cand if count[key][lex[c[0]]] == known]
        else:
            best = cand
        pick = min(best, key=lambda c: abs(c[1] - t / L))
        res[W[v]["lang"]].append((pick[0] == target, bool(known)))
    print(f"{'tongue':7s} {'words':>6s} {'the list picks its own word':>28s} {'words the list knows':>21s}")
    out = {}
    for g, x in list(res.items()) + [("all", [y for x in res.values() for y in x])]:
        x = np.array(x, float)
        out[g] = {"words": len(x), "own": float(x[:, 0].mean()), "known": float(x[:, 1].mean())}
        print(f"{g:7s} {len(x):6d} {x[:, 0].mean():28.3f} {x[:, 1].mean():21.3f}")
    (OUT / "lexicon.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data").set_defaults(fn=data)
    s = sub.add_parser("size")
    s.add_argument("--hebrew", type=int, default=2000)
    s.add_argument("--greek", type=int, default=1500)
    s.add_argument("--budget", type=int, default=1536)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=size)
    s = sub.add_parser("decide")
    s.add_argument("--write", nargs="*")
    s.set_defaults(fn=decide)
    s = sub.add_parser("train")
    s.add_argument("--arm", required=True, choices=ARMS)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--log-every", type=int, default=20)
    s.add_argument("--max-minutes", type=float, default=120)
    s.add_argument("--exact", type=float, default=0.0, help="weight of the loss on the exact words of the BSB verse beside the place loss (0: places only)")
    s.set_defaults(fn=train)
    s = sub.add_parser("places")
    s.add_argument("--arm", required=True, choices=("base",) + ARMS)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--checkpoint", choices=("25", "50"))
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--exact", type=float, default=0.0, help="any value above 0 picks the runs trained with the loss on the exact words")
    s.set_defaults(fn=places)
    sub.add_parser("read").set_defaults(fn=read)
    sub.add_parser("lexicon").set_defaults(fn=lexicon)
    args = p.parse_args()
    in_venv()
    if args.cmd in ("size", "train", "places"):
        keep_running()
    args.fn(args)


if __name__ == "__main__":
    main()
