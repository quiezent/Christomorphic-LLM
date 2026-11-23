"""The words in context (step 1 of runs/train/PROPOSAL_2026-11-15.md): where does the
model place each English phrase of the BSB, read by the Hebrew, Aramaic or Greek word behind it? Forward passes only; nothing is trained
here. The same readings are taken on form B's adapters.

THE TABLE (`table`, CPU). The publisher's word-by-word table (data/orig/BSB_tables.jsonl) cuts each BSB verse into English phrases and
says which phrase renders each original word. A row is a phrase that renders exactly one noun, verb or adjective and no proper name.
- The part of speech is the table's own tag. A Hebrew tag also names prefixes and suffixes ("Conj-w | N-fs"): the word's own part is
  read, not the tag's first letters. The design study read the first letters and so missed every Hebrew word with a prefixed "and",
  preposition or article: its count was 164,141 phrases.
- Proper names: the Hebrew tag marks them. Greek has no such tag, so a Greek noun or adjective counts as a name when every form of it
  is capitalised, one of them inside a verse, and its English is capitalised too (Jesus, Paul, Jerusalem, Pharisees). God, Lord, Spirit,
  Father and Son stand somewhere in lower case and stay, as elohim and adon stay in the Hebrew; Christ stays by one lower-case form.
- The content word is the phrase's last word that is not a function word (STOP); its key is that word in lower case. The model's place
  for the phrase is read at the token on which that word ends, never at a comma or at "of".
- Two English words are clearly different when their keys differ and do not share a stem: the first four letters once plain endings
  and the irregular forms in IRREGULAR are undone (son and sons, city and cities, said and say share one).
- A phrase that begins within its verse's first three words is marked early: it has no context to be read by, and no reading uses it.
- In a phrase with two or more content words (one row in twelve) the last may not be the rendering ("turn aside", "the generations
  to come"). `read --single` leaves such phrases out, as a check; compare then takes the names NAME_single.
- A row holds: v (verse), phrase and span (its characters in the BSB verse exactly as stored), word (the content word's span), key,
  words (how many content words the phrase has), lang, strong (the lexicon number as the table gives it), pos, early, at (words of the
  verse before the content word), prev (the word before it).

THE PLACES (`embed`, the card). Each verse alone after the model's own start token, as measure_base.embed feeds it. Kept as float16:
for every row of the table the hidden state at the token on which its content word ends, at each layer asked for; for every verse the
mean over its tokens and the state at its last token; and, for the base model, the rows of the input word table under each content
word (an adapter does not change them).

THE READINGS (`read`, CPU), at each layer, by cosine after subtracting the mean place of the unread verses' phrases. Pairs are drawn
from the table and the list of unread verses alone, so every model is read on the same pairs. No pair joins two verses that share a
run of eight words (parallel passages).
- Drawn together: two phrases in two unread verses that render one original word in clearly different English, against the first of
  them with a phrase of another original word of the same language and part of speech, also in clearly different English. At most 12
  pairs a word. The score is the chance that the same-word pair stands nearer; 0.5 is chance. The 95% interval resamples original
  words.
- Held apart: one English word in three unread verses: twice for one original word, once for another of the same language and part
  of speech. The score is the chance that the one-word pair stands nearer than the two-word pair. At most 12 a wording; the interval
  resamples English words.
- What needs no reading is printed beside both: the same score from the books alone (same book, same part of the canon), from how
  common the two English words are, from book and place in the verse together, and from the input word table with no context.
- The intervals, tried on made-up places that hold nothing: they held 0.5 in 192 and 191 of 200 trials (the two readings), and in
  188 and 188 when the phrases of one verse share their noise, which resampling words does not allow for. The interval on the
  difference of two models (`compare`) held zero in 183 to 190 of 200.
- The rehearsal (with --read): on the phrases of the read verses, a linear map is fitted by the loss the proposal describes for
  form A, and both readings are taken on the unread verses through it.
  - The states are first expressed in the read phrases' own spread, every direction scaled to one.
  - A step holds 8 original words of one language, two phrases each in two verses, every other phrase of those verses, and for each
    of the 16 one phrase with the same English word under another original word, where there is one.
  - Each of the 16 must stand nearer to its partner than to (1) phrases of other words and (2) its own English word under another
    original word, of its language and part of speech, by a margin, in cosine on step-centred states. The two parts weigh the same
    (--apart changes that). The margin is where half of the unfitted comparisons of the first part already hold.
  - The dose is the proposal's: the words that stand in 51 verses or more, about 600 steps.
  - Beside the true grouping, two fits of the same size. A false grouping: every (original word, English word) class stays whole
    and is dealt to another word of its language and part of speech. It is as consistent as the lexicon's, and false; it keeps one
    English word's two originals in two classes, so it is no control for "held apart". And labels dealt at random among the phrases
    of one language and part of speech, with nothing to learn.
  - A fit of this loss can move both readings whatever its labels: on one set of made-up places the false grouping rose half as
    far as the true one and the labels dealt at random a sixth as far. So the true grouping is read against the other two, not only
    against the row before fitting. "By what the fit was shown" splits "drawn together" three ways, because a map can learn the
    pairs of English words it was shown, as a list of synonyms, and carry nothing to the rest.
A negative means the model remains closer to the pattern of this world, never evidence against Scripture.

    python words_probe.py table                                              # -> runs/words/phrases.jsonl, phrases_stats.json (CPU, once)
    python words_probe.py embed --name base --limit 200                      # a smoke test on the card -> runs/words/base_limit200_*
    python words_probe.py embed --name base                                  # -> runs/words/base_{phrases,verses,wordtable}.npy, base_meta.json
    python words_probe.py embed --name pairs_links --adapter runs/train/ministral3_8b_pairs_links/adapter [--scale 0.5] [--verses FILE]
    python words_probe.py read --name base --unread FILE [--read FILE] [--single]   # -> runs/words/read_base.json (CPU)
    python words_probe.py compare --name pairs_links tongues_true            # each name's readings minus base's -> runs/words/compare.md
"""
import argparse
import json
import os
import random
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bible import CODES, verses
from measure_base import in_venv, keep_running, load_model, ordered, text_config
from probe import MODELS, budget_for, sha, write

HERE = Path(__file__).parent
OUT = HERE / "runs" / "words"
SOURCE = HERE / "data" / "orig" / "BSB_tables.jsonl"
KEY = "ministral3_8b"
SEED = "words 20261115"
DRAWS = 1000
PER_WORD = 12   # pairs a word, or a wording
EARLY = 3       # a phrase that begins within its verse's first three words has no context
RUN = 8         # two verses sharing a run of eight words are a parallel passage
DEVICE = 0
WATCH = {"H7307": "ruach", "H2451": "chokmah", "H998": "binah", "H6098": "etsah", "H1369": "gevurah", "H1847": "da'at",
         "H3374": "yirah", "H3519": "kavod"}
STOP = set("""the a an and but or nor for so yet then now also when that if because as while though although until since lest whether than of to
in on at by with from into upon over under before after against among between through about toward towards out up down off away back
forth around beside near within without across behind beyond along throughout unto aside together forward ahead else above below
during except despite amid inside outside onto like unlike concerning regarding according i me my mine we us our ours you your yours
he him his she her hers it its they them their theirs who whom whose which what this these those himself herself itself themselves
yourself yourselves myself ourselves one ones each every all any some no none both either neither other another such anyone anything
everyone everything someone something nothing whoever whatever whichever whenever wherever is am are was were be been being have has
had having do does did done will would shall should may might must can cannot could ought let not never there here o oh very too
more most even only just still again thus therefore indeed surely how why where s t m re ve ll d don doesn didn won wouldn couldn
shouldn isn aren wasn weren hasn haven hadn""".split())
HOMOGRAPHS = {"will", "might", "just", "near", "back"}  # function words that are also nouns or adjectives: "His might", "the will of God"
IRREGULAR = dict(p.split(":") for p in """said:say went:go gone:go goes:go going:go came:come became:become men:man women:woman
brethren:brother feet:foot teeth:tooth wives:wife knives:knife wolves:wolf thieves:thief calves:calf halves:half loaves:loaf
sheaves:sheaf lives:live life:live alive:live saw:see sight:see knew:know gave:give gift:give took:take made:make ate:eat spoke:speak
spoken:speak speech:speak told:tell brought:bring sent:send sat:sit stood:stand fell:fall rose:rise arose:arise wrote:write
found:find left:leave kept:keep slept:sleep wept:weep built:build sought:seek bought:buy sold:sell thought:think taught:teach
caught:catch fought:fight bore:bear borne:bear born:bear broke:break chose:choose chosen:choose choice:choose drank:drink
drunk:drink began:begin begun:begin sang:sing sung:sing song:sing swore:swear sworn:swear tore:tear torn:tear wore:wear worn:wear
drew:draw grew:grow threw:throw blew:blow flew:fly fled:flee fed:feed led:lead met:meet held:hold hid:hide lay:lie laid:lie
lain:lie lying:lie paid:pay lost:lose struck:strike stricken:strike smote:smite slew:slay slain:slay bound:bind felt:feel
knelt:kneel spent:spend bent:bend shone:shine shook:shake beheld:behold dying:die dead:die death:die better:good best:good
worse:bad worst:bad strength:strong strengthen:strong height:high length:long depth:deep breadth:broad youth:young pride:proud
salvation:save angry:anger hungry:hunger obedience:obey obedient:obey""".split())
WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")                   # an English word; "LORD’s" is LORD and s
SPACED = re.compile(r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*")       # a word as counted along the verse
SECTION = [0] * 5 + [1] * 12 + [2] * 5 + [3] * 17 + [4] * 4 + [5] + [6] * 13 + [7] * 8 + [8]  # law, history, poetry, prophets, gospels, Acts, Paul, general letters, Revelation


# ---------------------------------------------------------------- the table

def part_of_speech(lang, morph):
    """noun, verb, adjective, name or None, from the table's own tag. A Hebrew or Aramaic tag names prefixes and suffixes too
    ("Conj-w | N-fs", "Prep-l | V-Qal-Inf | 3ms"); the word's own part is the one that is a noun, a verb or an adjective."""
    m = morph or ""
    if lang == "grc":
        return {"N": "noun", "V": "verb", "Adj": "adjective"}.get(m.split("-")[0])
    for part in re.split(r"[|,]", m):
        head = re.match(r"\s*([A-Za-z]+)", part)
        head = head[1] if head else ""
        if head in ("N", "Noun"):
            return "name" if "proper" in part else "noun"
        if head in ("V", "Verb"):
            return "verb"
        if head == "Adj":
            return "adjective"
    return None


def locate(phrases, text):
    """The character span of each phrase in the verse text, None for a phrase the verse does not hold (a psalm's title leads the
    table's phrases and is not in the verse). Spaces aside, the phrases spell the verse in order; None if they do not."""
    flat = [i for i, c in enumerate(text) if not c.isspace()]
    spelled = "".join(text[i] for i in flat)
    bare = ["".join(p.split()) for p in phrases]
    for skip in range(len(bare)):
        if "".join(bare[skip:]) == spelled:
            spans, at = [None] * skip, 0
            for p in bare[skip:]:
                spans.append((flat[at], flat[at + len(p) - 1] + 1) if p else None)
                at += len(p)
            return spans
    return None


def content_word(text, start, end, pos):
    """(start, end, key, how many words of the phrase are not function words) for the phrase's last word that is not a function
    word, or None."""
    words = [(start + m.start(), start + m.end(), m[0].lower()) for m in WORD.finditer(text[start:end])]
    content = [w for w in words if w[2] not in STOP]
    if content:
        return (*content[-1], len(content))
    if words and pos in ("noun", "adjective") and words[-1][2] in HOMOGRAPHS:
        return (*words[-1], 1)
    return None


_stems = {}


def stem(key):
    """The first four letters of an English word once plain endings and the irregular forms are undone."""
    if key not in _stems:
        w = IRREGULAR.get(key, key)
        for ending, put, least in (("ies", "y", 2), ("ied", "y", 2), ("ing", "", 3), ("ed", "", 3), ("es", "", 3), ("s", "", 3)):
            if w.endswith(ending) and len(w) - len(ending) >= least:
                w = w[:-len(ending)] + put
                break
        if len(w) > 3 and w.endswith("e"):
            w = w[:-1]
        if len(w) > 3 and w.endswith("y"):
            w = w[:-1] + "i"
        _stems[key] = w[:4]
    return _stems[key]


def shares_stem(a, b):
    """Not clearly different English: the same key, or stems that agree as far as the shorter goes (son, sons; wise, wisdom)."""
    x, y = stem(a), stem(b)
    n = min(len(x), len(y))
    return a == b or x[:n] == y[:n]


def one_word_phrases(r, names=()):
    """For a verse of the publisher's table: {phrase index: (part of speech, lexicon number)} for the phrases that render exactly one
    noun, verb or adjective and no proper name; and why the others are left out."""
    per, why = defaultdict(list), Counter()
    for w in r["words"]:
        p = part_of_speech(r["lang"], w.get("morph"))
        if p and w.get("bsb") is not None:
            per[w["bsb"]].append(("name" if w.get("strong") in names else p, w.get("strong")))
    why["renders no noun, verb or adjective"] = len(r.get("bsb") or []) - len(per)
    out = {}
    for b, ws in per.items():
        if len(ws) > 1:
            why["renders two or more nouns, verbs, adjectives or names"] += 1
        elif ws[0][0] == "name":
            why["renders a proper name"] += 1
        elif not ws[0][1]:
            why["its word has no lexicon number"] += 1
        else:
            out[b] = ws[0]
    return out, why


def greek_names(tab, bsb):
    """Greek nouns and adjectives taken as proper names: every form capitalised, at least one of them not the first word of its verse
    (the Greek text has no punctuation, so a sentence's first word cannot be told otherwise), and the English content word capitalised
    nine times in ten."""
    total, capital, inside = Counter(), Counter(), Counter()
    for r in tab:
        if r["lang"] != "grc":
            continue
        for n, w in enumerate(r["words"]):
            s, form = w.get("strong"), w.get("form") or ""
            if s and part_of_speech("grc", w.get("morph")) in ("noun", "adjective"):
                up = form[:1].isupper()
                total[s] += 1
                capital[s] += up
                inside[s] += up and n > 0
    maybe = {s for s in total if capital[s] == total[s] and inside[s]}
    english, upper = Counter(), Counter()
    for r in tab:
        if r["lang"] != "grc":
            continue
        text = bsb[r["id"]]
        spans = locate(r.get("bsb") or [], text)
        for b, (pos, s) in one_word_phrases(r)[0].items():
            if s in maybe and spans and spans[b]:
                w = content_word(text, *spans[b], pos)
                if w:
                    english[s] += 1
                    upper[s] += text[w[0]].isupper()
    names = {s for s in maybe if upper[s] >= 0.9 * english[s]}
    return names, {"greek numbers taken as names": len(names), "their words": sum(total[s] for s in names),
                   "all forms capitalised but kept (English mostly lower case)": sorted(maybe - names)}


def table(args):
    """The table of phrases, from the publisher's word table and the BSB text alone. The same for every model."""
    OUT.mkdir(parents=True, exist_ok=True)
    bsb = verses("BSB")
    order = {v: n for n, v in enumerate(ordered()[0])}
    tab = sorted((json.loads(l) for l in open(SOURCE, encoding="utf-8")), key=lambda r: order[r["id"]])
    names, name_stats = greek_names(tab, bsb)
    rows, why, unmatched, prefixed = [], Counter(), [], 0
    for r in tab:
        text = bsb[r["id"]]
        spans = locate(r.get("bsb") or [], text)
        if spans is None:
            unmatched.append(r["id"])
            continue
        found, reasons = one_word_phrases(r, names)
        why.update(reasons)
        here = []
        for b, (pos, strong) in found.items():
            if spans[b] is None:
                why["not in the verse text (a psalm's title)"] += 1
                continue
            w = content_word(text, *spans[b], pos)
            if w is None:
                why["no English content word (is, all, before him)"] += 1
                continue
            before = [m[0].lower() for m in SPACED.finditer(text[:w[0]])]
            here.append({"v": r["id"], "phrase": text[spans[b][0]:spans[b][1]], "span": list(spans[b]), "word": [w[0], w[1]], "key": w[2],
                         "words": w[3], "lang": r["lang"], "strong": strong, "pos": pos, "early": len(SPACED.findall(text[:spans[b][0]])) < EARLY,
                         "at": len(before), "prev": before[-1] if before else ""})
            morph = next(x.get("morph") or "" for x in r["words"] if x.get("bsb") == b and part_of_speech(r["lang"], x.get("morph")))
            prefixed += r["lang"] != "grc" and not re.match(r"N|V|Adj", morph)
        rows += sorted(here, key=lambda x: x["span"])
    with open(OUT / "phrases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    in_verses = defaultdict(set)
    for row in rows:
        in_verses[row["strong"]].add(row["v"])
    keys_of = defaultdict(Counter)
    for row in rows:
        keys_of[row["strong"]][row["key"]] += 1
    words_of = defaultdict(set)
    for row in rows:
        words_of[row["key"], row["lang"], row["pos"]].add(row["strong"])
    rng = random.Random(SEED)
    differ = [(s, a, b) for s, c in sorted(keys_of.items()) for a in sorted(c) for b in sorted(c) if a < b]
    sample = lambda pairs: [" / ".join(p[1:]) for p in rng.sample(pairs, min(40, len(pairs)))]
    watch = {}
    for s, name in WATCH.items():
        mine = [row for row in rows if row["strong"] == s]
        watch[name] = {"number": s, "phrases": len(mine), "not early": sum(not row["early"] for row in mine), "verses": len(in_verses[s]),
                       "english words": len(keys_of[s]), "commonest": keys_of[s].most_common(8)}
    stats = {"verses": len(tab), "verses whose phrases spell the verse": len(tab) - len(unmatched), "not matched": unmatched,
             "rows": len(rows), "rows not early": sum(not row["early"] for row in rows), "left out": dict(why), "names": name_stats,
             "by language": dict(Counter(row["lang"] for row in rows)), "by part of speech": dict(Counter(row["pos"] for row in rows)),
             "hebrew or aramaic rows whose tag begins with a prefix (missed by a rule that reads the tag's first letters)": prefixed,
             "original words": len(in_verses), "in two verses or more": sum(len(v) >= 2 for v in in_verses.values()),
             "in 51 verses or more": sum(len(v) >= 51 for v in in_verses.values()),
             "share of rows under words in 51 verses or more": round(sum(len(in_verses[row["strong"]]) >= 51 for row in rows) / len(rows), 3),
             "english words": len({row["key"] for row in rows}),
             "an English word under two or more original words (counted for each language and part of speech)": sum(len(v) >= 2 for v in words_of.values()),
             "watched": watch,
             "for a second reader: 40 pairs of one word's English taken as clearly different": sample([p for p in differ if not shares_stem(p[1], p[2])]),
             "for a second reader: 40 taken as sharing a stem": sample([p for p in differ if shares_stem(p[1], p[2])]),
             "source_sha": sha(SOURCE), "bsb_sha": sha(HERE / "data" / "BSB_bible_segments.jsonl")}
    (OUT / "phrases_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    show = {k: v for k, v in stats.items() if k not in ("not matched", "names", "watched") and not k.startswith("for a second")}
    print(json.dumps(show, indent=1))
    print("names:", json.dumps({k: v for k, v in name_stats.items() if not isinstance(v, list)}), "| verses not matched:", len(unmatched))
    for name, w in watch.items():
        print(f"{name} ({w['number']}): {w['phrases']} phrases ({w['not early']} not early) in {w['verses']} verses, {w['english words']} English words; "
              + ", ".join(f"{k} {n}" for k, n in w["commonest"]))


def load_table():
    path = OUT / "phrases.jsonl"
    if not path.exists():
        raise SystemExit("run `words_probe.py table` first")
    return [json.loads(l) for l in open(path, encoding="utf-8")], sha(path)


# ---------------------------------------------------------------- the places

def tokens_of(offsets, special, start, end):
    """The tokens that hold any character of text[start:end], in order. The word ends on the last of them."""
    return [n for n, ((a, b), s) in enumerate(zip(offsets, special)) if not s and a < end and b > start]


def embed(args):
    """The places of every phrase and verse, for the base model or one adapter. Forward passes on the card."""
    import torch

    rows, table_sha = load_table()
    ids, texts = ordered()
    text_of = dict(zip(ids, texts))
    if args.verses:
        want = set(json.loads(Path(args.verses).read_text(encoding="utf-8")))
        if want - set(ids):
            raise SystemExit(f"{len(want - set(ids))} ids in {args.verses} are not BSB verses, e.g. {sorted(want - set(ids))[:3]}")
        ids = [v for v in ids if v in want]
    if args.limit:
        ids = ids[:args.limit]
    name = args.name + (f"_limit{args.limit}" if args.limit else "")
    layers = [int(x) for x in args.layers.split(",")]
    of_verse = defaultdict(list)
    for i, r in enumerate(rows):
        of_verse[r["v"]].append(i)
    kept = [i for v in ids for i in of_verse[v]]  # the table rows saved, in the table's order
    if not kept:
        raise SystemExit("no phrase of the table stands in these verses")
    if args.scale != 1 and not args.adapter:
        raise SystemExit("--scale needs --adapter")
    model_id, revision = MODELS[args.key]
    model, tok = load_model(model_id, revision, True, args.adapter, cpu_words=True)
    if args.scale != 1:
        from pairs_probe import scale_lora
        print("LoRA layers scaled by", args.scale, ":", scale_lora(model, args.scale), flush=True)
    torch.cuda.reset_peak_memory_stats()  # the peak below is the measurement's, not the loading's
    cfg = text_config(model)
    dim = cfg.hidden_size
    if max(layers) > cfg.num_hidden_layers:
        raise SystemExit(f"the model has {cfg.num_hidden_layers} layers")
    print(name, "loaded; layers", layers, "of", cfg.num_hidden_layers, "dim", dim, "| verses", len(ids), "phrases", len(kept), flush=True)

    place = {i: n for n, i in enumerate(kept)}
    enc, at_token, form, past = [], {}, {}, 0
    for v in ids:
        e = tok(text_of[v], add_special_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True)
        enc.append((list(e["input_ids"]), [n for n, s in enumerate(e["special_tokens_mask"]) if not s]))
        for i in of_verse[v]:
            ts = tokens_of(e["offset_mapping"], e["special_tokens_mask"], *rows[i]["word"])
            if not ts:
                raise SystemExit(f"no token holds the content word of row {i} ({v})")
            at_token[i] = ts[-1]
            past += e["offset_mapping"][ts[-1]][1] != rows[i]["word"][1]
            form[i] = tuple(e["input_ids"][t] for t in ts)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_meta.json").unlink(missing_ok=True)  # the index is written last: a run that stops early leaves places no one can read
    P = np.lib.format.open_memmap(OUT / f"{name}_phrases.npy", mode="w+", dtype=np.float16, shape=(len(kept), len(layers), dim))
    V = np.lib.format.open_memmap(OUT / f"{name}_verses.npy", mode="w+", dtype=np.float16, shape=(len(ids), len(layers), 2, dim))
    order = sorted(range(len(ids)), key=lambda n: len(enc[n][0]))
    t0, b, overflow = time.time(), 0, 0
    while b < len(order):
        n = 1
        while b + n < len(order) and (n + 1) * len(enc[order[b + n]][0]) <= args.budget:  # a padded-token budget, as probe.scorer packs
            n += 1
        batch = order[b:b + n]
        width = len(enc[batch[-1]][0])
        tokens = torch.full((n, width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((n, width), dtype=torch.long)
        real = torch.zeros((n, width, 1))
        last = torch.zeros(n, dtype=torch.long)
        for j, v in enumerate(batch):
            t, words = enc[v]
            tokens[j, :len(t)] = torch.tensor(t)
            mask[j, :len(t)] = 1
            real[j, words] = 1  # the start token is not part of the verse
            last[j] = words[-1]
        here = [(place[i], j, at_token[i]) for j, v in enumerate(batch) for i in of_verse[ids[v]]]
        with torch.inference_mode():
            hs = model(input_ids=tokens.to(DEVICE), attention_mask=mask.to(DEVICE), output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
            real, verse_of, token_of = real.to(DEVICE), torch.tensor([x[1] for x in here], dtype=torch.long), torch.tensor([x[2] for x in here], dtype=torch.long)
            for k, layer in enumerate(layers):
                h = hs[layer].float()
                mean = ((h * real).sum(1) / real.sum(1)).cpu().numpy().astype(np.float16)
                end = h[torch.arange(n), last].cpu().numpy().astype(np.float16)
                V[batch, k, 0], V[batch, k, 1] = mean, end
                overflow += int((~np.isfinite(mean)).sum() + (~np.isfinite(end)).sum())
                if here:
                    at = h[verse_of, token_of].cpu().numpy().astype(np.float16)
                    P[[x[0] for x in here], k] = at
                    overflow += int((~np.isfinite(at)).sum())
        del hs
        b += n
        if b // 4096 != (b - n) // 4096 or b == len(order):
            print(f"{b}/{len(ids)} verses, {b / (time.time() - t0):.0f}/s, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    P.flush()
    V.flush()
    meta = {"name": name, "key": args.key, "model": model_id, "revision": revision, "adapter": args.adapter, "scale": args.scale, "four_bit": True,
            "layers": layers, "dim": dim, "table_sha": table_sha, "verses": ids, "rows": kept, "tokens": [len(e[1]) for e in enc],
            "word ends inside a token": past, "values too large for float16": overflow, "budget": args.budget, "torch": torch.__version__,
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "minutes": round((time.time() - t0) / 60, 1)}
    if not args.adapter:  # the input word table under each content word: LoRA does not touch it, so base's serves every adapter
        forms = sorted(set(form.values()))
        weight = model.get_input_embeddings().weight.detach()
        np.save(OUT / f"{name}_wordtable.npy", np.stack([weight[torch.tensor(f, device=weight.device)].float().mean(0).cpu().numpy() for f in forms]).astype(np.float16))
        index = {f: n for n, f in enumerate(forms)}
        meta["word_form"] = [index[form[i]] for i in kept]
    write(OUT / f"{name}_meta.json", meta)
    print(f"{name}: {len(ids)} verses, {len(kept)} phrases, {meta['minutes']} min, peak GB {meta['peak_gb']}; word ends inside a token {past}; "
          f"values too large for float16 {overflow}", flush=True)


# ---------------------------------------------------------------- the readings

def runs_of(text):
    w = re.findall(r"[a-z']+", text.lower().replace("’", "'"))
    return {tuple(w[i:i + RUN]) for i in range(len(w) - RUN + 1)}


def twin_test(bsb):
    """True for two verses that share a run of eight words (a parallel passage, a refrain): no pair is drawn between them."""
    seen = {}

    def twin(a, b):
        for v in (a, b):
            if v not in seen:
                seen[v] = runs_of(bsb[v])
        return not seen[a].isdisjoint(seen[b])
    return twin


def draw(items, valid, rng, most=PER_WORD):
    """Up to `most` different pairs of items that pass `valid`, each in a random order (the first is the pair's anchor)."""
    if len(items) <= 60:
        pairs = [(i, j) for n, i in enumerate(items) for j in items[n + 1:] if valid(i, j)]
        pairs = rng.sample(pairs, min(most, len(pairs)))
    else:
        got = set()
        for _ in range(50 * most):
            i, j = rng.sample(items, 2)
            if valid(i, j):
                got.add((min(i, j), max(i, j)))
                if len(got) == most:
                    break
        pairs = sorted(got)
    return [p if rng.random() < 0.5 else p[::-1] for p in pairs]


def drawn_together(rows, pool, twin):
    """One original word in clearly different English, against another word of the same language and part of speech."""
    rng = random.Random(f"{SEED} drawn together")
    by_word, cell = defaultdict(list), defaultdict(list)
    for i in pool:
        by_word[rows[i]["strong"]].append(i)
        cell[rows[i]["lang"], rows[i]["pos"]].append(i)
    apart = lambda i, j: rows[i]["v"] != rows[j]["v"] and not shares_stem(rows[i]["key"], rows[j]["key"]) and not twin(rows[i]["v"], rows[j]["v"])
    alike = lambda i, j: rows[i]["lang"] == rows[j]["lang"] and rows[i]["pos"] == rows[j]["pos"]
    near, far, group = [], [], []
    for s in sorted(by_word):
        for i, j in draw(by_word[s], lambda i, j: alike(i, j) and apart(i, j), rng):
            others = cell[rows[i]["lang"], rows[i]["pos"]]
            for _ in range(200):
                c = rng.choice(others)
                if rows[c]["strong"] != s and apart(i, c):
                    near.append((i, j))
                    far.append((i, c))
                    group.append(s)
                    break
    return {"near": np.array(near).reshape(-1, 2), "far": np.array(far).reshape(-1, 2), "group": np.array(group)}


def held_apart(rows, pool, twin):
    """One English word: twice for one original word (near), once for another of the same language and part of speech (far)."""
    rng = random.Random(f"{SEED} held apart")
    by_key = defaultdict(list)
    for i in pool:
        by_key[rows[i]["key"]].append(i)
    apart = lambda i, j: rows[i]["v"] != rows[j]["v"] and not twin(rows[i]["v"], rows[j]["v"])
    near, far, group = [], [], []
    for k in sorted(by_key):
        cells = defaultdict(lambda: defaultdict(list))  # language and part of speech -> original word -> phrases
        for i in by_key[k]:
            cells[rows[i]["lang"], rows[i]["pos"]][rows[i]["strong"]].append(i)
        first = [i for words in cells.values() if len(words) > 1 for v in words.values() if len(v) > 1 for i in v]
        others, got = {}, set()
        for _ in range(20 * PER_WORD if first else 0):
            a = rng.choice(first)
            cell, s = (rows[a]["lang"], rows[a]["pos"]), rows[a]["strong"]
            if (cell, s) not in others:
                others[cell, s] = [i for w, v in cells[cell].items() if w != s for i in v]
            c, b = rng.choice(cells[cell][s]), rng.choice(others[cell, s])
            if c != a and apart(a, c) and apart(a, b):
                got.add((a, c, b))
                if len(got) == PER_WORD:
                    break
        for a, c, b in sorted(got):
            near.append((a, c))
            far.append((a, b))
            group.append(k)
    return {"near": np.array(near).reshape(-1, 2), "far": np.array(far).reshape(-1, 2), "group": np.array(group)}


def chance(near, far, group, tag, keep=None):
    """The chance that a near pair has the higher score than a far pair (ties count half), with a 95% interval from resampling the
    groups (original words, or English words) the pairs belong to. The same tag gives the same resamplings for every model."""
    if keep is not None:
        near, far, group = near[keep], far[keep], group[keep]
    if not len(near):
        return {"score": float("nan"), "low": float("nan"), "high": float("nan"), "pairs": 0, "draws": []}
    u, inv = np.unique(np.concatenate([near, far]), return_inverse=True)
    names, g = np.unique(group, return_inverse=True)

    def one(w):
        p = np.bincount(inv[:len(near)], w, len(u))
        n = np.bincount(inv[len(near):], w, len(u))
        return float((p * (np.cumsum(n) - n / 2)).sum() / (p.sum() * n.sum()))
    rng = np.random.default_rng(zlib.crc32(f"{SEED} {tag}".encode()))
    weights = [np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)).astype(float)[g] for _ in range(DRAWS)]
    draws = np.array([one(w) for w in weights])
    # beside it (added 21 Nov 2026, before any form B adapter was read): each near pair against its own far pair, which share an anchor
    won = (near > far) + 0.5 * (near == far)
    paired = np.array([float((won * w).sum() / w.sum()) for w in weights])
    return {"score": one(np.ones(len(near))), "low": float(np.quantile(draws, 0.025)), "high": float(np.quantile(draws, 0.975)),
            "pairs": int(len(near)), "draws": [round(float(x), 5) for x in draws],
            "paired": float(won.mean()), "paired_low": float(np.quantile(paired, 0.025)), "paired_high": float(np.quantile(paired, 0.975)),
            "paired_draws": [round(float(x), 5) for x in paired]}


def auc(near, far):
    """The chance that a near pair has the higher score than a far pair, ties counted half."""
    if not len(near) or not len(far):
        return float("nan")
    u, inv = np.unique(np.concatenate([near, far]), return_inverse=True)
    p, n = np.bincount(inv[:len(near)], minlength=len(u)), np.bincount(inv[len(near):], minlength=len(u))
    return float((p * (np.cumsum(n) - n / 2)).sum() / (p.sum() * n.sum()))


def crossfit(near, far, group):
    """The score a logistic fit of these features gives: fitted on half the groups, read on the other half, both ways, averaged."""
    names = sorted(set(group.tolist()))
    random.Random(f"{SEED} halves").shuffle(names)
    first = np.isin(group, names[:len(names) // 2])
    out = []
    for fit_on in (first, ~first):
        if fit_on.sum() < 20 or (~fit_on).sum() < 20:
            return float("nan")
        X = np.vstack([near[fit_on], far[fit_on]])
        y = np.concatenate([np.ones(fit_on.sum()), np.zeros(fit_on.sum())])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        X, w, b = (X - mu) / sd, np.zeros(X.shape[1]), 0.0
        for _ in range(400):
            p = 1 / (1 + np.exp(-(X @ w + b)))
            w -= 0.5 * (X.T @ (p - y) / len(y) + 1e-3 * w)
            b -= 0.5 * float(np.mean(p - y))
        out.append(auc(((near[~fit_on] - mu) / sd) @ w, ((far[~fit_on] - mu) / sd) @ w))
    return float(np.mean(out))


ORDER = {}


def no_reading(rows, pairs, vectors=None):
    """What needs no reading of the verse, on the same pairs: the books alone, how common the two English words are, book and place
    together, and the model's input word table (vectors: {row: its content word's rows of the table, averaged}, already centred)."""
    often = Counter(r["key"] for r in rows)
    book = lambda i: CODES.index(rows[i]["v"].split(".")[0])
    chapter_of = lambda i: rows[i]["v"].rsplit(".", 1)[0]
    if not ORDER:  # every BSB verse's place in the canon
        ORDER.update({v: n for n, v in enumerate(verses("BSB"))})

    def features(ps):
        out = []
        for i, j in ps:
            a, b = np.log(often[rows[i]["key"]]), np.log(often[rows[j]["key"]])
            wi, wj = rows[i]["words"], rows[j]["words"]
            out.append([book(i) == book(j), SECTION[book(i)] == SECTION[book(j)], abs(a - b), min(a, b), max(a, b),
                        rows[i]["prev"] == rows[j]["prev"], abs(rows[i]["at"] - rows[j]["at"]) / 10,
                        chapter_of(i) == chapter_of(j), np.log1p(abs(ORDER[rows[i]["v"]] - ORDER[rows[j]["v"]])),
                        rows[i]["phrase"].lower() == rows[j]["phrase"].lower(), min(wi, wj), max(wi, wj)])
        return np.array(out, dtype=float).reshape(-1, 12)
    near, far, group = features(pairs["near"]), features(pairs["far"]), pairs["group"]
    out = {"books alone": auc(near[:, 0] + near[:, 1], far[:, 0] + far[:, 1]),
           "how common the English words are": crossfit(near[:, 2:5], far[:, 2:5], group),
           "book and place in the verse": crossfit(near[:, [0, 1, 5, 6]], far[:, [0, 1, 5, 6]], group),
           # added 21 Nov 2026 after the review: the chapter, the distance in the canon, the same phrase wording, how many content words
           "book, chapter, place and the phrase's shape": crossfit(near[:, [0, 1, 5, 6, 7, 8, 9, 10, 11]], far[:, [0, 1, 5, 6, 7, 8, 9, 10, 11]], group),
           "the word table, no context": None}
    if vectors is not None:
        cos = lambda ps: np.array([float(vectors[i] @ vectors[j]) for i, j in ps])
        out["the word table, no context"] = auc(cos(pairs["near"]), cos(pairs["far"]))
    return out


def gather(P, k, where, chunk=4096):
    """Rows `where` of layer k of a saved array, as float16 in memory, read in the file's order."""
    where = np.asarray(where)
    order = np.argsort(where)
    out = np.empty((len(where), P.shape[2]), dtype=np.float16)
    for s in range(0, len(where), chunk):
        o = order[s:s + chunk]
        out[o] = P[where[o], k]
    return out


def mean_of(P, k, where, chunk=4096):
    where, total = np.sort(np.asarray(where)), np.zeros(P.shape[2])
    for s in range(0, len(where), chunk):
        total += P[where[s:s + chunk], k].astype(np.float64).sum(0)
    return (total / len(where)).astype(np.float32)


def cosines(X, at, pairs, centre, chunk=8192):
    """Cosine of each pair of rows of X (`at` maps a table row to its row of X), after subtracting `centre`."""
    out = np.empty(len(pairs))
    for s in range(0, len(pairs), chunk):
        ps = pairs[s:s + chunk]
        a, b = (X[[at[i] for i in ps[:, n]]].astype(np.float32) - centre for n in (0, 1))
        out[s:s + chunk] = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return out


def false_grouping(rows, mine, rng):
    """Every (original word, English word) class among `mine` stays whole and takes the label of another class's word, dealt within
    one language and part of speech: a grouping as consistent as the lexicon's, and false."""
    main = {}
    for s, ps in Counter((rows[i]["strong"], rows[i]["lang"], rows[i]["pos"]) for i in mine).most_common():
        main.setdefault(s[0], s[1:])
    label = {}
    for cell in sorted(set(main.values())):
        classes = sorted({(rows[i]["strong"], rows[i]["key"]) for i in mine if main[rows[i]["strong"]] == cell})
        dealt = [c[0] for c in classes]
        rng.shuffle(dealt)
        label.update(zip(classes, dealt))
    return {i: label[rows[i]["strong"], rows[i]["key"]] for i in mine}


def dealt_at_random(rows, mine, rng):
    """The original words dealt at random among the phrases `mine` of one language and part of speech: each word keeps as many
    phrases as it had, and which they are is chance. Nothing in it can be learned."""
    cells = defaultdict(list)
    for i in mine:
        cells[rows[i]["lang"], rows[i]["pos"]].append(i)
    label = {}
    for c in sorted(cells):
        words = [rows[i]["strong"] for i in cells[c]]
        rng.shuffle(words)
        label.update(zip(cells[c], words))
    return label


def plan(rows, label, focus, fit_rows, of_verse, same_english, rng, steps, words):
    """The steps of a fit. Each holds `words` label groups of one language: two phrases of each in two verses (in different English
    where it can), every other phrase of those verses, and for each of the two a phrase with the same English word under another
    label of its language and part of speech, where there is one. Returns [(rows of the step, [(phrase, its partner)])]."""
    groups = defaultdict(list)
    for i in fit_rows:
        if label[i] in focus:
            groups[label[i]].append(i)
    groups = {g: v for g, v in groups.items() if len({rows[i]["v"] for i in v}) >= 2}
    by_lang = defaultdict(list)
    for g in sorted(groups):
        by_lang[rows[groups[g][0]]["lang"]].append(g)
    each = steps * words / max(len(groups), 1)
    out = []
    for lang in sorted(by_lang):
        gs = by_lang[lang]
        visits = [g for _ in range(int(each)) for g in rng.sample(gs, len(gs))] + rng.sample(gs, round((each - int(each)) * len(gs)))
        for s in range(0, len(visits), words):
            used, pairs, extra = set(), [], set()
            for g in visits[s:s + words]:
                cand = [i for i in groups[g] if rows[i]["v"] not in used]
                for t in range(12):
                    if len({rows[i]["v"] for i in cand}) < 2:
                        break
                    a, p = rng.sample(cand, 2)
                    if rows[a]["v"] != rows[p]["v"] and (rows[a]["key"] != rows[p]["key"] or t >= 6):
                        pairs += [(a, p), (p, a)]
                        used |= {rows[a]["v"], rows[p]["v"]}
                        break
            for a, _ in pairs:
                others = same_english[rows[a]["key"], rows[a]["lang"], rows[a]["pos"]]
                for _ in range(10):
                    x = rng.choice(others)
                    if label[x] != label[a] and rows[x]["v"] not in used:
                        extra.add(x)
                        break
            if pairs:
                out.append((sorted({i for v in used for i in of_verse[v]} | extra), pairs))
    rng.shuffle(out)
    return out


def rehearse(rows, P, k, place, fit_rows, pool, sets, args):
    """The proposed loss for form A, fitted as a linear map on the places of the read verses' phrases, with a false grouping and
    labels dealt at random beside it; then both readings on the unread verses, before fitting and through each map."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # the fit is on the CPU; the card is never touched from here
    import torch

    in_verses = defaultdict(set)
    for r in rows:
        in_verses[r["strong"]].add(r["v"])
    of_verse, same_english = defaultdict(list), defaultdict(list)
    for i in fit_rows:
        of_verse[rows[i]["v"]].append(i)
        same_english[rows[i]["key"], rows[i]["lang"], rows[i]["pos"]].append(i)
    focus = {s for s, vs in in_verses.items() if len(vs) >= args.min_verses}  # the words the proposal's dose would train
    true = {i: rows[i]["strong"] for i in fit_rows}
    mine = [i for i in fit_rows if true[i] in focus]
    labels = {"true grouping": true,
              "false grouping": {**true, **false_grouping(rows, mine, random.Random(f"{SEED} false grouping"))},
              "labels dealt at random": {**true, **dealt_at_random(rows, mine, random.Random(f"{SEED} dealt at random"))}}
    plans = {name: plan(rows, label, focus, fit_rows, of_verse, same_english, random.Random(f"{SEED} plan"), args.steps, args.words)
             for name, label in labels.items()}
    if not all(plans.values()):
        return {"note": "no original word stands in two read verses often enough to fit on"}
    shown = defaultdict(set)
    for _, pairs in plans["true grouping"]:
        for a, _ in pairs:
            shown[rows[a]["strong"]].add(rows[a]["key"])

    # the read phrases' own spread: every direction scaled to one (--dim keeps only the largest); a direction with almost no spread
    # holds float16 rounding more than anything, so it is not blown up
    where = np.sort(np.array([place[i] for i in fit_rows]))
    total, square = np.zeros(P.shape[2]), np.zeros((P.shape[2], P.shape[2]))
    for s in range(0, len(where), 4096):
        x = P[where[s:s + 4096], k].astype(np.float32)
        total += x.sum(0, dtype=np.float64)
        square += x.T @ x
    mu = total / len(where)
    lam, vec = np.linalg.eigh(square / len(where) - np.outer(mu, mu))
    top = np.argsort(lam)[::-1][:args.dim or None]
    basis, mu = (vec[:, top] / np.sqrt(np.maximum(lam[top], 0) + 1e-3 * np.median(lam))).astype(np.float32), mu.astype(np.float32)

    need = sorted({i for steps in plans.values() for members, _ in steps for i in members} | {int(i) for s in sets.values() for kind in ("near", "far") for i in s[kind].ravel()})
    at = {i: n for n, i in enumerate(need)}
    in_file = np.array([place[i] for i in need])  # ascending, as the table is
    Y = np.empty((len(need), len(top)), dtype=np.float16)
    for s in range(0, len(need), 8192):
        Y[s:s + 8192] = (P[in_file[s:s + 8192], k].astype(np.float32) - mu) @ basis
    centre = (mean_of(P, k, [place[i] for i in pool]) - mu) @ basis
    code = lambda xs: {x: n for n, x in enumerate(sorted(set(xs)))}
    langs, parts, keys = code(r["lang"] for r in rows), code(r["pos"] for r in rows), code(r["key"] for r in rows)
    Yt = torch.from_numpy(Y)

    def tensors(label, steps):
        names = code(label.values())
        for members, pairs in steps:
            local = {i: n for n, i in enumerate(members)}
            t = lambda xs: torch.tensor(list(xs), dtype=torch.long)
            yield (t(at[i] for i in members), t(local[a] for a, _ in pairs), t(local[p] for _, p in pairs), t(names[label[i]] for i in members),
                   t(langs[rows[i]["lang"]] for i in members), t(parts[rows[i]["pos"]] for i in members), t(keys[rows[i]["key"]] for i in members))

    def compare_step(step, W):
        """For each of the step's phrase-partner pairs: how much nearer the partner stands than every phrase it is to stand apart from
        (cosine on step-centred states), and which of those are plain other words and which the same English word."""
        idx, a, p, lab, lang, part, key = step
        z = Yt[idx].float()
        z = z + z @ W
        z = torch.nn.functional.normalize(z - z.mean(0, keepdim=True), dim=1)
        sim = z @ z.T
        other = (lab[a][:, None] != lab[None, :]) & (lang[a][:, None] == lang[None, :]) & (part[a][:, None] == part[None, :])
        same = other & (key[a][:, None] == key[None, :])
        return sim[a, p][:, None] - sim[a], other & ~same, same

    zero = torch.zeros(len(top), len(top))
    start = [compare_step(s, zero) for s in tensors(true, plans["true grouping"][:200])]
    plain, same = torch.cat([g[m] for g, m, _ in start]), torch.cat([g[m] for g, _, m in start])
    margin = float(plain.median())
    share = lambda g: round(float((g > 0).float().mean()), 3) if len(g) else None
    out = {"dimensions": int(len(top)), "read phrases": len(fit_rows), "phrases of fitted words": len(mine), "margin": round(margin, 4),
           "steps asked": args.steps, "words a step": args.words, "words in at least this many verses": args.min_verses, "weight of the second part": args.apart,
           "comparisons that hold before fitting": {"other words": share(plain), "the same English word": share(same)}, "fits": {}}
    maps = {"before fitting": np.zeros((len(top), len(top)), dtype=np.float32)}
    for name, label in labels.items():
        torch.manual_seed(zlib.crc32(f"{SEED} fit".encode()))
        W = torch.zeros(len(top), len(top), requires_grad=True)
        opt = torch.optim.AdamW([W], lr=3e-4, weight_decay=1e-2)
        losses, crossings = [], 0
        for step in tensors(label, plans[name]):
            gap, plain, same = compare_step(step, W)
            hinge = torch.relu(margin - gap)
            loss = (hinge * plain).sum() / plain.sum().clamp(min=1) + args.apart * (hinge * same).sum() / same.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            crossings += int(same.sum())
        maps[name] = W.detach().numpy()
        out["fits"][name] = {"steps": len(losses), "label groups": len({label[a] for _, pairs in plans[name] for a, _ in pairs}),
                             "loss, first 50 steps": round(float(np.mean(losses[:50])), 4), "loss, last 50 steps": round(float(np.mean(losses[-50:])), 4),
                             "same-English comparisons": crossings}
    A = sets["drawn together"]
    both = np.array([s in focus and rows[i]["key"] in shown[s] and rows[j]["key"] in shown[s] for (i, j), s in zip(A["near"], A["group"])], dtype=bool)
    known = np.isin(A["group"], sorted(focus))
    strata = {"fitted words, both English words shown": both, "fitted words, an English word not shown": known & ~both, "words not fitted": ~known}
    in_pairs = sorted({int(i) for s in sets.values() for kind in ("near", "far") for i in s[kind].ravel()})
    spot = {i: n for n, i in enumerate(in_pairs)}
    for name, W in maps.items():
        through = np.eye(len(top), dtype=np.float32) + W
        Z = np.vstack([Y[[at[i] for i in in_pairs[n:n + 8192]]].astype(np.float32) @ through for n in range(0, len(in_pairs), 8192)])
        res = {}
        for reading, s in sets.items():
            near, far = (cosines(Z, spot, s[kind], centre @ through) for kind in ("near", "far"))
            res[reading] = chance(near, far, s["group"], reading)
            if reading == "drawn together":
                res["drawn together, by what the fit was shown"] = {label: {x: v for x, v in chance(near, far, s["group"], reading, keep).items() if not x.endswith("draws")}
                                                                    for label, keep in strata.items()}
        out[name] = res
    return out


ACROSS = "held apart, both pairs across books"


def read(args):
    """The readings for one saved set of places. CPU."""
    rows, table_sha = load_table()
    if not (OUT / f"{args.name}_meta.json").exists():
        raise SystemExit(f"no places are saved under {args.name}: run `words_probe.py embed --name {args.name}` on the card first")
    meta = json.loads((OUT / f"{args.name}_meta.json").read_text(encoding="utf-8"))
    if meta["table_sha"] != table_sha:
        raise SystemExit("the table has changed since these places were saved: embed again")
    P = np.load(OUT / f"{args.name}_phrases.npy", mmap_mode="r")
    place = {i: n for n, i in enumerate(meta["rows"])}
    bsb = verses("BSB")
    load = lambda p: set(json.loads(Path(p).read_text(encoding="utf-8")))
    unread, fit_on = load(args.unread), load(args.read) if args.read else set()
    if unread & fit_on:
        raise SystemExit(f"{len(unread & fit_on)} verses are in both lists")
    usable = lambda r: not r["early"] and (r["words"] == 1 or not args.single)
    pool = [i for i, r in enumerate(rows) if r["v"] in unread and usable(r)]
    fit_rows = [i for i, r in enumerate(rows) if r["v"] in fit_on and usable(r)]
    lacking = sum(i not in place for i in pool + fit_rows)
    if not pool:
        raise SystemExit(f"no phrase of the table stands in the verses of {args.unread}")
    if lacking:
        raise SystemExit(f"{lacking} phrases of the listed verses have no saved place under {args.name}: embed those verses first")
    twin = twin_test(bsb)
    sets = {"drawn together": drawn_together(rows, pool, twin), "held apart": held_apart(rows, pool, twin)}
    pairs_sha = f"{zlib.crc32(json.dumps([[s[k].tolist() for k in ('near', 'far')] for s in sets.values()]).encode()):08x}"
    vectors = None
    if (OUT / f"{args.name}_wordtable.npy").exists():
        T = np.load(OUT / f"{args.name}_wordtable.npy").astype(np.float32)
        of = lambda i: T[meta["word_form"][place[i]]]
        centre = np.mean([of(i) for i in pool], axis=0)
        vectors = {}
        for i in {int(i) for s in sets.values() for kind in ("near", "far") for i in s[kind].ravel()}:
            v = of(i) - centre
            vectors[i] = v / (np.linalg.norm(v) + 1e-12)
    counts = {"unread verses": len(unread), "their phrases, not early": len(pool), **{f"{name}: pairs": int(len(s["near"])) for name, s in sets.items()},
              "drawn together: original words": len(set(sets["drawn together"]["group"].tolist())),
              "held apart: English words": len(set(sets["held apart"]["group"].tolist()))}
    if fit_on:
        counts.update({"read verses": len(fit_on), "read phrases, not early": len(fit_rows)})
    saved_as = args.name + ("_single" if args.single else "") + (args.tag or "")  # --tag: a reading on another list of unread verses, kept apart
    out = {"name": saved_as, "model": meta["model"], "adapter": meta["adapter"], "scale": meta["scale"], "table_sha": table_sha, "pairs_sha": pairs_sha,
           "seed": SEED, "draws": DRAWS, "unread": args.unread, "read": args.read, "counts": counts,
           "no reading": {name: no_reading(rows, s, vectors) for name, s in sets.items()}, "layers": {}}
    need = sorted({int(i) for s in sets.values() for kind in ("near", "far") for i in s[kind].ravel()})
    # beside the fixed reading (added 21 Nov 2026, after the first adapters were read): the held-apart triples in which neither pair
    # stands within one book, so that what a book's wording gives is the same on both sides
    bk = lambda i: rows[int(i)]["v"].split(".")[0]
    ha = sets["held apart"]
    across = np.array([bk(a) != bk(c) and bk(a2) != bk(b) for (a, c), (a2, b) in zip(ha["near"], ha["far"])], dtype=bool)
    counts[ACROSS + ": pairs"] = int(across.sum())
    at = {i: n for n, i in enumerate(need)}
    for k, layer in enumerate(meta["layers"]):
        X = gather(P, k, [place[i] for i in need])
        centre = mean_of(P, k, [place[i] for i in pool])
        cos = {name: (cosines(X, at, s["near"], centre), cosines(X, at, s["far"], centre)) for name, s in sets.items()}
        res = {name: chance(*cos[name], s["group"], name) for name, s in sets.items()}
        res[ACROSS] = chance(*cos["held apart"], sets["held apart"]["group"], "held apart", across)
        if fit_rows:
            res["rehearsal"] = rehearse(rows, P, k, place, fit_rows, pool, sets, args)
        out["layers"][str(layer)] = res
        print(f"layer {layer} read", flush=True)
    write(OUT / f"read_{saved_as}.json", out)
    show(out)


def show(out):
    cell = lambda r: f"{r['score']:.3f} ({r['low']:.3f} to {r['high']:.3f})"
    names = ("drawn together", "held apart")
    print(f"\n{out['name']}: " + "; ".join(f"{k} {v:,}" for k, v in out["counts"].items()))
    print(f"\n{'':36s}{names[0]:>26s}{names[1]:>26s}")
    for cue in out["no reading"][names[0]]:
        vals = [out["no reading"][n][cue] for n in names]
        print(f"{cue:36s}" + "".join(f"{'not saved' if v is None else format(v, '.3f'):>26s}" for v in vals))
    for layer, res in out["layers"].items():
        print(f"{'layer ' + layer:36s}" + "".join(f"{cell(res[n]):>26s}" for n in names))
    for layer, res in out["layers"].items():
        if ACROSS in res:
            print(f"{'layer ' + layer + ', ' + ACROSS:60s}{cell(res[ACROSS])}, {res[ACROSS]['pairs']} pairs")
    for layer, res in out["layers"].items():
        r = res.get("rehearsal")
        if not r or "note" in r:
            continue
        print(f"\nrehearsal at layer {layer}: {r['dimensions']} dimensions, margin {r['margin']}, comparisons holding before fitting {r['comparisons that hold before fitting']}; "
              + "; ".join(f"{n}: loss {f['loss, first 50 steps']} to {f['loss, last 50 steps']} over {f['steps']} steps" for n, f in r["fits"].items()))
        for fit in ("before fitting", *r["fits"]):
            print(f"  {fit:34s}" + "".join(f"{cell(r[fit][n]):>26s}" for n in names))
            for label, s in r[fit]["drawn together, by what the fit was shown"].items():
                print(f"      {label:42s}{s['score']:.3f} ({s['low']:.3f} to {s['high']:.3f}), {s['pairs']} pairs")


def compare(args):
    """Each name's readings minus base's, on the same pairs and the same resamplings of words."""
    base = json.loads((OUT / f"read_{args.base}.json").read_text(encoding="utf-8"))
    md = [f"# The words in context: each model minus {args.base}", "",
          "The score is the chance that the same-word pair stands nearer (0.5 is chance). The interval is on the difference: both models are read on the same "
          "pairs and the same resamplings of original words (drawn together) or English words (held apart). An interval spanning zero reads: not shown.", "",
          "| model | layer | reading | model | " + args.base + " | difference (95% interval) |", "|---|---|---|---:|---:|---|"]
    for name in args.name:
        r = json.loads((OUT / f"read_{name}.json").read_text(encoding="utf-8"))
        if (r["pairs_sha"], r["table_sha"]) != (base["pairs_sha"], base["table_sha"]):
            raise SystemExit(f"{name} and {args.base} were read on different pairs: read both with the same table and --unread list")
        for layer, res in r["layers"].items():
            for reading in ("drawn together", "held apart", ACROSS) if layer in base["layers"] else ():
                if reading not in res or reading not in base["layers"][layer]:
                    continue
                x, y = res[reading], base["layers"][layer][reading]
                d = np.array(x["draws"]) - np.array(y["draws"])
                md.append(f"| {name} | {layer} | {reading} | {x['score']:.3f} | {y['score']:.3f} | {x['score'] - y['score']:+.3f} "
                          f"({np.quantile(d, 0.025):+.3f} to {np.quantile(d, 0.975):+.3f}) |")
                if "paired" in x and "paired" in y:
                    d = np.array(x["paired_draws"]) - np.array(y["paired_draws"])
                    md.append(f"| {name} | {layer} | {reading}, each pair against its own | {x['paired']:.3f} | {y['paired']:.3f} | {x['paired'] - y['paired']:+.3f} "
                              f"({np.quantile(d, 0.025):+.3f} to {np.quantile(d, 0.975):+.3f}) |")
    (OUT / ("compare.md" if args.base == "base" else f"compare_{args.base}.md")).write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))


def main():
    in_venv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table").set_defaults(fn=table)
    e = sub.add_parser("embed")
    e.add_argument("--name", required=True, help="base, or a name for the adapter's places")
    e.add_argument("--adapter", default=None, help="a LoRA adapter folder to apply over the base")
    e.add_argument("--scale", type=float, default=1.0, help="multiply every LoRA layer's scaling")
    e.add_argument("--verses", help="a JSON list of verse ids; default every verse")
    e.add_argument("--layers", default="8,17,25")
    e.add_argument("--limit", type=int, default=0, help="the first N verses only, saved under NAME_limitN")
    e.add_argument("--key", default=KEY, choices=sorted(MODELS))
    e.add_argument("--budget", type=int, default=budget_for(KEY), help="padded tokens a batch")
    e.set_defaults(fn=embed)
    r = sub.add_parser("read")
    r.add_argument("--name", required=True)
    r.add_argument("--unread", required=True, help="a JSON list of verse ids counted as unread")
    r.add_argument("--read", help="a JSON list of verse ids to fit the rehearsal on")
    r.add_argument("--single", action="store_true", help="use only phrases with one content word; written as read_NAME_single.json")
    r.add_argument("--tag", help="added to the saved name (read_NAME_TAG.json), for a reading on another --unread list")
    r.add_argument("--steps", type=int, default=600, help="rehearsal: steps (the proposal's dose)")
    r.add_argument("--words", type=int, default=8, help="rehearsal: original words a step")
    r.add_argument("--min-verses", type=int, default=51, help="rehearsal: fit the words that stand in at least this many verses")
    r.add_argument("--dim", type=int, default=0, help="rehearsal: keep only this many of the largest directions of the read phrases' spread (0: all)")
    r.add_argument("--apart", type=float, default=1.0, help="rehearsal: weight of the loss's second part (the same English word held apart) beside the first")
    r.set_defaults(fn=read)
    c = sub.add_parser("compare")
    c.add_argument("--name", required=True, nargs="+")
    c.add_argument("--base", default="base")
    c.set_defaults(fn=compare)
    args = p.parse_args()
    keep_running()
    args.fn(args)


if __name__ == "__main__":
    main()
