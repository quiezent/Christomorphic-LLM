"""Who is speaking: the third reading of step 1, by forward passes only, on Ministral.

Scripture says who speaks: "Then the LORD said to Moses," "But Jesus answered,". A scene here is two speeches by two different
speakers of one chapter, each with the clause that introduces it (its frame), standing within two verses of each other. Each
speech is scored after its own frame, after the other speaker's frame, and with no frame. Its margin is
log p(speech | its own frame) − log p(speech | the other speaker's frame), in nats: how much more the model expects the words
where the Bible puts them. Every piece is exact BSB text and only the joining differs. Nothing is trained, and nothing is sampled.

The twin: for each scene, two verses of the same chapter with no quotation mark and no speaker, each split at its first comma into
an opening and a rest, and swapped the same way. It shows how much of a margin comes merely from two pieces of text standing side
by side.

The table (python speakers_probe.py table), from the BSB text and the publisher's word table, with the repairs the skeptic asked for:
- a frame is narration that ends in a comma or a colon right before an opening mark. A clause inside its own speech
  (“…,” X said, “…”) is no frame;
- a speech runs from its opening mark to its closing mark. One that does not close within 3 verses is dropped, and one longer than
  60 words is cut at its last sentence end inside 60 words (failing that, at a comma);
- the speaker is the subject of the verb of saying. The English order decides: the nearest noun phrase before the verb that could
  be its subject, or a name in the verb's own phrase where the translators supplied it. Which English phrases are such nouns is
  read from the Hebrew or Greek (in the nominative; under no preposition or object marker). A frame is dropped where a pronoun
  speaks ("He said"), where it names the hearer ("the king was told"), and where the Hebrew or Greek order names an earlier noun
  than the English order does. No label is entered by hand;
- the two speeches stand within 2 verses of each other, and each speech stands in one scene only;
- the two speakers are different persons: two names (never one person's two names), or a group and one person, or God or Jesus and
  a human, or one frame names the other speaker. A noun such as "the king" or "the Philistine" beside a name is otherwise
  dropped, and so is "the Lord", "the angel of the LORD" or "a voice" beside God or Jesus. A refrain said twice is no scene;
- no speech with a capital You, Your, He, Him, His, Me, My, We, Us, Our where no sentence begins, or I AM (the BSB's capitals for
  God and for Jesus give the speaker or the hearer away);
- scenes whose speeches stand in near the same words in another chapter (parallel accounts) are tagged, and their chapters are
  kept together in the intervals and in the word counter's halves.

THE READING, fixed before any score (15 Nov 2026):
- The score is the capped margin: the smaller of 3 nats and the margin, a speech. It is read against base, by kind: God against a
  human, Jesus against another, two humans, other.
- RULE: if the TinyStories adapter stands at or above base on the capped margin (round 2's adapter, scored as pairs_tinystories:
  its mean over all scenes minus base's, zero or more), the reading is dropped. That adapter read no Scripture, so a score it
  does not lower is not a score of Scripture's attribution.
- It counts only beyond the twin (the scene figure minus the twin figure), and on the scenes a word counter does not settle: it
  picks the swapped joining, or cannot tell the two apart. The counter has no notion of a speaker: the verb of saying, a question
  mark, the length, the frame's end mark, a name recurring. It is fitted on half the chapters and applied to the other half, and
  the reverse.
- It counts only where the speech after its own frame is not less likely than in base. A margin also grows when both sides fall and
  the other frame's side falls more.
- Beside it, shown and not read: the plain mean margin and the share of speeches under 3 nats.
- Intervals are 95%, and everything from one chapter may move together (so may chapters that hold a parallel account). An interval
  spanning zero reads: not shown.
- It is a likelihood margin, offered for the Isaiah 11:3–5 test. It is not a measure of judgment, and it does not measure invented
  sayings. A negative means the model remains closer to the pattern of this world, never evidence against Scripture.

A caution, from the test on planted scores (15 Nov 2026): noise alone lowers a capped margin, because the cap takes from the high
side only. An arm planted with nothing but noise (2 nats a score) read −0.32 (−0.52 to −0.11) on all scenes. Its twins, planted
with smaller margins, read −0.05, so the scenes minus the twins still read −0.28. "Below base" may therefore mean no more than
that an arm's scores are noisier than base's; the card's own noise is printed when base is scored with --twice.

    python speakers_probe.py table                                   # CPU -> runs/speakers/scenes.json, the counts, 40 scenes to read
    python speakers_probe.py score --arm base --limit 8              # the card, a smoke run -> score_base_limit8.json (read --limit 8)
    python speakers_probe.py score --arm base --twice                # the card -> runs/speakers/score_base.json
    python speakers_probe.py score --adapter runs/train/ministral3_8b_pairs_tinystories/adapter --name pairs_tinystories
    python speakers_probe.py score --adapter PATH --name NAME --scale 0.5   # the adapter at half strength -> score_NAME_x0.5.json
    python speakers_probe.py read                                    # CPU -> runs/speakers/read.json, read.md
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from bible import CODES, verses
from measure_base import in_venv, keep_running, load_model
from pairs_probe import scale_lora
from probe import MODELS, budget_for, scorer, sha, write

HERE = Path(__file__).parent
OUT = HERE / "runs" / "speakers"
TABLES = HERE / "data" / "orig" / "BSB_tables.jsonl"
KEY = "ministral3_8b"
TINY = "pairs_tinystories"  # the arm the rule is applied to: round 2's TinyStories adapter, scored under this name
SEED = "speakers 20261115"
TIE = 1e-4   # the word counter scores the true and the swapped joining alike: it cannot tell
CAP = 3.0    # nats: the most one speech can add to the score
NEAR = 2     # verses from the end of one speech to the frame of the next
SPAN = 3     # verses a speech may run over
WORDS = 60   # a longer speech, or a twin's longer rest, is cut at its last sentence end inside this many words
SHOW = 40
SPREAD = 8.2         # nats an item: an adapter's margin minus base's, on round 2's saved scores (the skeptic's figure, 15 Nov)
SPREAD_CAPPED = 4.2  # the same for the capped margin (4.1 to 4.2 for the four Bible adapters, 4.8 for TinyStories)
TOGETHER = 0.1       # how far items of one chapter moved together there (0.05 to 0.12)
KINDS = {"god_human": "God against a human", "jesus_other": "Jesus against another", "humans": "two humans", "other": "other"}
O2, C2, O1, C1 = "“", "”", "‘", "’"
SKIP = (". . .", "-", "vvv")
SAY = {"H559", "H1696", "H6030", "H7121", "H6680", "H5046", "H7592", "H6817", "H2199", "H7650", "H5002", "H560", "H6032",
       "H4448", "H5001", "H7442", "H7891", "H6419", "H5012", "H3789", "H3384", "H5749",
       "G3004", "G2036", "G611", "G5346", "G2980", "G2896", "G1905", "G2065", "G994", "G2046", "G4483", "G1125",
       "G3140", "G2753", "G3870", "G4395", "G1263", "G5455", "G349", "G4377", "G2564", "G1781", "G3853", "G4336", "G5537"}
DIVINE = {"H3068", "H3069", "H430", "H136", "H410", "H7706", "H433", "H426", "G2316"}
ADVERSARY = {"H5175", "H7854", "G1228", "G4567", "G3985", "G3789"}
SAID = set("said says say saying answered answer replied asked asks ask told tells tell called call cried cry declared declares spoke spoken speaking speak "
           "speaks commanded shouted exclaimed instructed warned ordered begged pleaded prayed swore urged inquired continued added "
           "proclaimed announced responded questioned charged testified sang wrote insisted demanded protested explained reported "
           "informed rebuked blessed vowed promised cursed lamented taunted challenged appealed implored summoned directed telling "
           "calling crying asking replying answering commanding prophesied preached objected admitted confessed complained grumbled "
           "mocked scoffed greeted addressed exhorted advised".split())
PRON = set("he she they we i you it who whom which one some others another each all these those this many both someone anyone "
           "everyone none what whoever".split())
STOP = PRON | set("then but and so now again thus for yet still therefore finally meanwhile once at more in reply also moreover instead let "
                  "however later afterward immediately suddenly just when as after while because since if that the a an to o his her "
                  "their my your our its on by with from of there here not no yes behold look see surely truly before until though "
                  "although unless how why where next him them us me mine hers ours yours himself herself themselves myself yourself "
                  "had has have was were been would did kept began".split())
DET = set("the a an his her their its this that these those some".split())
SUBORD = set("that because for when while after before since as if although though until unless where whenever".split())
PREPS = set("to unto with by from of in at against concerning about into on upon through among toward over under within beside near around behind".split())
LEAD = re.compile(r"^(?:(?:Then|But|And|So|Now|Again|Thus|For|Yet|Still|Therefore|Finally|Meanwhile|Once again|At this|Once more|"
                  r"In reply|Also|Moreover|Furthermore|Instead|However|Nevertheless|Likewise|Later|Afterward|Immediately|Suddenly|Just then|"
                  r"Next|Soon|When|As|After|While|Because|Since|If|That|Who|Let)\b,?\s+)+", re.I)
OWNER = re.compile(r"^(?:(?:the|a|an|his|her|their|my|your|our|its|this|that|these|those|some|all|whole|certain|other|own|any|every|each|one|two|"
                   r"three)\s+)+", re.I)
TITLE = re.compile(r"^(?:(?:king|queen|prince|prophet|priest|governor|commander|apostle)\s+)+")
CAPS = re.compile(r"\b(You|Your|Yours|Yourself|Yourselves|He|Him|His|Himself|Me|My|Mine|Myself|We|Us|Our|Ours|Ourselves)\b")
FORMULA = re.compile(r"^[“‘]?\s*(This is what|Thus says|Hear the word|This is the word)")
MANY = set("people crowd crowds congregation assembly multitude army council company household family nation tribe city house "
           "community remnant group band camp sanhedrin church".split())
THING = re.compile(r"^(voice|word|creature|creatures|beast|beasts|donkey|cherub|cherubim|seraphim|star|wind|thunders?|sea|deep|earth|"
                   r"heavens|trees?|vine|bramble|altar|horn|eagle|death|abaddon|hades|sheol|wisdom|folly|glory|dragon|spirits?|demons?|"
                   r"legion|gabriel|michael|righteousness|scriptures?|law|faith|heart|soul|blood|stones?|land|mountains?|zion|jerusalem)\b")
ALIASES = [set(g.split()) for g in (
    "jacob israel", "abram abraham", "sarai sarah", "saul paul", "simon peter cephas", "esau edom", "gideon jerubbaal",
    "joshua hoshea", "daniel belteshazzar", "hananiah shadrach", "mishael meshach", "azariah abednego uzziah", "esther hadassah",
    "levi matthew", "thomas didymus", "jehoiachin jeconiah coniah", "solomon jedidiah", "naomi mara", "jethro reuel", "joram jehoram",
    "joash jehoash", "xerxes ahasuerus", "zedekiah mattaniah", "eliakim jehoiakim", "shallum jehoahaz", "tabitha dorcas",
    "bar-jesus elymas", "silas silvanus", "pharaoh egypt")]
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
        2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
VERBS = ("said answered replied asked told cried called commanded declared spoke shouted exclaimed instructed warned ordered begged "
         "pleaded prayed swore urged inquired continued added proclaimed announced").split()


def order(v):
    b, c, n = v.split(".")
    return CODES.index(b), int(c), int(n)


def t975(df):
    """The 97.5% point of Student's t: with few chapters, 1.96 would make the intervals too narrow."""
    return T975[df - 1] if df <= len(T975) else 1.96 + 2.4 / df


# ---------------------------------------------------------------------------------------------------------- the table of scenes
def closes_single(text, i):
    """Is the ’ at i a closing quotation mark, and not an apostrophe?"""
    prev = text[i - 1] if i else " "
    nxt = text[i + 1] if i + 1 < len(text) else " "
    if nxt.isalpha():
        return False
    if prev in ".,;:?!—" or nxt in (C2, O2) or prev == C2:
        return True
    return not (prev == "s" and nxt == " ")


def marks(text, depth):
    """One verse cut at its quotation marks, given the depth it starts at (0 is the narrator): [(depth, start, end, the depth after
    it)], the depth at the verse's end, and the marks that did not fit. A closing mark stays with its speech."""
    segs, start, odd, skip = [], 0, 0, 0
    lead = re.match("[“‘]+", text)
    if lead and depth >= 1:
        run = lead.group(0)
        if run[0] == O2 and all(c == (O2 if x % 2 == 0 else O1) for x, c in enumerate(run)):
            depth = skip = len(run)  # a verse inside a speech that reopens its marks states the depth itself: one mark a level
    for i, c in enumerate(text):
        if i < skip:
            continue
        nd = depth
        if c == O2:
            nd = 1 if depth == 0 else depth + 1 if depth in (2, 4) else depth
        elif c == C2:
            if depth == 0:
                odd += 1
            else:
                nd, odd = depth - 1, odd + (depth not in (1, 3))
        elif c == O1:
            if depth in (1, 3):
                nd = depth + 1
            elif depth == 0:
                nd, odd = 1, odd + 1
        elif c == C1 and depth in (2, 4) and closes_single(text, i):
            nd = depth - 1
        if nd > depth:  # opening: the lower segment ends before the mark
            if i > start:
                segs.append((depth, start, i, nd))
            start = i
        elif nd < depth:
            segs.append((depth, start, i + 1, nd))
            start = i + 1
        depth = nd
    if start < len(text):
        segs.append((depth, start, len(text), depth))
    return segs, depth, odd


def align(phrases, text):
    """Where each phrase of the word table stands in the verse text. The table spaces its dashes and marks differently and carries
    the Psalm titles, so characters are matched with the spaces left out. None where the two do not match."""
    squash = lambda s: re.sub(r"\s+", "", s)
    whole, want = squash(" ".join(phrases)), squash(text)
    if not whole.endswith(want):
        return None
    at = [i for i, c in enumerate(text) if not c.isspace()]
    out, k = [], len(want) - len(whole)
    for p in phrases:
        n = len(squash(p))
        out.append((at[k], at[k + n - 1] + 1) if k >= 0 and n else None)
        k += n
    return out


def word_roles(words, lang):
    """One record a word of the Hebrew or Greek, in its own order. role: v a verb; n a noun that could be a subject (Greek: in the
    nominative; Hebrew: under no preposition or object marker, and not the second noun of a construct pair); g such a second noun;
    o a noun that could not; p a pronoun standing alone; a a Greek nominative article."""
    out = []
    for i, w in enumerate(words):
        m, gloss = w.get("morph") or "", (w.get("gloss") or "-").strip()
        d = {"i": i, "j": w.get("bsb"), "strong": w.get("strong"), "gloss": gloss, "real": gloss not in SKIP and w.get("bsb") is not None, "role": "x",
             "inf": False, "finite": False, "plural": False, "proper": False, "bare": False, "construct": False, "and": False, "under": False, "nom": False,
             "first": False, "passive": False, "np": ""}
        if lang == "grc":
            f = m.split("-")
            d["and"] = w.get("strong") == "G2532"
            if f[0] == "V" and len(f) > 1:
                mood = f[1][1:2]
                d.update(role="v", inf=mood == "N", finite=mood in "ISOM", plural=len(f) > 2 and f[-1].endswith("P"), nom=mood == "P" and f[-1].startswith("N"))
            elif re.match(r"(N|Adj)-N", m):
                d.update(role="n", plural=m.endswith("P"), proper=(w.get("form") or "")[:1].isupper() and re.sub(r"\[.*?\]", "", gloss).strip()[:1].isupper())
            elif re.match(r"(PPro|DPro|RelPro|IPro|RefPro|RecPro)-N", m):
                d["role"] = "p"
            elif m.startswith("Art-N"):
                d["role"] = "a"
            elif re.match(r"(N|Adj)-", m):
                d["role"] = "o"
        else:
            parts = [p.strip() for p in re.split(r" \| |, ", m)]
            verb = next((p for p in parts if p.startswith("V-")), None)
            noun = next((p for p in parts if re.match(r"(N|Adj)-", p)), None)
            prep = any(p.startswith("Prep") or p == "DirObjM" for p in parts)
            suffix = any(re.fullmatch(r"[123][mfc][spd]e?2?", p) for p in parts)  # "his hand": the suffix is the second noun
            d["and"], d["under"] = parts[0].startswith("Conj-w"), prep
            if verb and "Prtcpl" in verb and "Art" in parts:
                noun, verb = verb, None  # a participle with the article is a noun or an adjective: "the murdered woman"
            if verb:
                d.update(role="v", inf="Inf" in verb, finite="Inf" not in verb and "Prtcpl" not in verb, plural=bool(re.search(r"[mfc]p[cd]?$", verb)),
                         first="ConsecImperf" in verb, passive="Hofal" in verb or "Pual" in verb)
            elif noun:
                d.update(role="n", plural=bool(re.search(r"-[mfc][pd][cd]?$", noun)), proper=noun.startswith("N-proper"), construct=bool(re.search(r"-[mfc][spd]c$", noun)) and not suffix)
            elif any(re.match(r"Pro-(?!r$)", p) for p in parts):
                d["role"] = "p"
            else:  # a preposition or the object marker standing alone, with no suffix: it governs the next word
                d["bare"] = prep and not suffix
                d["number"] = any(p.startswith("Number") for p in parts)
        out.append(d)
    for i, d in enumerate(out):
        prev = out[i - 1] if i else None
        if lang == "grc":
            if d["role"] == "v" and d["nom"] and prev and prev["role"] == "a":
                d["role"] = "n"  # a participle with its article is a noun: "the tempter"
        else:
            d["under"] = d["under"] or bool(prev and (prev["bare"] or (prev["construct"] and prev["under"])))
            # the phrase goes on: "to Jacob her son", "before his associates and the army of Samaria"
            on = prev["np"] if prev and prev["np"] and not prev["construct"] and (prev["np"] == "o" or (prev["np"] == "g" and not d["and"])) else ""
            if d["role"] == "n":
                d["role"] = d["np"] = "o" if d["under"] else on or ("g" if prev and prev["construct"] else "n")
            elif d.get("number"):
                d["np"] = "o" if d["under"] else on
    return out


def plain(s):
    """A subject as a name: no connective, no article."""
    s = LEAD.sub("", s.strip(" ,:;.“”‘’—")).strip()
    return re.sub(r"^(the|a|an)\s+|,\s+(however|then|also|therefore)$", "", s, flags=re.I).strip(" ,")


def in_phrase(tokens, k, gloss="", finite=True):
    """A speaker named inside a verb's own phrase, where the Hebrew or Greek has no word for him apart from the verb: a name
    ("But Jesus answered", "Elijah called"), a noun with its article before a verb of saying ("So the king said"), or what the
    gloss marks as supplied before the verb ("[The man] looked up"). k is the place of the verb among the phrase's words, None
    where it is not known. Returns the subject, "pronoun" ("He said"), "hearer" ("the king was told"), "unknown" (a noun with its
    article, where the verb's place is not known), or None."""
    before = tokens[:k] if k is not None else tokens
    if k is not None and k < len(tokens) and not tokens[k].endswith("ing") and any(t.lower() in ("was", "were", "is", "are", "been", "being", "be") for t in before):
        return "hearer"
    m = re.match(r"^(?:[A-Za-z]+,? )?\[([^\]]+)\]\s*\S", gloss) if k is None else None
    if m and any(t.lower() not in STOP for t in m.group(1).split()):
        return "pronoun" if m.group(1).split()[0].lower() in PRON else m.group(1)
    while before and before[0].lower() in STOP - DET - PRON:
        before = before[1:]
    if any(t.lower() in PRON for t in (before if k is not None else before[:2])):
        return "pronoun"
    led = bool(before) and before[0].lower() in DET
    rest = before[1:] if led else before
    caps = []
    for t in rest:
        if not (t[:1].isupper() and t.lower() not in STOP):
            break
        caps.append(t)
    if k is not None:
        while before and before[-1].lower() in STOP - DET:
            before = before[:-1]
        return " ".join(before) if (led and rest and len(before) > 1) or caps else None
    if caps and len(rest) > len(caps) and not caps[0].endswith(("ing", "ed")):
        return " ".join(caps)
    return "unknown" if led and rest and finite else None


def who(r, text, fa, fb):
    """The speaker that the framing clause text[fa:fb] names. The English order decides: from the verb of saying backwards, the
    first noun phrase that could be its subject, or a name inside a verb's own phrase. The Hebrew or Greek order is read beside
    it: where it names an earlier noun of the same chain ("Azariah, the chief priest, answered") the chain is the subject, and
    where it names an earlier noun elsewhere the frame is dropped. Returns (record, None), or (None, why not)."""
    spans = align(r["bsb"], text)
    if spans is None:
        return None, "the word table and the text differ"
    W, lang, ph = word_roles(r["words"], r["lang"]), r["lang"], r["bsb"]
    inside = [j for j, s in enumerate(spans) if s and s[0] >= fa and s[1] <= fb]
    F = [d for d in W if d["j"] in set(inside)]
    says = [d for d in F if d["strong"] in SAY and d["role"] == "v" and not d["inf"] and d["real"]]
    if not says:
        return None, "no verb of saying in the frame"
    v = max(says, key=lambda d: (d["j"], d["i"]))
    jv = v["j"]
    if v["passive"]:
        return None, "the frame names the hearer, not the speaker"
    real = lambda j: [d for d in F if d["j"] == j and d["real"]]
    role = {j: next((x for x in "vnpgao" if x in {d["role"] for d in real(j)}), "x") for j in inside}
    finite = {j: any(d["role"] == "v" and d["finite"] for d in real(j)) for j in inside}
    tokens = lambda j: re.findall(r"[A-Za-z’\-]+", ph[j])
    ends = lambda j: ph[j].rstrip(" ”’)").endswith((",", ";", ":", "—"))
    relative = lambda j: any(t.lower() in ("who", "whom", "whose", "which") for t in tokens(j))
    opener = lambda j: (tokens(j) or [""])[0].lower() in SUBORD  # "that Jeremiah had been telling", "because the Ephraimites had said"
    later = [j for j in inside if j > jv and any(t.lower() in SAID - {"saying"} for t in tokens(j))]
    if later and any(role[x] in "npg" for x in inside if jv < x <= later[-1]):
        return None, "another verb of saying follows, with a subject of its own"  # "…did not answer, for Hezekiah had commanded,"
    # noun phrases in the English order: each chain of them, cut at its commas, and whether its first part is an object or stands
    # in a relative clause (reached from a verb, or from "whom", with no comma and no "that" or "because" between). Nothing in
    # brackets is a subject, and nothing that opens with a preposition.
    runs, cur, depth, bracketed = {}, [], 0, set()
    for j in inside:
        if depth or "(" in ph[j]:
            bracketed.add(j)
        depth += ph[j].count("(") - ph[j].count(")")
    for j in [j for j in inside if j < jv] + [None]:
        if j is not None and j not in bracketed and (role[j] in "nga" or (cur and role[j] == "x" and ph[j].strip(" ,").lower() == "and")):
            cur.append(j)
            continue
        while cur and role[cur[-1]] == "x":
            cur.pop()
        if any(role[x] == "n" for x in cur):
            b4 = cur[0] - 1
            while b4 in role and not ends(b4) and not (role[b4] == "v" and finite[b4]) and not relative(b4) and not (opener(b4) and role[b4] != "v"):
                b4 -= 1
            clause = b4 not in role or ends(b4) or (opener(b4) and role[b4] != "v" and not relative(b4)) or opener(cur[0])
            parts = [[]]
            for x in cur:
                parts[-1].append(x)
                if ends(x) and x != cur[-1]:
                    parts.append([])
            parts = [p for p in parts if any(role[x] == "n" for x in p) and (tokens(p[0]) or [""])[0].lower() not in PREPS]  # not "to Moses"
            if parts:
                runs[cur[-1]] = (parts, not clause or relative(cur[0]))
        cur = []
    # the English order, from the verb of saying backwards: the first phrase that holds a subject
    found, j = None, jv
    while j >= inside[0] and found is None:
        if j < jv and j in role and role[j] != "v" and any(t.lower() in ("he", "she", "they", "we", "i") for t in tokens(j)):
            return None, "a pronoun speaks"  # "Balak’s anger burned against Balaam, and he struck his hands together and said"
        if j < jv and j in role and role[j] == "x" and (tokens(j) or [""])[0].lower() in SUBORD - {"for", "after", "before", "since", "as", "until"}:
            return None, "no speaker named"  # "…struck them down because the Ephraimites had said,": the table marks no subject in that clause
        if j in runs:
            parts, is_object = runs[j]
            if len(parts) > 1 or not is_object:  # after a comma a new clause may begin: the last part can be the subject
                found = ("run", parts[-1], parts, is_object)
            j = parts[0][0] - 1
            continue
        if j in role and role[j] == "v" and j not in bracketed:
            vt = tokens(j)
            got = in_phrase(vt, next((x for x, t in enumerate(vt) if t.lower() in SAID), len(vt)) if j == jv else None,
                            " ".join(d["gloss"] for d in real(j) if d["role"] == "v"), finite[j])
            if got in ("pronoun", "unknown", "hearer"):
                return None, {"pronoun": "a pronoun speaks", "unknown": "no speaker named", "hearer": "the frame names the hearer, not the speaker"}[got]
            if got:
                found = ("phrase", [j], got)
        elif j in role and role[j] == "p":  # "But Jesus took her by the hand", where the Greek has "he": the English names him
            got = in_phrase(tokens(j), len(tokens(j)))
            if not got or got in ("pronoun", "unknown", "hearer"):
                return None, "a pronoun speaks"
            found = ("phrase", [j], got)
        j -= 1
    if found is None:
        return None, "no speaker named"
    where = found[1]
    heads = [d for d in F if d["j"] in where and d["role"] == "n"]
    # the Hebrew or Greek order, for a subject that has its own word there
    if found[0] == "run":
        lo, hi = min(d["i"] for d in F), max(d["i"] for d in F)
        seq = [d for d in W if lo <= d["i"] <= hi and (d["j"] is None or d["j"] in role)]
        iv, head = next(n for n, d in enumerate(seq) if d["i"] == v["i"]), None
        stops = lambda d: d["role"] == "v" and not (d["inf"] and d["strong"] in SAY)
        noun = lambda d: d["role"] == "n" and d["j"] is not None
        if lang == "grc":
            nouns = [n for n, d in enumerate(seq) if noun(d)]
            clear = [n for n in nouns if not any(seq[x]["role"] == "v" and seq[x]["finite"] for x in range(min(n, iv) + 1, max(n, iv)))]
            pick = min(clear, key=lambda n: (abs(n - iv), n)) if clear else max((n for n in nouns if n < iv), default=None)
            head = seq[pick] if pick is not None else None
        else:
            for d in seq[iv + 1:]:  # the subject after its verb
                if stops(d) or d["role"] == "p":
                    break
                if noun(d):
                    head = d
                    break
            if head is None and not v["first"]:  # or right before it, where the verb does not open its clause
                for d in reversed(seq[:iv]):
                    if noun(d):
                        head = d
                    if d["role"] != "g" and not (d["role"] == "x" and not d["bare"]):
                        break
            if head is None:  # or carried from an earlier verb of the clause: the first noun after the nearest one that has any
                bounds = [-1] + [n for n in range(iv) if stops(seq[n])] + [iv]
                for a, b in reversed(list(zip(bounds, bounds[1:]))):
                    part = seq[a + 1:b]
                    if any(d["role"] == "p" for d in part):
                        break
                    head = next((d for d in part if noun(d)), None)
                    if head:
                        break
        if head is not None and head["j"] < where[0]:
            # the Hebrew or Greek names an earlier noun. Within the same chain it is the same subject, named first and described
            # after ("Azariah, the chief priest, answered") or a list; anywhere else the two readings differ.
            parts, is_object = found[2], found[3]
            at = next((n for n, p in enumerate(parts) if head["j"] in p), None)
            if at is None or (at == 0 and is_object):
                return None, "the two readings differ"
            where = [x for p in parts[at:] for x in p]
            heads = [d for d in F if d["j"] in where and d["role"] == "n"]
    subject = found[2] if found[0] == "phrase" else text[spans[where[0]][0]:spans[where[-1]][1]]
    name = plain(subject)
    if not name or name.split()[0].lower() in PRON:
        return None, "a pronoun speaks"
    if heads and name.lower() in STOP:  # the table hangs the noun on a connective: take the noun's own gloss
        name = plain(re.sub(r"[\[\]]", "", heads[0]["gloss"]))
        if not name or name.split()[0].lower() in PRON:
            return None, "a pronoun speaks"
    proper = [d for d in heads if d["proper"]]
    first = heads[0]["strong"] if heads else None
    front = name.split(" of ")[0]
    names, several = False, False
    if first in ADVERSARY or re.search(r"\b(serpent|Satan|devil|tempter)\b", front):
        cls, key = "adversary", "ADVERSARY"
    elif re.search(r"\bJesus\b", front):
        cls, key = "Jesus", "JESUS"
    elif first in DIVINE or re.search(r"\b(LORD|GOD|God|Almighty|Most High|Holy One)\b", front):
        cls, key = "God", "GOD"
    else:
        clean = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z\- ]", "", t.lower().replace("’s", ""))).strip()
        caps = lambda t: [w for w in re.findall(r"[A-Za-z\-]+", t) if w[:1].isupper() and w.lower() not in STOP]
        names = [c[-1] for c in (caps(d["gloss"]) for d in proper) if c] or (caps(name) if found[0] == "phrase" else [])
        if len(caps(name)) == 1 and re.match(r"(?i:the|a|an|this|that)\s+[A-Z]", LEAD.sub("", subject.strip(" ,:;“”‘’")).strip()):
            names = []  # "the Philistine", "a Cushite": what he is, not his name. Goliath is the Philistine.
        people = [x for x in re.split(r",?\s+and\s+|,\s+", name) if x] if re.search(r"\band\b", name) else [name]
        if len(people) > 1:  # "Pharaoh and his officials", "Peter and John": several speak, each by his name where he has one
            key = " + ".join((caps(x) or [""])[0].lower() or clean(OWNER.sub("", plain(x).lower())) for x in people)
        else:
            key = TITLE.sub("", clean(" ".join(names))) if names else clean(OWNER.sub("", name.lower()))
        key = key or name.lower()
        noun = OWNER.sub("", front.lower())
        cls = ("Lord" if first == "G2962" or name == "Lord" else "Spirit" if re.search(r"\bSpirit\b", front) else
               "angel" if re.search(r"\bangels?\b", noun) else "thing" if THING.match(noun) or THING.match(key) else "human")
        # "the men of Judah", "the people", "his brothers": the English noun says that several speak (a name in -s says nothing)
        word = (OWNER.sub("", front).split() or [front])[0]
        several = len(people) > 1 or word.lower() in MANY or re.fullmatch(r"men|women|children|brothers|[a-z\-]+[^s]s", word) is not None
    one = cls in ("God", "Jesus", "adversary", "Lord", "Spirit")
    plural = not one and (v["plural"] or bool(heads and heads[0]["plural"] and first not in ("H113", "H1167")) or several)  # "his master" is one man
    return {"speaker": name, "subject": subject, "key": key, "class": cls, "plural": bool(plural), "named": bool(one or names),
            "supplied": found[0] == "phrase" and not heads}, None


def capital_cue(speech):
    """The BSB's capital for God and for Jesus where no sentence begins, or I AM: it gives the speaker or the hearer away."""
    if re.search(r"\bI AM\b", speech):
        return "I AM"
    for m in CAPS.finditer(speech):
        before = speech[:m.start()].rstrip(" “‘(").rstrip("”’")
        if before and before[-1] not in ".!?:":
            return m.group(1)
    return None


def capped_text(text):
    """A text of more than WORDS words, cut at its last sentence end inside them (after 20 words or more), or else at its last comma
    in their second half, or else at WORDS words. Returns (text, whole)."""
    words = text.split(" ")
    if len(words) <= WORDS:
        return text, True
    ends = ([i for i in range(20, WORDS + 1) if re.search(r"[.!?][”’]*$", words[i - 1])]
            or [i for i in range(WORDS // 2, WORDS + 1) if re.search(r"[,;:—][”’]*$", words[i - 1])])
    return " ".join(words[:ends[-1] if ends else WORDS]), False


def hears(x, y):
    """Does x's frame name y outside its own subject (as the one spoken to)?"""
    rest = x["frame"].replace(x["subject"], " ", 1) if x["subject"] in x["frame"] else x["frame"]
    if y["class"] == "God":
        return bool(re.search(r"\b(LORD|GOD|God)\b", rest))
    if y["class"] == "Jesus":
        return bool(re.search(r"\bJesus\b", rest))
    if y["class"] == "adversary":
        return bool(re.search(r"\b(serpent|Satan|devil|tempter)\b", rest))
    words = [w for w in y["key"].replace(" + ", " ").split() if w not in STOP][:1 if not y["named"] else None]
    return any(re.search(r"\b%s\b" % re.escape(w), rest, re.I) for w in words)


def apart(a, b):
    """The ground on which the two speakers are two persons, or None where the frames do not show it."""
    ta, tb = (set(x["key"].replace(" + ", " ").split()) - STOP for x in (a, b))
    if a["key"] == b["key"] or not ta or not tb or ta <= tb or tb <= ta or any(ta & g and tb & g for g in ALIASES):
        return None  # one speaker, or one name inside the other, or one person's two names
    high, title = ("God", "Jesus"), ("Lord", "angel", "Spirit", "thing")
    ca, cb = a["class"], b["class"]
    if (ca in high and cb in title) or (cb in high and ca in title) or (ca in title and cb in title):
        return None  # "the Lord", "the angel of the LORD", "a voice": perhaps God or Jesus under another name
    if ca in high and cb in high:
        return "God and Jesus"
    if a["plural"] != b["plural"]:
        return "a group and one person"
    if not a["plural"] and a["named"] and b["named"]:
        return "two names"
    if not a["plural"] and (ca in high or cb in high) and "human" in (ca, cb):
        return "God or Jesus and a human"
    return "one frame names the other speaker" if hears(a, b) or hears(b, a) else None


def kind_of(a, b):
    c = (a["class"], b["class"])
    if "God" in c and "human" in c:
        return "god_human"
    if "Jesus" in c and "God" not in c:
        return "jesus_other"
    return "humans" if c == ("human", "human") else "other"


def alike(x, y):
    """Near the same words in the same order (a parallel account, or a refrain): the same words, or 0.6 of the two word sequences
    where both have 6 words or more."""
    a, b = (re.findall(r"[a-z]+", s.lower().replace("’", "")) for s in (x, y))
    if min(len(a), len(b)) < 4:
        return False
    return a == b or (min(len(a), len(b)) >= 6 and SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.6)


def feats(frame, speech):
    """What a word counter sees of a frame joined to a speech, with no notion of who speaks (the skeptic's surface_baseline3):
    the verb of saying with a question mark, with the speech's first word and with its length; the frame's end mark with the length;
    a name of the frame recurring in the speech."""
    out = Counter()
    sw = [w.lower() for w in re.findall(r"[A-Za-z']+", speech.replace("’", "'"))]
    q, ln = "?" in speech, min(len(sw) // 8, 4)
    for v in [v for v in VERBS if re.search(r"\b" + v + r"\b", frame)] or ["noverb"]:
        out[f"V={v}|q={q}"] += 1
        out[f"V={v}|first={sw[0] if sw else ''}"] += 1
        out[f"V={v}|len={ln}"] += 1
    out[f"end={frame.rstrip()[-1:]}|len={ln}"] += 1
    names = lambda s: ({m.group(1) for m in re.finditer(r"(?<![.!?:;—] )(?<!^)\b([A-Z][a-z]+)\b", re.sub(r"[“”‘’]", "", s))}
                       | set(re.findall(r"\b(LORD|GOD)\b", s)))
    fn, sn = names(frame), names(speech)
    to = set(re.findall(r"\bto (?:the |his |her |their )?([A-Z][A-Za-z]+)", frame))
    out["echo_any"] += len(fn & sn)
    out["echo_addressee"] += len(to & sn)
    out["echo_subject"] += len((fn - to) & sn)
    out["bias"] = 1
    return out


def word_counter(scenes, fold):
    """Fit the counter on the scenes of one half of the chapters and apply it to the other half, and the reverse. For each scene:
    does it score the true joining (each speech after its own frame) above the swapped one? Returns [own − swapped]."""
    X = [feats(f["frame"], s["speech"]) for sc in scenes for f, s in ((sc["a"], sc["a"]), (sc["b"], sc["b"]), (sc["b"], sc["a"]), (sc["a"], sc["b"]))]
    edge = [0.0] * len(scenes)
    for k in (0, 1):
        tr, te = ([n for n, sc in enumerate(scenes) if (fold[sc["cluster"]] == k) == other] for other in (False, True))
        cnt = Counter(f for n in tr for x in X[4 * n:4 * n + 4] for f in x)
        vocab = {f: i for i, f in enumerate(f for f, c in cnt.most_common() if c >= 3)}
        mat = lambda ns: np.array([[x.get(f, 0) for f in vocab] for n in ns for x in X[4 * n:4 * n + 4]], dtype=np.float32).reshape(4 * len(ns), len(vocab))
        M, y, w = mat(tr), np.array([1, 1, 0, 0] * len(tr), dtype=np.float32), np.zeros(len(vocab), dtype=np.float32)
        for _ in range(300):  # the skeptic's fit: plain gradient steps on a logistic loss with an L2 pull
            p = 1 / (1 + np.exp(-(M @ w)))
            w -= np.float32(0.5) * (M.T @ (p - y) / np.float32(len(y)) + np.float32(5.0) * w / np.float32(len(y)))
        s = mat(te) @ w
        for x, n in enumerate(te):
            edge[n] = float(s[4 * x] + s[4 * x + 1] - s[4 * x + 2] - s[4 * x + 3])
    return edge


def halves(scenes, rng):
    clusters = sorted({sc["cluster"] for sc in scenes})
    rng.shuffle(clusters)
    return {c: i % 2 for i, c in enumerate(clusters)}


def resolves(sizes, spread, together):
    """The half-width of a 95% interval on a mean over clusters of these sizes, at this spread an item, when items of one cluster
    move together by this correlation."""
    n, g = sum(sizes), len(sizes)
    return t975(g - 1) * spread * math.sqrt(g / (g - 1) * sum(m + m * (m - 1) * together for m in sizes)) / n if g > 1 else float("nan")


def items_of(scenes):
    """Every text to score, as the scorer takes it: (context, continuation, separator). The space before a speech goes with the
    speech, since the tokenizer joins it to the opening mark: so the loss starts at that mark and falls on the speech only.
    Returns the texts and, for each scene and each twin, where its six scores stand."""
    texts, where = [], {"scenes": [], "twins": []}

    def add(ctx, tgt):
        texts.append((None, tgt) if ctx is None else (ctx, " " + tgt, ""))
        return len(texts) - 1
    for sc in scenes:
        for part, pair, c, t in (("scenes", (sc["a"], sc["b"]), "frame", "speech"), ("twins", sc["twin"] and (sc["twin"]["a"], sc["twin"]["b"]), "opening", "rest")):
            if not pair:
                where[part].append(None)
                continue
            x, y = pair
            where[part].append({"own": [add(x[c], x[t]), add(y[c], y[t])], "other": [add(y[c], x[t]), add(x[c], y[t])], "alone": [add(None, x[t]), add(None, y[t])]})
    return texts, where


def falls_on(tok, texts):
    """The scorer's own rule (probe.scorer), replayed on the tokenizer alone: on which characters does the loss fall? Returns the
    texts where it is not exactly the continuation, and for each text (tokens scored, tokens in all)."""
    bad, counts = [], []
    for n, item in enumerate(texts):
        ctx, tgt = item[:2]
        sep = item[2] if len(item) > 2 else "\n"
        full = tgt if ctx is None else ctx + sep + tgt
        start = 0 if ctx is None else len(ctx) + len(sep)
        e = tok(full, add_special_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True)
        offs, sp = list(e["offset_mapping"]), list(e["special_tokens_mask"])
        keep = [i for i, ((s, _), p) in enumerate(zip(offs, sp)) if not p and s >= start and i > 0]
        if not keep or offs[keep[0]][0] != start or offs[keep[-1]][1] != len(full) or keep != list(range(keep[0], len(offs))):
            bad.append(n)
        counts.append((len(keep), len(offs)))
    return bad, counts


def show(n, sc):
    lines = [f"[{n}] {sc['chapter']} | {KINDS[sc['kind']]} | two persons because: {sc['apart']}" + (f" | parallel: {', '.join(sc['parallel'])}" if sc["parallel"] else "")
             + f" | word counter: {'right' if sc['counter']['right'] else 'cannot tell' if sc['counter']['tie'] else 'wrong'}"]
    for s in ("a", "b"):
        x = sc[s]
        lines += [f"   {s} {x['verse']}  speaker: {x['speaker']} ({x['class']}{', a group' if x['plural'] else ''}{', name supplied by the translators' if x['supplied'] else ''})"
                  + ("" if x["whole"] else " | cut at 60 words"), f"      frame : {x['frame']}", f"      speech: {x['speech']}"]
    if sc["twin"]:
        for s in ("a", "b"):
            x = sc["twin"][s]
            lines.append(f"   twin {s} {x['verse']}: {x['opening']} || {x['rest']}")
    else:
        lines.append("   no twin: the chapter has no two verses left without a quotation mark")
    return lines


def table(args):
    """The scenes and their twins, from the BSB text and the publisher's word table alone. The same for every arm."""
    OUT.mkdir(parents=True, exist_ok=True)
    bsb = verses("BSB")
    rows = sorted((r for r in (json.loads(l) for l in open(TABLES, encoding="utf-8")) if r["id"] in bsb), key=lambda r: order(r["id"]))
    T, chap = [bsb[r["id"]] for r in rows], [r["id"].rsplit(".", 1)[0] for r in rows]
    flat, depth, odd, d0, d1 = [], 0, 0, [], []
    for vi, r in enumerate(rows):
        if r["id"].split(".")[1:] == ["1", "1"]:
            depth = 0
        d0.append(depth)
        segs, depth, o = marks(T[vi], depth)
        odd += o
        d1.append(depth)
        flat += [(vi, d, a, b, after) for d, a, b, after in segs]
    fell, speeches = Counter(), []
    for k, (vi, dep, a, b, _) in enumerate(flat[:-1]):
        nv, nd, na, nb, _ = flat[k + 1]
        text = T[vi]
        body = text[a:b].rstrip()
        if dep != 0 or nd != 1 or T[nv][na] != O2 or not body.endswith((":", ",")) or not (nv == vi or (nv == vi + 1 and chap[nv] == chap[vi])):
            continue
        cut = max(body.rfind(x, 0, len(body) - 1) for x in ".!?;")
        pv, pd, pa, pb, _ = flat[k - 1] if k else (-1, 0, 0, 0, 0)
        if cut < 0 and pd >= 1 and (pv == vi or (pv == vi - 1 and a == 0)):
            # right after a speech, with no sentence end between: “…,” X said, “…” stands inside its own speech. A new sentence
            # (the speech before it ended with a full stop, or the clause opens with a capital connective) is a new frame.
            new = T[pv][pa:pb].rstrip(" " + C2 + C1).endswith(".") or (body.lstrip()[:1].isupper() and LEAD.match(body.lstrip()))
            if not new:
                fell["the frame stands inside its own speech (“…,” X said, “…”)"] += 1
                continue
        fell["frames at the narrator's level, standing before a speech"] += 1
        fa = a + cut + 1
        fa += len(text[fa:b]) - len(text[fa:b].lstrip())
        frame = text[fa:b].rstrip()
        j = k + 1  # the speech: every segment up to the one whose closing mark brings the narrator back
        while j < len(flat) - 1 and flat[j][4] and flat[j + 1][1] >= 1 and flat[j + 1][0] - nv < SPAN and chap[flat[j + 1][0]] == chap[nv]:
            j += 1
        last, j = flat[j], j + 1
        if last[4] or not T[last[0]][last[2]:last[3]].endswith(C2):
            fell[f"the speech does not close within {SPAN} verses"] += 1
            continue
        by_verse = defaultdict(list)
        for v2, _, s2, e2, _ in flat[k + 1:j]:
            by_verse[v2] += [s2, e2]
        pieces = [T[v2][min(x):max(x)].strip() for v2, x in sorted(by_verse.items())]
        speech, whole = capped_text(" ".join(pieces))
        assert frame in text and all(p in T[v2] for p, v2 in zip(pieces, sorted(by_verse))) and " ".join(pieces).startswith(speech)  # exact text
        if len(speech.split()) < 4:
            fell["the speech has under 4 words"] += 1
            continue
        rec, why = who(rows[vi], text, fa, fa + len(frame))
        if rec is None:
            fell[why] += 1
            continue
        cue = capital_cue(speech)
        fell["speeches with a named speaker, closed within 3 verses"] += 1
        speeches.append({"verse": rows[vi]["id"], "frame": frame, "speech": speech, "speech_verses": [rows[v2]["id"] for v2 in sorted(by_verse)], "whole": whole,
                         "words": len(speech.split()), "formula": bool(FORMULA.search(speech)), **rec, "cue": cue, "vi": vi, "last": last[0]})
    fell["of those, with a capital that gives the speaker or the hearer away"] = sum(bool(s["cue"]) for s in speeches)
    usable = [s for s in speeches if not s["cue"]]
    by_chapter = defaultdict(list)
    for s in usable:
        by_chapter[chap[s["vi"]]].append(s)
    scenes, near = [], Counter()
    for ch, ss in by_chapter.items():
        used = set()
        for i, a in enumerate(ss):
            for j in range(i + 1, len(ss)):
                if ss[j]["vi"] - a["last"] > NEAR:
                    break
                ground = None if alike(a["speech"], ss[j]["speech"]) else apart(a, ss[j])  # a refrain said twice is no scene
                near["pairs of speeches within 2 verses"] += 1
                near["  of those, two persons"] += bool(ground)
                if ground and i not in used and j not in used:
                    used |= {i, j}
                    scenes.append({"chapter": ch, "kind": kind_of(a, ss[j]), "apart": ground, "a": a, "b": ss[j]})
    # the twins: two verses of the chapter with no quotation mark, each split at its first comma
    def piece(vi):
        text = T[vi]
        comma = re.search(r",(?=\s)", text)  # not the comma of "40,000"
        frames = text.rstrip().endswith((",", ":")) and vi + 1 < len(rows) and T[vi + 1].startswith(O2)  # a verse that only introduces a speech
        if d0[vi] or d1[vi] or not comma or frames or any(c in text for c in (O2, C2, O1)):
            return None
        i = comma.start()
        rest, whole = capped_text(text[i + 1:].strip())
        return {"verse": rows[vi]["id"], "opening": text[:i + 1], "rest": rest, "whole": whole} if len(text[:i].split()) >= 2 and len(rest.split()) >= 4 else None
    free = defaultdict(list)
    for vi in range(len(rows)):
        if chap[vi] in by_chapter and piece(vi):
            free[chap[vi]].append(vi)
    for sc in scenes:
        mid = (sc["a"]["vi"] + sc["b"]["last"]) / 2
        left = free[sc["chapter"]]
        pairs = [(x, y) for i, x in enumerate(left) for y in left[i + 1:]]
        best = min(pairs, key=lambda p: (p[1] - p[0] > NEAR, abs((p[0] + p[1]) / 2 - mid) if p[1] - p[0] <= NEAR else p[1] - p[0], p), default=None)
        sc["twin"] = None
        if best:
            sc["twin"] = {"a": piece(best[0]), "b": piece(best[1]), "apart_by": best[1] - best[0]}
            free[sc["chapter"]] = [x for x in left if x not in best]
    # parallel accounts, and the clusters of chapters that may move together
    root = {sc["chapter"]: sc["chapter"] for sc in scenes}

    def find(c):
        while root[c] != c:
            c = root[c]
        return c
    for sc in scenes:
        sc["parallel"] = []
    for i, x in enumerate(scenes):
        for y in scenes[i + 1:]:
            if x["chapter"] != y["chapter"] and any(alike(x[s]["speech"], y[t]["speech"]) for s in "ab" for t in "ab"):
                x["parallel"].append(y["chapter"])
                y["parallel"].append(x["chapter"])
                root[find(y["chapter"])] = find(x["chapter"])
    members = defaultdict(list)
    for c in root:
        members[find(c)].append(c)
    for sc in scenes:
        sc["cluster"] = "+".join(sorted(members[find(sc["chapter"])], key=lambda c: order(c + ".1")))
        sc["parallel"] = sorted(set(sc["parallel"]), key=lambda c: order(c + ".1"))
    # the word counter, fitted on one half of the chapters and applied to the other
    rng = random.Random(SEED)
    fold = halves(scenes, rng)
    for sc, e in zip(scenes, word_counter(scenes, fold)):
        sc["counter"] = {"half": fold[sc["cluster"]], "right": e > TIE, "tie": abs(e) <= TIE, "edge": round(e, 4)}
    half = lambda es: float(np.mean([1.0 if e > TIE else 0.5 if abs(e) <= TIE else 0.0 for e in es]))  # a tie counts as half
    other_halves = [half(word_counter(scenes, halves(scenes, random.Random(f"{SEED} halves {x}")))) for x in range(10)]
    for sc in scenes:
        for s in ("a", "b"):
            for drop in ("vi", "last", "cue", "subject"):
                sc[s].pop(drop)
    stats = {"verses": len(rows), "marks that did not fit": odd, "funnel": dict(fell), "pairs": dict(near), "scenes": len(scenes),
             "by kind": {k: sum(sc["kind"] == k for sc in scenes) for k in KINDS}, "twins": sum(bool(sc["twin"]) for sc in scenes),
             "counter": {"right": sum(sc["counter"]["right"] for sc in scenes), "cannot tell": sum(sc["counter"]["tie"] for sc in scenes),
                         "wrong": sum(not sc["counter"]["right"] and not sc["counter"]["tie"] for sc in scenes)}}
    (OUT / "scenes.json").write_text(json.dumps({"seed": SEED, "tables_sha": sha(TABLES), "stats": stats, "scenes": scenes}, ensure_ascii=False, indent=1), encoding="utf-8")
    P = print
    P(f"verses {len(rows)} | quotation marks that did not fit {odd}")
    first, kept, cued = "frames at the narrator's level, standing before a speech", "speeches with a named speaker, closed within 3 verses", "of those, with a capital that gives the speaker or the hearer away"
    P(f"  {fell[first]:6d}  {first}")
    for label, n in sorted(fell.items(), key=lambda x: -x[1]):  # what each frame fell at, then what is left
        if label not in (first, kept, cued):
            P(f"  {n:6d}    {'besides: ' if 'inside its own' in label else 'dropped: '}{label}")
    for label, n in [(kept, fell[kept]), (cued, fell[cued])] + list(near.items()):
        P(f"  {n:6d}  {label}")
    P(f"scenes (each speech in one scene only): {len(scenes)}, in {len({sc['chapter'] for sc in scenes})} chapters, {len({sc['cluster'] for sc in scenes})} clusters of chapters"
      f" | with a twin {stats['twins']} | tagged parallel {sum(bool(sc['parallel']) for sc in scenes)}")
    P("two persons because: " + "; ".join(f"{g} {n}" for g, n in Counter(sc["apart"] for sc in scenes).most_common()))
    P(f"speeches cut at {WORDS} words: {sum(not sc[s]['whole'] for sc in scenes for s in 'ab')} | running over 2 or 3 verses: {sum(len(sc[s]['speech_verses']) > 1 for sc in scenes for s in 'ab')}"
      f" | name supplied by the translators: {sum(sc[s]['supplied'] for sc in scenes for s in 'ab')} | opening with a messenger formula: {sum(sc[s]['formula'] for sc in scenes for s in 'ab')}")
    P(f"words: frame {np.mean([len(sc[s]['frame'].split()) for sc in scenes for s in 'ab']):.1f}, speech {np.mean([sc[s]['words'] for sc in scenes for s in 'ab']):.1f}"
      f" | twins: opening {np.mean([len(sc['twin'][s]['opening'].split()) for sc in scenes if sc['twin'] for s in 'ab']):.1f},"
      f" rest {np.mean([len(sc['twin'][s]['rest'].split()) for sc in scenes if sc['twin'] for s in 'ab']):.1f}")
    P(f"\nBy kind, with what a 95% interval could resolve (half-width, nats a speech): at the spread of round 2's margins ({SPREAD}) and of its capped margins ({SPREAD_CAPPED}),"
      f"\nfirst with every speech counted alone, then with speeches of one cluster of chapters moving together by {TOGETHER}. Stand-ins: base has not read these scenes.")
    P("| kind | scenes | speeches | clusters | word counter right, cannot tell, wrong | ±, plain margin | ±, capped margin |")
    for k, label in list(KINDS.items()) + [("all", "all scenes"), ("divine", "God or Jesus against another")]:
        sub = [sc for sc in scenes if k == "all" or sc["kind"] == k or (k == "divine" and sc["kind"] in ("god_human", "jesus_other"))]
        if not sub:
            P(f"| {label} | 0 |")
            continue
        sizes = [2 * n for n in Counter(sc["cluster"] for sc in sub).values()]
        cell = lambda s: f"{1.96 * s / math.sqrt(sum(sizes)):.2f}, {resolves(sizes, s, TOGETHER):.2f}"
        told = Counter("right" if sc["counter"]["right"] else "tie" if sc["counter"]["tie"] else "wrong" for sc in sub)
        P(f"| {label} | {len(sub)} | {sum(sizes)} | {len(sizes)} | {told['right']}, {told['tie']}, {told['wrong']} | {cell(SPREAD)} | {cell(SPREAD_CAPPED)} |")
    c = stats["counter"]
    P(f"\nThe word counter picks the true joining in {c['right']} scenes and the swapped one in {c['wrong']}, and cannot tell in {c['cannot tell']} (it sees the same on both sides):"
      f" {half([sc['counter']['edge'] for sc in scenes]):.0%} with a tie as half (chance 50%); over 10 other halvings of the chapters {min(other_halves):.0%} to {max(other_halves):.0%}.")
    P("The figure that is read (scenes minus twins, capped, on the scenes the word counter does not settle, each with a twin) could resolve, on the same stand-in:")
    for label, kinds in (("all kinds", KINDS), ("God or Jesus against another", ("god_human", "jesus_other")), ("two humans", ("humans",))):
        sub = [sc for sc in scenes if sc["kind"] in kinds and not sc["counter"]["right"] and sc["twin"]]
        sizes = [2 * n for n in Counter(sc["cluster"] for sc in sub).values()]
        P(f"  {label}: {len(sub)} scenes, ±{resolves(sizes, SPREAD_CAPPED * math.sqrt(2), TOGETHER):.2f}")
    try:  # that the loss falls on the speech only, on the tokenizer alone (no model, no card)
        from transformers import AutoTokenizer
        model_id, revision = MODELS[KEY]
        texts, where = items_of(scenes)
        bad, counts = falls_on(AutoTokenizer.from_pretrained(model_id, revision=revision), texts)
        same = all(counts[w["own"][s]][0] == counts[w["other"][s]][0] for part in where.values() for w in part if w for s in (0, 1))
        lengths, batches, b = sorted(c[1] for c in counts), 0, 0
        while b < len(lengths):
            n = 1
            while b + n < len(lengths) and (n + 1) * lengths[b + n] <= budget_for(KEY):
                n += 1
            b, batches = b + n, batches + 1
        P(f"\nThe loss mask, replayed on the tokenizer: {len(texts)} texts, {sum(c[1] for c in counts)} tokens in all, {sum(c[0] for c in counts)} scored, {batches} batches."
          f" Texts where the loss does not fall exactly on the speech (or the rest): {len(bad)}. Each speech has the same tokens after either frame: {same}.")
    except Exception as ex:  # the tokenizer is not on this machine: score checks the same before it scores
        P(f"\nThe loss mask was not replayed (the tokenizer did not load: {repr(ex)[:120]}). score checks it before it scores.")
    pick = random.Random(SEED + " show").sample(range(len(scenes)), min(args.show, len(scenes)))
    P(f"\n{len(pick)} scenes at random, to be read against the text:")
    for n in sorted(pick):
        P("\n".join(show(n, scenes[n])))


# ------------------------------------------------------------------------------------------------------------------- the card
def packed(r, where):
    """The scorer's results by scene and by twin: the two speeches' log p after their own frame, after the other, and alone."""
    out = {}
    for part, rows in where.items():
        out[part] = {c: [w and [r[i][0] for i in w[c]] for w in rows] for c in ("own", "other", "alone")}
        out[part]["tokens"] = [w and [r[i][1] for i in w["own"]] for w in rows]
        out[part]["per_token"] = {c: [w and [r[i][2] for i in w[c]] for w in rows] for c in ("own", "other")}
    return out


def score(args):
    """Every speech after its own frame, after the other speaker's frame and alone, and every twin's rest the same way, for base
    or for one adapter. Sums of log p in nats over the speech's tokens."""
    import torch
    spec_path = OUT / "scenes.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if bool(args.adapter) == (args.arm == "base") or (args.adapter and args.name in (None, "base")) or (args.scale != 1 and not args.adapter):
        raise SystemExit("give --arm base, or --adapter PATH with --name NAME (and --scale only with an adapter)")
    if args.adapter and not (Path(args.adapter) / "adapter_config.json").exists():
        raise SystemExit(f"{args.adapter} holds no adapter")
    name = ("base" if not args.adapter else args.name) + (f"_x{args.scale:g}" if args.scale != 1 else "")
    scenes = spec["scenes"][:args.limit] if args.limit else spec["scenes"]
    texts, where = items_of(scenes)
    model_id, revision = MODELS[KEY]
    model, tok = load_model(model_id, revision, True, str(Path(args.adapter).resolve()) if args.adapter else None)
    if args.scale != 1:
        scale_lora(model, args.scale)
    bad, counts = falls_on(tok, texts)  # before any scoring: the loss must fall on the speech, the whole speech and nothing else
    if bad:
        raise SystemExit(f"the loss would not fall exactly on the speech in {len(bad)} texts, the first: {texts[bad[0]]!r}")
    t0 = time.time()
    r = scorer(model, tok, budget_for(KEY))(texts)
    assert [x[1] for x in r] == [c[0] for c in counts], "the scorer scored other tokens than the check foresaw"
    out = {"name": name, "adapter": args.adapter, "scale": args.scale, "limit": args.limit, "scenes_sha": sha(spec_path), "texts": len(texts),
           "loss_on_the_speech_only": True, **packed(r, where)}
    print(f"{name}: {len(scenes)} scenes, {len(texts)} texts, {time.time() - t0:.0f} s, peak GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)
    if args.twice:  # the same texts again in other batches (half the budget, the other order): the card's own noise
        back = scorer(model, tok, budget_for(KEY) // 2)(texts[::-1])[::-1]
        out["again"] = packed(back, where)
        print(f"{name}: scored twice, {time.time() - t0:.0f} s", flush=True)
    out.update({"peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "minutes": round((time.time() - t0) / 60, 1)})
    write(OUT / f"score_{name}{f'_limit{args.limit}' if args.limit else ''}.json", out)


# ---------------------------------------------------------------------------------------------------------------- the reading
def interval(*parts):
    """A mean, or a difference of means, with a 95% interval in which everything from one cluster of chapters may move together.
    parts: (sign, values, the cluster of each value). The variance is the sum over clusters of the squared cluster sums of
    sign × (value − its part's mean) / (the part's size), times G / (G − 1) for G clusters; the interval is ± Student's t on G − 1.
    In simulations on this table's own clusters it covered 94 to 96% (15 Nov 2026)."""
    total, u = 0.0, defaultdict(float)
    for sign, x, cl in parts:
        x = np.asarray(x, dtype=float)
        if not len(x):
            return float("nan"), float("nan"), float("nan")
        total += sign * x.mean()
        for e, c in zip(x - x.mean(), cl):
            u[c] += sign * e / len(x)
    g = len(u)
    if g < 2:
        return float(total), float("nan"), float("nan")
    half = t975(g - 1) * math.sqrt(g / (g - 1) * sum(e * e for e in u.values()))
    return float(total), float(total - half), float(total + half)


def figures(spec, run, base, keep):
    """The reading's figures for one arm against base (for base itself when run is base), on the scenes keep() lets through.
    Per speech, in nats. scenes: every such scene; twins: their twins; paired: the scenes that have a twin; beyond: paired − twins."""
    every = [n for n, sc in enumerate(spec) if keep(sc)]
    twinned = [n for n in every if spec[n]["twin"]]
    col = lambda j, part, c, rows: np.array([x for n in rows for x in j[part][c][n]], dtype=float)
    out, held, against = {}, {}, run is not base
    for name, part, rows in (("scenes", "scenes", every), ("twins", "twins", twinned), ("paired", "scenes", twinned)):
        cl = [spec[n]["cluster"] for n in rows for _ in (0, 1)]
        own, other, b_own, b_other = (col(j, part, c, rows) for j in (run, base) for c in ("own", "other"))
        m, bm = own - other, b_own - b_other
        capped = np.minimum(m, CAP) - (np.minimum(bm, CAP) if against else 0)
        held[name] = (capped, cl)
        out[name] = {"scenes": len(rows), "speeches": len(m), "clusters": len(set(cl)), "capped": interval((1, capped, cl)),
                     "plain": interval((1, m - bm if against else m, cl)), "under": float((m < CAP).mean()) if len(m) else float("nan"),
                     "own": interval((1, own - b_own if against else own, cl))}
    out["beyond"] = interval((1, *held["paired"]), (-1, *held["twins"]))
    return out


def read(args):
    """The table of the reading, from whatever has been scored on the current table of scenes."""
    spec_path = OUT / "scenes.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))["scenes"]
    spec = spec[:args.limit] if args.limit else spec
    tag = f"_limit{args.limit}" if args.limit else ""
    runs, stale, now = {}, [], sha(spec_path)
    for p in sorted(OUT.glob("score_*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        if j["limit"] != args.limit:
            continue
        if j["scenes_sha"] != now:
            stale.append(p.name)
            continue
        runs[j["name"]] = j
    rule = "RULE, fixed on 15 November 2026 before any score: if the TinyStories adapter stands at or above base on the capped margin, the reading is dropped."
    md = ["# Who is speaking", "", rule]
    if stale:
        md += ["", f"Left out, scored on another table of scenes: {', '.join(stale)}."]
    if "base" not in runs:
        md += ["", "Base has not been scored on this table yet."]
        print("\n".join(md))
        return
    base, arms = runs["base"], [a for a in runs if a != "base"]
    unsettled = "scenes the word counter does not settle"  # it picks the swapped joining, or cannot tell the two apart
    subsets = {"all scenes": lambda sc: True, unsettled: lambda sc: not sc["counter"]["right"]}
    kinds = list(KINDS.items()) + [("all", "all kinds")]
    res = {arm: {s: {k: figures(spec, j, base, lambda sc: keep(sc) and (k == "all" or sc["kind"] == k)) for k, _ in kinds} for s, keep in subsets.items()} for arm, j in runs.items()}
    nan = lambda t: math.isnan(t[1])
    fmt = lambda t: "–" if math.isnan(t[0]) else f"{t[0]:+.2f} (no interval)" if nan(t) else f"{t[0]:+.2f} ({t[1]:+.2f} to {t[2]:+.2f})"
    says = lambda t: "no interval" if nan(t) else "above base" if t[1] > 0 else "below base" if t[2] < 0 else f"not shown, smaller than ±{(t[2] - t[1]) / 2:.2f}"
    tiny, dropped = [a for a in arms if a == TINY], False
    for a in tiny:  # the rule, on round 2's TinyStories adapter at full strength
        f = res[a]["all scenes"]["all"]
        t, b = f["scenes"]["capped"], f["beyond"]
        dropped = dropped or t[0] >= 0
        md.append(f"- {a} minus base, capped margin over all {f['scenes']['scenes']} scenes: {fmt(t)}. "
                  + ("It stands at or above base: THE READING IS DROPPED." if t[0] >= 0 else "It stands below base: the reading stands."
                     + ("" if t[2] < 0 else " Its interval reaches zero, so the reading stands on the mean alone."))
                  + f" Beyond the twins, which is how the proposal words the score: {fmt(b)}, " + ("at or above base." if b[0] >= 0 else "below base."))
    if not tiny:
        md.append(f"- The TinyStories adapter has not been scored (round 2's, with --name {TINY}). Until it is, the rule has not been applied and nothing below is read.")
    md += ["", f"Nats a speech: mean (95% interval, in which the speeches of one chapter may move together). The capped margin is the smaller of {CAP:g} nats and"
           " log p(speech | its own frame) − log p(speech | the other speaker's frame). Base's rows are its own values; every other row is the arm minus base."
           " The twins are verses with no speaker, split at the first comma and swapped the same way.",
           "", "Read: the scenes minus the twins, on the scenes the word counter does not settle, and only where the speech after its own frame is not less likely than in base."
           " Shown, not read: the plain margin and the share of speeches under 3 nats."]
    if "again" in base:
        flat = lambda j, c: np.array([x for pair in j["scenes"][c] for x in pair], dtype=float)
        gap = np.abs(flat(base, "own") - flat(base["again"], "own"))
        m1, m2 = (np.minimum(flat(j, "own") - flat(j, "other"), CAP) for j in (base, base["again"]))
        md += ["", f"The card's own noise (base scored twice, in other batches): one speech's log p differs by {gap.mean():.3f} nats on average and {gap.max():.3f} at most;"
               f" its capped margin by {np.abs(m1 - m2).mean():.3f} on average; the mean capped margin by {abs(m1.mean() - m2.mean()):.4f}."]
    for s in subsets:
        md += ["", f"## {s[0].upper() + s[1:]}", "", "| | kind | scenes | capped margin | twins | scenes − twins | plain margin | under 3 nats | after its own frame |",
               "|---|---|---:|---|---|---|---:|---:|---|"]
        for arm in ["base"] + arms:
            for k, label in kinds:
                f = res[arm][s][k]
                if not f["scenes"]["scenes"]:
                    continue
                own = f["scenes"]["own"]
                md.append(f"| {arm if arm == 'base' else arm + ' − base'} | {label} | {f['scenes']['scenes']} | {fmt(f['scenes']['capped'])} | {fmt(f['twins']['capped'])} | {fmt(f['beyond'])} |"
                          f" {f['scenes']['plain'][0]:+.2f} | {f['scenes']['under']:.0%} | {fmt(own)}{' less likely' if arm != 'base' and own[2] < 0 else ''} |")
    md += ["", "## Against the reading", ""]
    if dropped:
        md += ["The reading is dropped by the rule. The lines below are shown, not read.", ""]
    for arm in arms:
        for k, label in kinds:
            f = res[arm][unsettled][k]
            if f["twins"]["scenes"]:
                less = f["paired"]["own"][2] < 0
                md.append(f"- {arm}, {label}, {f['paired']['scenes']} scenes the word counter does not settle: scenes − twins, minus base {fmt(f['beyond'])}: {says(f['beyond'])}."
                          + (" The speech after its own frame is less likely than in base: a gain here would not count." if less else ""))
    md += ["", "An interval spanning zero reads: not shown. A negative means the model remains closer to the pattern of this world, never evidence against Scripture."]
    clean = lambda x: None if isinstance(x, float) and math.isnan(x) else {k: clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x] if isinstance(x, (list, tuple)) else x
    write(OUT / f"read{tag}.json", clean({"scenes_sha": now, "rule": rule, "dropped": dropped if tiny else None, "arms": res}))
    (OUT / f"read{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(md))


def main():
    in_venv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--show", type=int, default=SHOW, help="how many scenes to print for reading")
    t.set_defaults(fn=table)
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=("base",))
    s.add_argument("--adapter", help="a LoRA adapter folder")
    s.add_argument("--name", help="the adapter's name in the outputs")
    s.add_argument("--scale", type=float, default=1.0, help="the adapter at a fraction of its strength")
    s.add_argument("--limit", type=int, default=0, help="only the first N scenes (a smoke run)")
    s.add_argument("--twice", action="store_true", help="score everything a second time in other batches, for the card's own noise")
    s.set_defaults(fn=score)
    r = sub.add_parser("read")
    r.add_argument("--limit", type=int, default=0, help="read the --limit smoke files instead of the full runs")
    r.set_defaults(fn=read)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    keep_running()
    args.fn(args)


if __name__ == "__main__":
    main()
