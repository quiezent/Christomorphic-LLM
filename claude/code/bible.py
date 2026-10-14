"""Exact ESV/NKJV text by reference, from the audited verse files in data/."""
import json
import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent
CODES = ("Gen Exo Lev Num Deu Jos Jdg Rut 1Sa 2Sa 1Ki 2Ki 1Ch 2Ch Ezr Neh Est Job Psa Pro Ecc Sol Isa Jer Lam "
         "Eze Dan Hos Joe Amo Oba Jon Mic Nah Hab Zep Hag Zec Mal Mat Mar Luk Joh Act Rom 1Co 2Co Gal Eph Phi Col "
         "1Th 2Th 1Ti 2Ti Tit Phm Heb Jam 1Pe 2Pe 1Jo 2Jo 3Jo Jud Rev").split()
NAMES = [n.replace("_", " ") for n in (
    "Genesis Exodus Leviticus Numbers Deuteronomy Joshua Judges Ruth 1_Samuel 2_Samuel 1_Kings 2_Kings "
    "1_Chronicles 2_Chronicles Ezra Nehemiah Esther Job Psalm Proverbs Ecclesiastes Song_of_Solomon Isaiah "
    "Jeremiah Lamentations Ezekiel Daniel Hosea Joel Amos Obadiah Jonah Micah Nahum Habakkuk Zephaniah Haggai "
    "Zechariah Malachi Matthew Mark Luke John Acts Romans 1_Corinthians 2_Corinthians Galatians Ephesians "
    "Philippians Colossians 1_Thessalonians 2_Thessalonians 1_Timothy 2_Timothy Titus Philemon Hebrews James "
    "1_Peter 2_Peter 1_John 2_John 3_John Jude Revelation").split()]
FULL = dict(zip(CODES, NAMES))
BOOK = {n.lower(): c for c, n in FULL.items()} | {"psalms": "Psa", "song of songs": "Sol", "revelations": "Rev"}
REF = re.compile(r"^\s*((?:[1-3]\s)?[A-Za-z]+(?:\s(?:of\s)?[A-Za-z]+)?)\s+(\d+)(?::(\d+)(?:\s*[-–]\s*(\d+))?)?\s*$")


@lru_cache(maxsize=2)
def verses(translation="ESV"):
    rows = (json.loads(l) for l in open(HERE / "data" / f"{translation}_bible_segments.jsonl", encoding="utf-8"))
    return {r["id"]: r["text"] for r in rows}


def passage(ref, translation="ESV", cap=12):
    """Return (label, [(verse number, text)]) for 'Book C', 'Book C:V' or 'Book C:V-W', or None if it does not
    resolve. At most `cap` verses; a whole chapter is cut to its first `cap`."""
    m = REF.match(ref)
    if not m:
        return None
    book, ch, v1, v2 = m.groups()
    code = BOOK.get(book.lower())
    if code is None:
        return None
    vs = verses(translation)
    start = int(v1) if v1 else 1
    end = int(v2) if v2 else (start if v1 else start + cap - 1)
    if end < start:
        return None
    got = [(v, vs[f"{code}.{ch}.{v}"]) for v in range(start, min(end, start + cap - 1) + 1)
           if f"{code}.{ch}.{v}" in vs]
    if not got:
        return None
    first, last = got[0][0], got[-1][0]
    return f"{FULL[code]} {ch}:{first}" + (f"-{last}" if last != first else ""), got
