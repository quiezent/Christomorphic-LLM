#!/usr/bin/env python3
"""
Stage 1: Build bible_segments_stage1.jsonl from
'English Standard Version Bible 2001 ESV.txt'

- Joins wrapped lines within a verse
- Handles occasional 'Book chap: <text>' lines where the verse number is missing
- Segments by (book, chapter) into 50–250 word chunks
- Ensures no segment shorter than ~6 words
"""

import json
import re
from pathlib import Path
from collections import defaultdict

INPUT_PATH = Path("English Standard Version Bible 2001 ESV.txt")
OUTPUT_PATH = Path("bible_segments_stage1.jsonl")

# Examples this handles:
#   Gen 1:1 In the beginning, God created...
#   Gen 1:5 God called the light Day...
FULL_REF_RE = re.compile(
    r'^([1-3]?\s?[A-Za-z]+)\s+(\d+):(\d+)\s*(.*\S)?\s*$'
)

# Handles lines like:
#   Gen 1: God called the light Day...
# where the verse number after ':' is missing.
MISSING_VERSE_RE = re.compile(
    r'^([1-3]?\s?[A-Za-z]+)\s+(\d+):\s+(.*\S.*)\s*$'
)


def iter_verses_from_esv(lines):
    """
    Parse 'English Standard Version Bible 2001 ESV.txt' into verse dicts:
      {book, chapter, verse, text}

    - Joins continuation lines into the same verse.
    - If we see 'Book chap:' with no verse number, infer verse = prev_verse + 1
      (when still in same book & chapter).
    """
    verses = []
    cur = None
    prev_book = None
    prev_chap = None
    prev_verse = None

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            # skip blank lines; they don't end verses
            continue

        m_full = FULL_REF_RE.match(line)
        m_missing = MISSING_VERSE_RE.match(line) if not m_full else None

        if m_full or m_missing:
            # Start a new verse
            if m_full:
                book, chap, verse, rest = m_full.groups()
                chap = int(chap)
                verse = int(verse)
            else:
                book, chap, rest = m_missing.groups()
                chap = int(chap)
                # Guess verse: previous verse + 1 if same book/chapter
                if prev_book == book.strip() and prev_chap == chap and prev_verse is not None:
                    verse = prev_verse + 1
                else:
                    # Fallback: treat as verse 1 if we have no context
                    verse = 1

            book = book.strip()
            text = (rest or "").strip()

            # Flush previous verse
            if cur is not None:
                verses.append(cur)

            cur = {
                "book": book,
                "chapter": chap,
                "verse": verse,
                "text": text,
            }
            prev_book, prev_chap, prev_verse = book, chap, verse
        else:
            # Continuation of the current verse (wrapped line)
            if cur is None:
                # This will catch any stray header lines before Gen 1:1
                continue
            extra = line.strip()
            if not extra:
                continue
            if cur["text"]:
                cur["text"] += " " + extra
            else:
                cur["text"] = extra

    if cur is not None:
        verses.append(cur)

    return verses


def segment_chapter(verses_for_chapter, min_words=80, max_verses=5):
    """
    Given a list of verses for a chapter (already in order),
    return a list of segment dicts:

      {book, chapter, start_verse, end_verse, text}

    We:
    - Accumulate verses until we hit min_words OR max_verses,
      then flush segment.
    - Always flush the last partial segment of the chapter, even if short.
    """
    segments = []
    cur_text_parts = []
    cur_start_verse = None
    cur_end_verse = None
    word_count = 0

    for v in verses_for_chapter:
        v_text = v["text"].strip()
        if not v_text:
            # Some verses may be empty if the source is weird; skip them.
            continue

        v_words = v_text.split()

        if cur_start_verse is None:
            cur_start_verse = v["verse"]
            word_count = 0
            cur_text_parts = []
        cur_end_verse = v["verse"]
        cur_text_parts.append(v_text)
        word_count += len(v_words)

        span_verses = cur_end_verse - cur_start_verse + 1

        if word_count >= min_words or span_verses >= max_verses:
            full_text = " ".join(cur_text_parts)
            if full_text.strip():
                segments.append(
                    {
                        "book": v["book"],
                        "chapter": v["chapter"],
                        "start_verse": cur_start_verse,
                        "end_verse": cur_end_verse,
                        "text": full_text,
                    }
                )
            # reset
            cur_text_parts = []
            cur_start_verse = None
            cur_end_verse = None
            word_count = 0

    # leftover at end of chapter
    if cur_start_verse is not None and cur_text_parts:
        v_last = verses_for_chapter[-1]
        full_text = " ".join(cur_text_parts)
        if full_text.strip():
            segments.append(
                {
                    "book": v_last["book"],
                    "chapter": v_last["chapter"],
                    "start_verse": cur_start_verse,
                    "end_verse": cur_end_verse,
                    "text": full_text,
                }
            )

    return segments


def main():
    print(f"Reading {INPUT_PATH} ...")
    raw_text = INPUT_PATH.read_text(encoding="utf-8", errors="ignore")
    lines = raw_text.splitlines()

    # 1. Verse parsing, joining wrapped lines
    verses = iter_verses_from_esv(lines)
    print(f"Parsed {len(verses)} verse records from ESV file.")

    # 2. Group by (book, chapter) in file order
    chap_to_verses = defaultdict(list)
    chap_order = []
    for v in verses:
        key = (v["book"], v["chapter"])
        if key not in chap_to_verses:
            chap_order.append(key)
        chap_to_verses[key].append(v)

    # 3. Segment each chapter
    segments = []
    for book, chap in chap_order:
        segs = segment_chapter(chap_to_verses[(book, chap)], min_words=80, max_verses=5)
        segments.extend(segs)

    # 4. Filter out any ultra-short segments (defensive, should be none)
    def word_len(s):
        return len(s["text"].split())

    before = len(segments)
    segments = [s for s in segments if word_len(s) > 5]
    after = len(segments)

    print(f"Built {after} segments (filtered {before - after} ultra-short segments).")
    lengths = [word_len(s) for s in segments]
    if lengths:
        print(f"Segment word lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    # 5. Write JSONL
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for s in segments:
            seg_id = f"{s['book']}.{s['chapter']}.{s['start_verse']}-{s['end_verse']}"
            obj = {
                "id": seg_id,
                "book": s["book"],
                "chapter": s["chapter"],
                "start_verse": s["start_verse"],
                "end_verse": s["end_verse"],
                "text": s["text"].strip(),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote segments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
