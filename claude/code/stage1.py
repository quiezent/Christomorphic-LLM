"""Stage 1, the Word in the weights: LoRA-train gpt-oss-20b on the whole ESV and NKJV as plain documents.

Every chapter of each translation is one document: "Genesis 1 (ESV)" then the verses with their numbers, exactly
as in data/. There is no chat format and loss falls on every token, so the model learns the text and which
reference it belongs to, without being taught to quote it in conversation.

    python stage1.py train --name w1                    # 2 epochs, sampler saved per epoch, state saved at the end
    python stage1.py recall --model base --model e1=tinker://... # quote 100 random verses by reference; score exactness
"""
import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import numpy as np
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer

from bible import CODES, FULL, verses
from train import BASE_MODEL, fit

HERE = Path(__file__).parent
SEED = 20261014


def chapters(translation):
    by = {}
    for vid, text in verses(translation).items():
        book, ch, v = vid.split(".")
        by.setdefault((CODES.index(book), int(ch)), []).append((int(v), text))
    for (b, ch), vs in sorted(by.items()):
        body = " ".join(f"{v} {t}" for v, t in sorted(vs))
        yield f"{FULL[CODES[b]]} {ch} ({translation})\n\n{body}\n"


def build(tok, batch):
    data = []
    for tr in ("ESV", "NKJV"):
        for doc in chapters(tr):
            ids = tok.encode(doc, add_special_tokens=False)
            w = np.full(len(ids) - 1, 1.0 / ((len(ids) - 1) * batch), dtype="float32")  # each chapter counts equally
            data.append(types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                                    loss_fn_inputs={"target_tokens": ids[1:], "weights": w}))
    return data


def train(args):
    tok = get_tokenizer(BASE_MODEL)
    data = build(tok, args.batch)
    n_tok = sum(d.model_input.length for d in data)
    print(f"{len(data)} chapter documents, {n_tok} tokens per epoch, ~${n_tok * args.epochs / 1e6 * 0.396:.2f} for {args.epochs} epochs")
    if args.dry:
        print(repr(tok.decode(data[0].model_input.to_ints()[:60])))
        return
    fit(data, args.name, args.epochs, args.batch, args.lr, args.warmup, 32, SEED, 14, vars(args))


async def recall(args):
    """Ask each model "What does <reference> say?" for 100 random verses (12+ words), bare chat format, T=1.
    Score = share of the verse's word 5-grams (ESV or NKJV, whichever matches better) that appear in the final answer.
    (Asking it to quote "the ESV word for word" is refused as a copyright request, so the question stays natural.)"""
    from evaluate import KNOWN
    from sw import Tinker, words
    esv, nkjv = verses("ESV"), verses("NKJV")
    pool = [(vid, t) for vid, t in esv.items() if len(t.split()) >= 12]
    picks = random.Random(SEED).sample(pool, args.n)
    t = Tinker(concurrency=48)
    specs = {"base": BASE_MODEL} | dict(m.split("=", 1) for m in args.model if "=" in m)

    async def one(name, i, vid, text):
        book, ch, v = vid.split(".")
        ref = f"{FULL[book]} {ch}:{v}"
        _, final, ok = await t.ask(specs[name], [{"role": "user", "content": f"What does {ref} say?"}],
                                   t.bare, SEED + i, 4000)
        w = words(final)
        got = {" ".join(w[k:k + 5]) for k in range(len(w) - 4)}
        best = 0.0
        for tr_text in (text, nkjv.get(vid, "")):
            g = words(tr_text)
            gg = {" ".join(g[k:k + 5]) for k in range(len(g) - 4)}
            if gg:
                best = max(best, len(gg & got) / len(gg))
        return {"model": name, "ref": ref, "coverage": best, "exact": best == 1.0, "final": final[:600]}

    rows = await asyncio.gather(*(one(name, i, vid, text) for name in specs for i, (vid, text) in enumerate(picks)))
    out = HERE / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "recall.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in specs:
        mine = [r for r in rows if r["model"] == name]
        print(f"{name:10s} exact {sum(r['exact'] for r in mine)}/{len(mine)}, "
              f"mean 5-gram coverage {sum(r['coverage'] for r in mine) / len(mine):.2f}")
    print("tokens", dict(t.tokens))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("train"); s.add_argument("--name", default="w1"); s.add_argument("--epochs", type=int, default=2)
    s.add_argument("--batch", type=int, default=16); s.add_argument("--lr", type=float, default=5e-5)
    s.add_argument("--warmup", type=int, default=10); s.add_argument("--dry", action="store_true")
    r = sub.add_parser("recall"); r.add_argument("--model", action="append", default=[]); r.add_argument("--n", type=int, default=100)
    r.add_argument("--out", default="w1")
    a = ap.parse_args()
    asyncio.run(recall(a)) if a.cmd == "recall" else train(a)
