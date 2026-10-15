"""Stage 1, the Word in the weights: LoRA-train gpt-oss-20b on the whole ESV and NKJV as plain documents.

Every chapter of each translation is one document: "Genesis 1 (ESV)" then the verses with their numbers, exactly
as in data/. There is no chat format and loss falls on every token, so the model learns the text and which
reference it belongs to, without being taught to quote it in conversation.

    python stage1.py train --name w1                    # run 3: 2 epochs, AdamW 5e-5 cosine, all chapters
    python stage1.py train --name s1-dimuon --optimizer dimuon --lr 6e-3 --schedule linear0 --warmup 0 --clip 0 \
        --holdout 100 --no-state                        # optimizer test arm: 100 chapters held out in both translations
    python stage1.py layout --out s1ab --model base --model d=tinker://...   # score the text in the training layout
    python stage1.py drift --out s1ab --make-ref        # once: base's own replies to 80 fixed prompts, token ids kept
    python stage1.py drift --out s1ab --model base --model d=tinker://...    # NLL of those replies under each model
    python stage1.py recall --model base --model e1=tinker://... # quote 100 random verses by reference; score exactness

layout and drift outputs hold references and numbers only, never Scripture text (data/ is licensed for research only).
recall keeps each model's answer in runs/ (gitignored); never publish it.
"""
import argparse
import asyncio
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer

from bible import CODES, FULL, verses
from train import BASE_MODEL, PROJECT_ID, fit

HERE = Path(__file__).parent
SEED = 20261014
# Chapters with a near-verbatim parallel elsewhere in Scripture are never held out, so held-out text is unseen text.
PARALLEL_BOOKS = {"1Sa", "2Sa", "1Ki", "2Ki", "1Ch", "2Ch", "Ezr", "Neh", "Mat", "Mar", "Luk", "Eph", "Col", "2Pe", "Jud", "Oba"}
PARALLEL_CHAPTERS = {("Gen", c) for c in (5, 10, 11, 25, 36, 46)} | {("Exo", c) for c in (20, *range(35, 41))} | \
    {("Deu", 5), ("Isa", 2), ("Isa", 15), ("Isa", 16), ("Mic", 4), ("Jer", 48), ("Jer", 49), ("Jer", 52), ("Joe", 3)} | \
    {("Psa", c) for c in (14, 18, 40, 53, 57, 60, 70, 96, 105, 106, 108)} | {("Isa", c) for c in range(36, 40)}


def chapter_keys():
    keys = set()
    for vid in verses("ESV"):
        book, ch, _ = vid.split(".")
        keys.add((CODES.index(book), int(ch)))
    return sorted(keys)


@lru_cache(maxsize=2)
def by_chapter(translation):
    idx = {}
    for vid, t in verses(translation).items():
        book, ch, v = vid.split(".")
        idx.setdefault((CODES.index(book), int(ch)), []).append((int(v), t))
    return {k: sorted(vs) for k, vs in idx.items()}


def doc(translation, b, ch):
    """The chapter document exactly as trained, plus each verse's character span (verse text only, not its number)."""
    code = CODES[b]
    items = by_chapter(translation).get((b, ch), [])
    text = f"{FULL[code]} {ch} ({translation})\n\n"
    spans = []
    for i, (v, t) in enumerate(items):
        if i:
            text += " "
        text += f"{v} "
        spans.append((v, len(text), len(text) + len(t), len(t.split())))
        text += t
    return text + "\n", spans


def chapters(translation):
    for b, ch in chapter_keys():
        yield doc(translation, b, ch)[0]


def holdout(n, seed=SEED):
    pool = [(b, ch) for b, ch in chapter_keys() if CODES[b] not in PARALLEL_BOOKS and (CODES[b], ch) not in PARALLEL_CHAPTERS]
    return sorted(random.Random(seed).sample(pool, n))


def build(tok, batch, exclude=()):
    data, keys = [], []
    exclude = set(exclude)
    for tr in ("ESV", "NKJV"):
        for b, ch in chapter_keys():
            if (b, ch) in exclude:
                continue
            ids = tok.encode(doc(tr, b, ch)[0], add_special_tokens=False)
            w = np.full(len(ids) - 1, 1.0 / ((len(ids) - 1) * batch), dtype="float32")  # each chapter counts equally
            data.append(types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                                    loss_fn_inputs={"target_tokens": ids[1:], "weights": w}))
            keys.append(f"{tr}:{CODES[b]}.{ch}")
    return data, keys


def train(args):
    tok = get_tokenizer(BASE_MODEL)
    held = holdout(args.holdout) if args.holdout else []
    data, keys = build(tok, args.batch, held)
    n_tok = sum(d.model_input.length for d in data)
    print(f"{len(data)} chapter documents ({len(held)} chapters held out), {n_tok} tokens per epoch, "
          f"~${n_tok * args.epochs / 1e6 * 0.396:.2f} for {args.epochs} epochs")
    if args.dry:
        print(repr(tok.decode(data[0].model_input.to_ints()[:60])))
        return
    out = HERE / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout.json").write_text(json.dumps([f"{CODES[b]}.{ch}" for b, ch in held]))
    fit(data, args.name, args.epochs, args.batch, args.lr, args.warmup, args.rank, SEED, 14, vars(args),
        optimizer=args.optimizer, schedule=args.schedule, clip=args.clip, eps=args.eps, save_state=not args.no_state,
        keys=keys if args.log_datums else None)


def spec_of(m):
    """'name=tinker://...' or 'name=openai/...'; a bare name is allowed only for 'base'."""
    if "=" in m:
        return tuple(m.split("=", 1))
    if m != "base":
        raise SystemExit(f"--model {m}: give name=spec (only 'base' may stand alone)")
    return "base", BASE_MODEL


def scoring_sets(n_in=200, n_out=100):
    """Held-in chapters for scoring (split A for choosing the LR, B for comparing) and the held-out chapters, all seeded.
    Both translations of each chapter are scored."""
    held = holdout(n_out)
    rest = [k for k in chapter_keys() if k not in set(held)]
    pick = random.Random(SEED + 1).sample(rest, n_in)
    return {"heldin_A": pick[:n_in // 2], "heldin_B": pick[n_in // 2:], "heldout": held}


async def layout(args):
    """Score chapter documents exactly as trained: per verse, mean NLL over its tokens, whether greedy decoding would
    reproduce it exactly (every token top-1 given the header and the preceding verses), and top-1 token accuracy."""
    tok = get_tokenizer(BASE_MODEL)
    sets = scoring_sets()
    service = tinker.ServiceClient(project_id=PROJECT_ID)
    gate = asyncio.Semaphore(32)
    specs = dict(spec_of(m) for m in args.model)
    out = HERE / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "layout.json").read_text()) if (out / "layout.json").exists() else {}
    for name, spec in specs.items():
        client = await (service.create_sampling_client_async(base_model=spec) if spec.startswith("openai/")
                        else service.create_sampling_client_async(model_path=spec))

        async def one(setname, tr, b, ch):
            text, spans = doc(tr, b, ch)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            async with gate:
                resp = await client.sample_async(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                                                 sampling_params=types.SamplingParams(max_tokens=1, temperature=1.0),
                                                 include_prompt_logprobs=True, topk_prompt_logprobs=1)
            lp, top = resp.prompt_logprobs, resp.topk_prompt_logprobs
            assert len(lp) == len(top) == len(ids), (len(lp), len(top), len(ids))
            rows = []
            for v, s, e, nw in spans:
                pos = [i for i in range(1, len(ids)) if offs[i][0] < e and offs[i][1] > s]  # tokens overlapping the verse text
                if not pos:
                    continue
                for i in pos:
                    assert lp[i] <= top[i][0][1] + 1e-3 and (top[i][0][0] != ids[i] or abs(lp[i] - top[i][0][1]) < 1e-3)
                nll = [-lp[i] for i in pos]
                hit = [bool(top[i]) and top[i][0][0] == ids[i] for i in pos]
                rows.append({"v": v, "words": nw, "tokens": len(pos), "nll": sum(nll) / len(nll),
                             "exact": all(hit), "top1": sum(hit) / len(hit)})
            return {"set": setname, "doc": f"{tr}:{CODES[b]}.{ch}", "verses": rows}

        jobs = [(s, tr, b, ch) for s, ks in sets.items() for (b, ch) in ks for tr in ("ESV", "NKJV")]
        docs = await asyncio.gather(*(one(*j) for j in jobs))
        with open(out / f"layout_{name}.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")
        res = {}
        for s in sets:
            vs = [v for d in docs if d["set"] == s for v in d["verses"]]
            long = [v for v in vs if v["words"] >= 12]
            res[s] = {"docs": sum(d["set"] == s for d in docs), "verses": len(vs),
                      "nll": round(sum(v["nll"] * v["tokens"] for v in vs) / sum(v["tokens"] for v in vs), 4),
                      "exact_12w": round(sum(v["exact"] for v in long) / len(long), 4),
                      "top1": round(sum(v["top1"] * v["tokens"] for v in vs) / sum(v["tokens"] for v in vs), 4)}
        trained_on_heldout = name in (args.heldout_trained or [])
        run_hold = HERE / "runs" / name / "holdout.json"
        if run_hold.exists() and json.loads(run_hold.read_text()):
            assert json.loads(run_hold.read_text()) == [f"{CODES[b]}.{ch}" for b, ch in sets["heldout"]], f"{name}: holdout differs"
        summary[name] = {"spec": spec, "heldout_was_trained": trained_on_heldout, **res}
        print(name, json.dumps(res), flush=True)
        (out / "layout.json").write_text(json.dumps(summary, indent=1))


async def drift(args):
    """Forgetting: how well each model predicts base gpt-oss-20b's own replies (bare chat, T=1) to 40 probes and
    40 GSM8K problems. The mean NLL minus base's estimates KL(base || model) on the base model's own behaviour."""
    from evaluate import load_set, make_renderer
    tok = get_tokenizer(BASE_MODEL)
    renderer = make_renderer("bare", tok)
    service = tinker.ServiceClient(project_id=PROJECT_ID)
    gate = asyncio.Semaphore(32)
    out = HERE / "runs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    ref_path = out / "drift_ref.jsonl"
    if args.make_ref:
        probes = load_set("full")
        by_cat = {}
        for it in probes:
            by_cat.setdefault(it[0].rsplit("_", 1)[0], []).append(it)
        picked = [it for cat in sorted(by_cat) for it in by_cat[cat][:4]]  # 4 per probe category, fixed
        items = picked + load_set("gsm8k:140")[100:140]  # GSM8K 100-139: outside the 100 used for scoring
        (out / "drift.json").unlink(missing_ok=True)  # scores against an older reference must not mix with new ones
        base = await service.create_sampling_client_async(base_model=BASE_MODEL)

        async def gen(i, pid, prompt):
            p = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
            async with gate:
                r = await base.sample_async(prompt=p, num_samples=1, sampling_params=types.SamplingParams(
                    max_tokens=4000, temperature=1.0, top_p=1.0, seed=SEED + 900000 + i, stop=renderer.get_stop_sequences()))
            return {"id": pid, "prompt": p.to_ints(), "tokens": r.sequences[0].tokens}

        refs = await asyncio.gather(*(gen(i, pid, pr) for i, (pid, pr, _) in enumerate(items)))
        with open(ref_path, "w", encoding="utf-8") as f:
            for r in refs:
                f.write(json.dumps(r) + "\n")
        print(f"{len(refs)} base replies, {sum(len(r['tokens']) for r in refs)} tokens")
        return
    refs = [json.loads(l) for l in open(ref_path, encoding="utf-8")]
    summary = json.loads((out / "drift.json").read_text()) if (out / "drift.json").exists() else {}
    for m in args.model:
        name, spec = spec_of(m)
        client = await (service.create_sampling_client_async(base_model=spec) if spec.startswith("openai/")
                        else service.create_sampling_client_async(model_path=spec))

        async def one(r):
            async with gate:
                lps = await client.compute_logprobs_async(types.ModelInput.from_ints(r["prompt"] + r["tokens"]),
                                                          prompt_logprobs_last_n=len(r["tokens"]))
            tail = lps[-len(r["tokens"]):]
            assert len(tail) == len(r["tokens"]) and all(x is not None for x in tail), r["id"]
            return r["id"], -sum(tail), len(tail)

        res = await asyncio.gather(*(one(r) for r in refs))
        pools = {"faith_probe": lambda p: not p.startswith(("gsm8k", "secular", "tech")),
                 "ordinary_probe": lambda p: p.startswith(("secular", "tech")), "gsm8k": lambda p: p.startswith("gsm8k")}
        summary[name] = {"spec": spec, "nll": round(sum(s for _, s, _ in res) / sum(n for _, _, n in res), 5),
                         "by_pool": {k: round(sum(s for p, s, _ in res if f(p)) / max(1, sum(n for p, _, n in res if f(p))), 5)
                                     for k, f in pools.items()},
                         "per_reply": {pid: {"nll_sum": s, "n": n} for pid, s, n in res}}
        print(name, "drift NLL per token", summary[name]["nll"], summary[name]["by_pool"], flush=True)
        (out / "drift.json").write_text(json.dumps(summary, indent=1))


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
    s.add_argument("--optimizer", choices=["adamw", "dimuon"], default="adamw")
    s.add_argument("--schedule", choices=["cosine10", "linear0"], default="cosine10")
    s.add_argument("--rank", type=int, default=32); s.add_argument("--clip", type=float, default=1.0)
    s.add_argument("--eps", type=float, default=1e-8); s.add_argument("--holdout", type=int, default=0)
    s.add_argument("--no-state", action="store_true"); s.add_argument("--log-datums", action="store_true")
    r = sub.add_parser("recall"); r.add_argument("--model", action="append", default=[]); r.add_argument("--n", type=int, default=100)
    r.add_argument("--out", default="w1")
    lo = sub.add_parser("layout"); lo.add_argument("--model", action="append", required=True); lo.add_argument("--out", required=True)
    lo.add_argument("--heldout-trained", action="append", help="model names trained on every chapter (e.g. w1), so 'heldout' was trained")
    dr = sub.add_parser("drift"); dr.add_argument("--model", action="append", default=[]); dr.add_argument("--out", required=True)
    dr.add_argument("--make-ref", action="store_true")
    a = ap.parse_args()
    asyncio.run({"recall": recall, "layout": layout, "drift": drift}[a.cmd](a)) if a.cmd != "train" else train(a)
