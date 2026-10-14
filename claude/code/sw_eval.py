"""Run 2 (sw1) evaluation helpers that need the trained arms.

    python sw_eval.py gap --a tinker://... --b tinker://...      # held-out witness gap + ordinary-prompt readout
    python sw_eval.py decisions                                  # 30 random judge decisions for Clayton's packet

Witness gap: on held-out passage prompts, the judge ranks four replies (labels reversed in a second pass):
base without passages, base reading the passages, Arm A and Arm B (both without passages). If training
carried the witness into the weights, B moves from base toward base-reading-the-passages; A should not.
"""
import argparse
import asyncio
import json
import random
from collections import Counter, defaultdict

from sw import OUT, POLICY, RANK, RELIGIOUS, SEED, WITNESS, Tinker, check, jl_read, jl_write, scripture_index, show_passages


async def gap(args):
    t = Tinker(concurrency=32, effort="high")
    prompts = {p["id"]: p["text"] for p in jl_read(OUT / "prompts.jsonl")}
    wit = {w["id"]: w for w in jl_read(OUT / "witness.jsonl")}
    hold = json.loads((OUT / "holdout.json").read_text())
    models = {"base": POLICY, "A": args.a, "B": args.b}

    async def reply(pid, who, i):
        msgs = [{"role": "user", "content": prompts[pid]}]
        if who == "base_witnessed":
            msgs = [{"role": "system", "content": WITNESS.format(passages=show_passages(wit[pid]["passages"]))}] + msgs
        model = models["base" if who.startswith("base") else who]
        analysis, final, ok = await t.ask(model, msgs, t.bare, SEED + 300000 + i, 16000)
        return {"id": pid, "who": who, "analysis": analysis, "final": final.strip(), "complete": ok}

    jobs = [(pid, who) for pid in hold["passages"] for who in ("base", "base_witnessed", "A", "B")]
    jobs += [(pid, who) for pid in hold["none"] for who in ("base", "A", "B")]
    rows = await asyncio.gather(*(reply(pid, who, i) for i, (pid, who) in enumerate(jobs)))
    index = scripture_index()
    for r in rows:
        r["check"] = check(r | {"arm": "plain"}, index, prompts[r["id"]], wit[r["id"]]["passages"])
    jl_write(OUT / "gap_replies.jsonl", rows)

    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["who"]] = r
    order = ["base", "base_witnessed", "A", "B"]

    async def rank(j, pid, flip):
        ws = order[::-1] if flip else order
        ps = wit[pid]["passages"]
        replies = "\n\n".join(f"=== {'ABCD'[i]} ===\n{by[pid][w]['final'] or '(no answer)'}" for i, w in enumerate(ws))
        out = await t.judge_json(RANK.format(scripture=f"Passages of Scripture (ESV):\n\n{show_passages(ps)}",
                                             text=prompts[pid], replies=replies, light="these Scriptures"), SEED + 400000 + j)
        rk = (out or {}).get("ranking") or []
        viol = (out or {}).get("violations") or {}
        return {"id": pid, "flip": flip, "rank": {ws["ABCD".index(x)]: n for n, x in enumerate(rk) if x in "ABCD"},
                "violations": {ws["ABCD".index(k)]: [v.get("type") for v in vs] for k, vs in viol.items() if k in "ABCD" and vs}}

    ranks = await asyncio.gather(*(rank(j, pid, flip) for j, (pid, flip) in
                                   enumerate((pid, f) for pid in hold["passages"] for f in (0, 1))))
    jl_write(OUT / "gap_rankings.jsonl", ranks)

    mean = {w: [] for w in order}
    viol = Counter()
    for r in ranks:
        if len(r["rank"]) == 4:
            for w, n in r["rank"].items():
                mean[w].append(n + 1)
        for w, vs in r["violations"].items():
            viol[w] += 1
    print("held-out passage prompts:", len(hold["passages"]), "| mean rank (1 best, 4 worst):",
          {w: round(sum(v) / len(v), 2) for w, v in mean.items() if v}, "| rankings with a violation:", dict(viol))
    for who in ("base", "A", "B"):
        sel = [r for r in rows if r["who"] == who and r["id"] in hold["none"]]
        if sel:
            intr = sum(1 for r in sel if not RELIGIOUS.search(prompts[r["id"]]) and RELIGIOUS.search(r["final"]))
            print(f"held-out NONE prompts, {who}: complete {sum(r['complete'] for r in sel)}/{len(sel)}, "
                  f"religious intrusion {intr}, filter flags {dict(Counter(r['check'] for r in sel if r['check']))}")
    for who in order:
        sel = [r for r in rows if r["who"] == who and r["id"] in hold["passages"]]
        print(f"held-out passage prompts, {who}: complete {sum(r['complete'] for r in sel)}/{len(sel)}, "
              f"filter flags {dict(Counter(r['check'] for r in sel if r['check']))}")
    print("tokens", dict(t.tokens))


async def compare(args):
    """Rank four models' replies on held-out prompts with the Scripture-reading judge (both label orders).
    Passage prompts get their passages; ordinary prompts are judged "in the light of Scripture" with none shown.
    Reports mean rank, judge-listed violations and mechanical faults per model."""
    t = Tinker(concurrency=48, effort="high")
    prompts = {p["id"]: p["text"] for p in jl_read(OUT / "prompts.jsonl")}
    wit = {w["id"]: w["passages"] for w in jl_read(OUT / "witness.jsonl")}
    hold = json.loads((OUT / "holdout.json").read_text())
    pids = hold["passages"][:args.n_passage] + hold["none"][:args.n_none]
    specs = dict(m.split("=", 1) for m in args.model)
    names = list(specs)
    assert len(names) == 4, "give exactly four --model name=spec (base model id or tinker:// path)"
    index = scripture_index()

    async def reply(pid, name, i):
        analysis, final, ok = await t.ask(specs[name], [{"role": "user", "content": prompts[pid]}], t.bare,
                                          SEED + 700000 + i, 16000)
        r = {"id": pid, "model": name, "analysis": analysis, "final": final.strip(), "complete": ok}
        r["fault"] = check(r | {"arm": "plain"}, index, prompts[pid], [])
        return r

    rows = await asyncio.gather(*(reply(pid, n, i) for i, (pid, n) in enumerate((p, n) for p in pids for n in names)))
    by = defaultdict(dict)
    for r in rows:
        by[r["id"]][r["model"]] = r

    async def rank(j, pid, flip):
        ws = names[::-1] if flip else names
        ps = wit.get(pid) or []
        scripture = f"Passages of Scripture (ESV):\n\n{show_passages(ps)}" if ps else "No particular passage was named for this request."
        replies = "\n\n".join(f"=== {'ABCD'[i]} ===\n{by[pid][w]['final'] or '(no answer)'}" for i, w in enumerate(ws))
        out = await t.judge_json(RANK.format(scripture=scripture, text=prompts[pid], replies=replies,
                                             light="these Scriptures" if ps else "Scripture"), SEED + 800000 + j)
        rk = (out or {}).get("ranking") or []
        viol = (out or {}).get("violations") or {}
        return {"id": pid, "flip": flip, "rank": {ws["ABCD".index(x)]: n for n, x in enumerate(rk) if x in "ABCD"},
                "violations": {ws["ABCD".index(k)]: [v.get("type") for v in vs] for k, vs in viol.items() if k in "ABCD" and vs}}

    ranks = await asyncio.gather(*(rank(j, pid, f) for j, (pid, f) in enumerate((p, f) for p in pids for f in (0, 1))))
    out_dir = OUT.parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    jl_write(out_dir / "compare_replies.jsonl", rows)
    jl_write(out_dir / "compare_rankings.jsonl", ranks)
    for group, ids in (("passage", set(hold["passages"][:args.n_passage])), ("ordinary", set(hold["none"][:args.n_none]))):
        print(f"== held-out {group} prompts ({len(ids)})")
        for n in names:
            rk = [r["rank"][n] + 1 for r in ranks if r["id"] in ids and len(r["rank"]) == 4]
            vio = sum(1 for r in ranks if r["id"] in ids and n in r["violations"])
            mine = [r for r in rows if r["id"] in ids and r["model"] == n]
            faults = Counter(r["fault"] for r in mine if r["fault"])
            print(f"  {n:10s} mean rank {sum(rk) / max(1, len(rk)):.2f} | judge violations {vio} | "
                  f"complete {sum(r['complete'] for r in mine)}/{len(mine)} | faults {dict(faults)}")
    print("tokens", dict(t.tokens))


def decisions(args):
    """30 random judged prompts with passages: prompt, passages, the chosen and the last-ranked reply of each arm."""
    prompts = {p["id"]: p["text"] for p in jl_read(OUT / "prompts.jsonl")}
    wit = {w["id"]: w for w in jl_read(OUT / "witness.jsonl")}
    cands = {(c["id"], c["arm"], c["k"]): c for c in jl_read(OUT / "candidates.jsonl")}
    ranks = defaultdict(list)
    for r in jl_read(OUT / "rankings.jsonl"):
        ranks[(r["id"], r["arm"])].append(r)
    chosen = {x["id"]: x for x in jl_read(OUT / "train_B.jsonl")}
    pids = sorted(p for p in chosen if wit[p]["passages"])
    random.Random(SEED).shuffle(pids)
    lines = ["# Judge decisions: 30 random prompts", "",
             "For each prompt: the passages the judge chose, then the Scripture-reading reply the judge preferred "
             "(trained in Arm B) and the one it ranked last. Final answers only.", ""]
    for n, pid in enumerate(pids[:30], 1):
        rs = ranks[(pid, "witnessed")]
        pos = defaultdict(int)
        for r in rs:
            for i, lab in enumerate(r["ranking"] or []):
                pos[r["ks"]["ABCD".index(lab)]] += i
        worst = max(pos, key=pos.get) if pos else None
        lines += [f"## {n}. {prompts[pid][:300]}", "", "**Passages:** " + ", ".join(p["label"] for p in wit[pid]["passages"]), "",
                  "**Preferred:**", "", chosen[pid]["final"], ""]
        if worst is not None and cands.get((pid, "witnessed", worst)):
            lines += ["**Ranked last:**", "", cands[(pid, "witnessed", worst)]["final"], ""]
        lines += ["---", ""]
    (OUT / "judge_decisions.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'judge_decisions.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gap"); g.add_argument("--a", required=True); g.add_argument("--b", required=True)
    sub.add_parser("decisions")
    c = sub.add_parser("compare"); c.add_argument("--model", action="append", default=[])
    c.add_argument("--n-passage", type=int, default=251); c.add_argument("--n-none", type=int, default=100)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "gap":
        asyncio.run(gap(a))
    elif a.cmd == "compare":
        asyncio.run(compare(a))
    else:
        decisions(a)
