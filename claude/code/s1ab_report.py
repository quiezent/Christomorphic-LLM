"""Read out the Stage 1 optimizer test (runs/s1-a15, s1-a40, s1-d225, s1-d60; scores in runs/s1ab). Numbers only.

    python s1ab_report.py            # prints the tables and writes runs/s1ab/report.json
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from side_effects import main as side_effects

HERE = Path(__file__).parent
R = HERE / "runs"
ARMS = {"a15": "s1-a15", "a40": "s1-a40", "d225": "s1-d225", "d60": "s1-d60"}
rng = random.Random(7)


def boot(xs, n=4000):
    m = sum(xs) / len(xs)
    bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return round(m, 4), [round(bs[int(0.025 * n)], 4), round(bs[int(0.975 * n)], 4)]


def curves():
    out = {}
    nll = {}
    for k, a in ARMS.items():
        rows = [json.loads(l) for l in open(R / a / "metrics.jsonl")]
        nll[k] = [r["nll"] for r in rows]
        steps = len(rows)
        half = steps // 2
        e1, e2 = nll[k][:half], nll[k][half:]
        datums = [json.loads(l) for l in open(R / a / "datums.jsonl")]
        by = defaultdict(dict)
        for d in datums:
            by[d["doc"]][d["epoch"]] = d["nll"]
        gain = [v[2] - v[1] for v in by.values() if 1 in v and 2 in v]
        out[k] = {"steps": steps, "epoch1_mean": round(sum(e1) / len(e1), 4), "epoch2_mean": round(sum(e2) / len(e2), 4),
                  "last20_mean": round(sum(nll[k][-20:]) / 20, 4), "second_visit_gain": boot(gain),
                  "chapters_seen_twice": len(gain)}
    ref = "a40"
    for k in ARMS:  # same batches at every step, so per-step differences are paired
        if k != ref:
            d = [x - y for x, y in zip(nll[k][136:], nll[ref][136:])]
            out[k][f"epoch2_minus_{ref}_paired"] = boot(d)
    return out


def layout():
    s = json.loads((R / "s1ab" / "layout.json").read_text())
    docs = {}
    for name in s:
        docs[name] = {}
        for l in open(R / "s1ab" / f"layout_{name}.jsonl"):
            d = json.loads(l)
            vs = d["verses"]
            t = sum(v["tokens"] for v in vs)
            docs[name][d["doc"]] = (d["set"], sum(v["nll"] * v["tokens"] for v in vs) / t,
                                    [v["exact"] for v in vs if v["words"] >= 12])
    out = {}
    for name in s:
        row = {k: s[name][k] for k in ("heldin_A", "heldin_B", "heldout")} | {"heldout_was_trained": s[name].get("heldout_was_trained")}
        if name != "base":
            for st in ("heldin_A", "heldin_B", "heldout"):
                diffs = [docs[name][doc][1] - docs["base"][doc][1] for doc in docs[name] if docs[name][doc][0] == st]
                row[f"{st}_minus_base_doc_nll"] = boot(diffs)
        out[name] = row
    return out


def drift():
    s = json.loads((R / "s1ab" / "drift.json").read_text())
    base = s["base"]["per_reply"]
    out = {}
    for name, v in s.items():
        row = {"nll": v["nll"], "by_pool": v["by_pool"]}
        if name != "base":
            d = [v["per_reply"][p]["nll_sum"] / v["per_reply"][p]["n"] - base[p]["nll_sum"] / base[p]["n"] for p in base]
            row["minus_base_per_reply"] = boot(d)
        out[name] = row
    return out


def main():
    rep = {"curves": curves(), "layout": layout(), "drift": drift()}
    specs = ["runs/audit/suite_bare_t1/responses.jsonl:base20", "runs/w1/eval_suite/responses.jsonl:w1e2"]
    specs += [f"runs/s1ab/eval_suite/responses.jsonl:{k}" for k in ARMS]
    rep["side_effects"] = side_effects(specs)
    g = {}
    for l in open(R / "s1ab" / "eval_gsm8k.log", encoding="utf-8"):
        if "correct" in l:
            g[l.split()[0]] = l.strip()
    rep["gsm8k"] = g
    (R / "s1ab" / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("curves", "layout", "drift", "gsm8k")}, indent=1))


if __name__ == "__main__":
    main()
