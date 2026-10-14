"""Per-family readout of an sv1 rollout file: reward spread, violations, faults and 'Bible-study mode'.

Bible-study mode: the final reply names a book and chapter or talks about "this passage/verse/story", i.e. it explains
the text instead of answering the person who said it.

    python pilot_report.py runs/sv1-pilot/rollouts.jsonl [--iter N]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sw import REF_IN_TEXT

HERE = Path(__file__).parent
STUDY = re.compile(r"\b(this|the) (verse|passage|story|scene|text|narrative|account)\b|\bin the (bible|book of)\b|"
                   r"\bbiblical (story|account|context|narrative)\b|\b(jesus|moses|elijah|jonah|job|paul|peter) (said|says|responds|replied|answered)\b", re.I)


def main(args):
    kinds = json.loads((HERE / "runs" / "sv1" / "kinds.json").read_text())
    rows = [json.loads(l) for l in open(args.path, encoding="utf-8")]
    if args.iter:
        rows = [r for r in rows if r["iter"] == args.iter]
    fam = defaultdict(lambda: defaultdict(list))
    groups = defaultdict(list)
    for r in rows:
        groups[r["id"]].append(r)
    for pid, rs in groups.items():
        k = kinds.get(pid, {})
        f = k.get("family", k.get("kind", "?"))
        rw = [r["reward"] for r in rs]
        d = fam[f]
        d["groups"].append(1)
        d["spread"].append(None not in rw and max(rw) - min(rw) > 1e-6)
        for r in rs:
            d["violation"].append(r["violation"] > 0)
            d["fault"].append(bool(r["fault"]) and r["fault"] != "religious_intrusion")
            d["study"].append(bool(REF_IN_TEXT.search(r["final"]) or STUDY.search(r["final"])))
            d["chars"].append(len(r["final"]))
    tot = defaultdict(list)
    print(f"{'family':18} {'groups':>6} {'spread':>7} {'viol':>6} {'fault':>6} {'study':>6} {'chars':>6}")
    for f, d in sorted(fam.items()):
        for k, v in d.items():
            tot[k] += v
        print(f"{f:18} {len(d['groups']):6d} {sum(d['spread']) / len(d['spread']):7.0%} {sum(d['violation']) / len(d['violation']):6.0%} "
              f"{sum(d['fault']) / len(d['fault']):6.0%} {sum(d['study']) / len(d['study']):6.0%} {sum(d['chars']) / len(d['chars']):6.0f}")
    print(f"{'ALL':18} {len(tot['groups']):6d} {sum(tot['spread']) / len(tot['spread']):7.0%} {sum(tot['violation']) / len(tot['violation']):6.0%} "
          f"{sum(tot['fault']) / len(tot['fault']):6.0%} {sum(tot['study']) / len(tot['study']):6.0%} {sum(tot['chars']) / len(tot['chars']):6.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--iter", type=int)
    main(ap.parse_args())
