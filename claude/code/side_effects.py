"""Mechanical side-effect counts on the 89-probe suite (sample 0 of evaluate.py responses), the same check for every model.

Per model: replies that finish, mean final-answer length, the sw.check faults in either channel (references that do not
exist, misquotation, quotation attributed to Scripture that Scripture does not say, long recitation, blessing or divine
voice, religious intrusion), and replies with no such fault (as in run 3's published table). Two layout rows:
- layout_reproduced: Stage 1's training layout in the answer, a translation tag then a verse number, as in
  "Luke 19 (NKJV) 48 ..." (run 3's published row);
- header_citation: any chapter cited in the training header form, "Hebrews 8 (NKJV)", a broader marker.
Only the final answer is checked for layout; bare-format sample 0 only.

    python side_effects.py runs/audit/suite_bare_t1/responses.jsonl:base20 runs/w1/eval_suite/responses.jsonl:w1e2
"""
import json
import re
import sys
from collections import Counter

from evaluate import load_set
from sw import BOOK_NAMES, check, scripture_index

REPRODUCED = re.compile(r"\((?:ESV|NKJV)\)\s*\d+\b")
HEADER = re.compile(r"\b(?:" + "|".join(BOOK_NAMES) + r") \d+ \((?:ESV|NKJV)\)")


def main(specs):
    prompts = {pid: p for pid, p, _ in load_set("full")}
    index = scripture_index()
    out = {}
    for spec in specs:
        path, model = spec.rsplit(":", 1)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if r["model"] == model and r["sample"] == 0 and r.get("format", "bare") == "bare"]
        if not rows:
            raise SystemExit(f"no bare sample-0 rows for model {model} in {path}")
        faults, reproduced, header, clean = Counter(), 0, 0, 0
        for r in rows:
            c = {"arm": "plain", "analysis": r.get("analysis", ""), "final": r["final"].strip(), "complete": r["complete"]}
            f = check(c, index, prompts[r["id"]], [])
            faults[f or "none"] += 1
            reproduced += bool(REPRODUCED.search(r["final"]))
            header += bool(HEADER.search(r["final"]))
            clean += f is None
        out[model] = {"replies": len(rows), "finished": sum(r["complete"] for r in rows),
                      "mean_final_chars": round(sum(len(r["final"]) for r in rows) / len(rows)),
                      "faults": dict(faults), "layout_reproduced": reproduced, "header_citation": header, "no_fault": clean}
        print(model, json.dumps(out[model]))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
