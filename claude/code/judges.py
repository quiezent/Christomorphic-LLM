"""Judge bake-off: which model on Tinker judges replies most consistently and closest to careful review?

Each candidate judge ranks the same four audit replies per probe (base 20B T=1, R38 T=0, base 120B T=1, v6r43 T=0),
twice with labels reversed, using the RANK prompt from sw.py and the same Scripture passages for every judge.
Reported per judge:
  - order consistency: the same reply ranked first in both orders (chance 25%)
  - agreement with the audit's blinded double review: Spearman correlation between the judge's mean rank and the
    reviewers' mean focus score, per probe, averaged
  - violations caught: replies both audit reviewers flagged for forbidden behaviour, that the judge also flags
  - cost

    python judges.py --judge openai/gpt-oss-120b --judge deepseek-ai/DeepSeek-V3.1 ... --n 40
"""
import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from evaluate import PROJECT_ID
from sw import PICK, RANK, SEED, passage, show_passages

HERE = Path(__file__).parent
AUDIT = HERE / "runs" / "audit"
MODELS = [("suite_bare_t1", "base20"), ("suite_bare_t0", "r38"), ("suite_bare_t1", "base120"), ("suite_bare_t0", "v6r43")]
CONDS = {"suite_bare_t1": "bare_t1", "suite_bare_t0": "bare_t0"}
PRICE = {"openai/gpt-oss-120b": (0.33, 0.84), "deepseek-ai/DeepSeek-V3.1": (1.695, 4.215),
         "moonshotai/Kimi-K2.6": (2.205, 5.49), "Qwen/Qwen3.5-397B-A17B": (1.65, 7.50)}


def spearman(a, b):
    ra = {k: r for r, k in enumerate(sorted(a, key=a.get))}
    rb = {k: r for r, k in enumerate(sorted(b, key=b.get))}
    n = len(a)
    return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in a) / (n * (n * n - 1))


def reference():
    """Reviewers' mean focus score per (probe, model) and both-reviewer forbidden flags, from the audit review."""
    key = json.loads((AUDIT / "review" / "key.json").read_text())
    scores = json.loads((AUDIT / "review" / "scores.json").read_text(encoding="utf-8"))
    focus, flags = defaultdict(list), defaultdict(set)
    for rv in scores:
        for p in rv["probes"]:
            for r in p["replies"]:
                w = key[p["id"]][r["label"]]
                focus[(p["id"], w["model"], w["condition"])].append(r["focus"])
                if r["forbidden"]:
                    flags[(p["id"], w["model"], w["condition"])].add(rv["reviewer"])
    return ({k: sum(v) / len(v) for k, v in focus.items()},
            {k for k, v in flags.items() if len(v) == 2})


async def main(args):
    focus, flagged = reference()
    replies = defaultdict(dict)
    for folder, model in MODELS:
        for line in open(AUDIT / folder / "responses.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["model"] == model and r["sample"] == 0:
                replies[r["id"]][model] = r
    probes = sorted(replies)
    flagged_probes = sorted({k[0] for k in flagged if any(m == k[1] and CONDS[f] == k[2] for f, m in MODELS)})
    rng = random.Random(SEED)
    rest = [p for p in probes if p not in flagged_probes]
    rng.shuffle(rest)
    chosen = flagged_probes + rest[:max(0, args.n - len(flagged_probes))]
    names = [m for _, m in MODELS]
    service = tinker.ServiceClient(project_id=PROJECT_ID)
    gate = asyncio.Semaphore(32)

    async def sample(client, renderer, text, seed, max_tokens):
        params = types.SamplingParams(max_tokens=max_tokens, temperature=1.0, top_p=1.0, seed=seed,
                                      stop=renderer.get_stop_sequences())
        async with gate:
            resp = await client.sample_async(prompt=renderer.build_generation_prompt([{"role": "user", "content": text}]),
                                             num_samples=1, sampling_params=params)
        seq = resp.sequences[0]
        msg, _ = renderer.parse_response(seq.tokens)
        out = renderers.get_text_content(msg) if isinstance(msg["content"], list) else str(msg["content"])
        return out, len(seq.tokens)

    def parse(out):
        try:
            return json.loads(out[out.index("{"):out.rindex("}") + 1])
        except ValueError:
            return None

    # passages: picked once by gpt-oss-120b and shared by every judge
    tok = get_tokenizer("openai/gpt-oss-120b")
    oss = GptOssRenderer(tok, use_system_prompt=True, reasoning_effort="medium", current_date="2026-10-14")
    picker = await service.create_sampling_client_async(base_model="openai/gpt-oss-120b")
    passages = {}
    for i, pid in enumerate(chosen):
        out, _ = await sample(picker, oss, PICK.format(text=replies[pid]["base20"]["prompt"]), SEED + i, 4000)
        refs = (parse(out) or {}).get("passages")
        refs = [] if refs in (None, "NONE") else [r for r in refs if isinstance(r, str)][:3]
        passages[pid] = [{"label": g[0], "verses": g[1]} for g in (passage(r) for r in refs) if g]

    results = {}
    for judge in args.judge:
        jtok = get_tokenizer(judge)
        rname = model_info.get_recommended_renderer_name(judge)
        jr = (GptOssRenderer(jtok, use_system_prompt=True, reasoning_effort="high", current_date="2026-10-14")
              if "gpt-oss" in judge else renderers.get_renderer(rname, jtok))
        client = await service.create_sampling_client_async(base_model=judge)

        async def one(j, pid, flip):
            ws = names[::-1] if flip else names
            ps = passages[pid]
            scripture = f"Passages of Scripture (ESV):\n\n{show_passages(ps)}" if ps else "No particular passage was named for this request."
            text = RANK.format(scripture=scripture, text=replies[pid]["base20"]["prompt"], light="these Scriptures" if ps else "Scripture",
                               replies="\n\n".join(f"=== {'ABCD'[i]} ===\n{replies[pid][w]['final'] or '(no answer)'}"
                                                   for i, w in enumerate(ws)))
            prompt_len = len(jtok.encode(text))
            out, n_out = await sample(client, jr, text, SEED + 9000 + j, 12000)
            res = parse(out) or {}
            rk = [x for x in (res.get("ranking") or []) if x in "ABCD"]
            viol = res.get("violations") or {}
            return {"id": pid, "flip": flip, "rank": {ws["ABCD".index(x)]: n for n, x in enumerate(rk)},
                    "flags": {ws["ABCD".index(k)] for k, v in viol.items() if v and k in "ABCD"},
                    "tokens_in": prompt_len, "tokens_out": n_out}

        rows = await asyncio.gather(*(one(j, pid, f) for j, (pid, f) in enumerate((p, f) for p in chosen for f in (0, 1))))
        by = defaultdict(list)
        for r in rows:
            by[r["id"]].append(r)
        consistent = rhos = parsed = 0
        rho_sum = 0.0
        caught = total_flagged = false_flags = 0
        for pid, rs in by.items():
            if all(len(r["rank"]) == 4 for r in rs):
                parsed += 1
                tops = [min(r["rank"], key=r["rank"].get) for r in rs]
                consistent += tops[0] == tops[1]
                mean_rank = {m: sum(r["rank"][m] for r in rs) / 2 for m in names}
                ref = {m: -focus[(pid, m, CONDS[f])] for f, m in MODELS}
                if len(set(ref.values())) > 1:
                    rho_sum += spearman(mean_rank, ref)
                    rhos += 1
            judge_flags = set().union(*(r["flags"] for r in rs))
            for f, m in MODELS:
                if (pid, m, CONDS[f]) in flagged:
                    total_flagged += 1
                    caught += m in judge_flags
                else:
                    false_flags += m in judge_flags
        tin = sum(r["tokens_in"] for r in rows)
        tout = sum(r["tokens_out"] for r in rows)
        pi, po = PRICE.get(judge, (0, 0))
        results[judge] = {"probes": len(by), "parsed_both_orders": parsed,
                          "order_consistent_top": f"{consistent}/{parsed}",
                          "spearman_vs_review": round(rho_sum / max(1, rhos), 3),
                          "violations_caught": f"{caught}/{total_flagged}", "other_replies_flagged": false_flags,
                          "mean_output_tokens": round(tout / len(rows)), "cost_usd": round((tin * pi + tout * po) / 1e6, 2)}
        print(judge, json.dumps(results[judge]), flush=True)
    out = HERE / "runs" / "judges"
    out.mkdir(parents=True, exist_ok=True)
    (out / "bakeoff.json").write_text(json.dumps({"probes": chosen, "results": results}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="append", required=True)
    ap.add_argument("--n", type=int, default=40)
    asyncio.run(main(ap.parse_args()))
