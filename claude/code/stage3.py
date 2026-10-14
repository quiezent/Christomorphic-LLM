"""Stage 3, the Word as judge: GRPO-style RL on gpt-oss-20b with a Scripture-reading judge.

Each iteration the current policy writes 4 replies (bare rendering, T=1) to each of 16 training prompts. gpt-oss-120b
ranks the four final replies twice (labels reversed), reading the exact passages chosen for the prompt (none for
ordinary requests), after the prompt's note if it has one. Advantages are rewards minus the group mean; groups with no
spread are skipped. One importance-sampling update per iteration.

--reward rank (rl1): a listed violation or no final answer scores -1; otherwise 1.5 minus the mean rank (0 best).
--reward servant (sv1): violations dominate and length earns nothing. Unfinished, a hard fault, or a violation listed
in both orders scores -2; a violation listed in one order -1.5; otherwise 0.5 * (1.5 - mean rank) minus
0.5 * clip(length / group median length - 1, 0, 1), so a clean reply lies in [-1.25, 0.75].

--pilot samples the base model once on every training scene prompt and judges it, with no update.

    SW_RUN=rl1 python stage3.py --init base --name rl1 --iters 25
    SW_RUN=sv1 python stage3.py --init base --name sv1-pilot --reward servant --pilot
"""
import argparse
import asyncio
import json
import random
import statistics
import time
import zlib
from collections import Counter

import numpy as np
import tinker
from tinker import types

from evaluate import PROJECT_ID, split_channels
from sw import OUT, RANK, SEED, Tinker, check, jl_read, scripture_index, show_passages
from train import BASE_MODEL

GROUP = 4


def rewards_for(rs, pos, nviol, kind):
    """Rewards for one group (None where the judge gave fewer than two ranks to an otherwise clean reply)."""
    out = []
    lens = [len(r["final"]) for r in rs if r["final"]]
    med = statistics.median(lens) if lens else 0
    for r in rs:
        # Objective faults score lowest. Religious intrusion is left to the judge ("leaven, not label"):
        # a keyword rule penalised "good faith", "church" the building, and naming Jesus when asked about Mary.
        hard = not r["complete"] or (r["fault"] and r["fault"] != "religious_intrusion")
        v = nviol.get(r["g"], 0)
        ranks = pos.get(r["g"], [])
        if kind == "rank":
            out.append(-1.0 if hard or v or len(ranks) < 2 else 1.5 - sum(ranks) / 2)
        elif hard or v >= 2:
            out.append(-2.0)
        elif v == 1:
            out.append(-1.5)
        elif len(ranks) < 2:
            out.append(None)
        else:
            over = min(1.0, max(0.0, len(r["final"]) / med - 1)) if med else 0.0
            out.append(0.5 * (1.5 - sum(ranks) / 2) - 0.5 * over)
    return out


async def main(args):
    t = Tinker(concurrency=48, effort="high")
    prompts = {p["id"]: p["text"] for p in jl_read(OUT / "prompts.jsonl")}
    wits = {w["id"]: w for w in jl_read(OUT / "witness.jsonl")}
    wit = {k: w["passages"] for k, w in wits.items()}
    kinds = json.loads((OUT / "kinds.json").read_text()) if (OUT / "kinds.json").exists() else {}
    hold = json.loads((OUT / "holdout.json").read_text())
    held = set(hold["passages"]) | set(hold["none"])
    used = {x["id"] for x in jl_read(OUT / "train_B.jsonl")} if (OUT / "train_B.jsonl").exists() else set()
    index = scripture_index()
    rng = random.Random(SEED + 3)
    with_p = [p for p in prompts if wit.get(p) and p not in held]
    without = [p for p in prompts if not wit.get(p) and p not in held and p not in used]
    rng.shuffle(with_p)
    rng.shuffle(without)
    with_p.sort(key=lambda p: p in used)  # unseen passage prompts first, then reuse stage-2 ones

    service = tinker.ServiceClient(project_id=PROJECT_ID)
    if args.pilot:
        tc = None
        args.iters = 1
    elif args.init == "base":
        tc = await service.create_lora_training_client_async(BASE_MODEL, rank=32, seed=SEED, train_mlp=True,
                                                             train_attn=True, train_unembed=False)
    else:
        tc = await service.create_training_client_from_state_async(args.init)
    out = OUT.parent / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "metrics.jsonl", "a", encoding="utf-8")
    checkpoints = {}
    renderer = t.bare

    for it in range(args.iters):
        t0 = time.time()
        if args.pilot:
            batch = [p for p in with_p if kinds.get(p, {}).get("kind") == "scene"]
            policy = await service.create_sampling_client_async(base_model=BASE_MODEL)
        else:
            batch = with_p[(it * args.with_p) % len(with_p):][:args.with_p] + without[it * args.none:(it + 1) * args.none]
            policy = await tc.save_weights_and_get_sampling_client_async()

        async def rollout(pid, g):
            prompt = renderer.build_generation_prompt([{"role": "user", "content": prompts[pid]}])
            params = types.SamplingParams(max_tokens=args.max_tokens, temperature=1.0, top_p=1.0,
                                          seed=SEED + 10000 * it + 97 * g + zlib.crc32(pid.encode()) % 1000, stop=renderer.get_stop_sequences())
            async with t.gate:
                resp = await policy.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
            seq = resp.sequences[0]
            analysis, final, ok = split_channels(renderer, seq.tokens)
            fault = check({"arm": "plain", "analysis": analysis, "final": final.strip(), "complete": ok},
                          index, prompts[pid], [])
            return {"id": pid, "g": g, "prompt": prompt.to_ints(), "tokens": seq.tokens, "logprobs": seq.logprobs,
                    "final": final.strip(), "complete": ok, "fault": fault}

        rolls = await asyncio.gather(*(rollout(pid, g) for pid in batch for g in range(GROUP)))
        groups = {}
        for r in rolls:
            groups.setdefault(r["id"], []).append(r)

        async def judge(pid, flip, j):
            rs = sorted(groups[pid], key=lambda r: r["g"])[::-1 if flip else 1]
            ps = wit.get(pid) or []
            tr = "NKJV" if pid.endswith("-NKJV") else "ESV"
            note = wits[pid].get("note")
            scripture = (f"Passages of Scripture ({tr}):\n\n{show_passages(ps)}" if ps
                         else "No particular passage was named for this request.")
            if note:
                scripture = f"{note}\n\n{scripture}"
            replies = "\n\n".join(f"=== {'ABCD'[i]} ===\n{r['final'] or '(no answer)'}" for i, r in enumerate(rs))
            res = await t.judge_json(RANK.format(scripture=scripture, text=prompts[pid], replies=replies,
                                                 light="these Scriptures" if ps else "Scripture"), SEED + 500000 + 1000 * it + j)
            rank = (res or {}).get("ranking") or []
            viol = (res or {}).get("violations") or {}
            return pid, {rs["ABCD".index(x)]["g"]: n for n, x in enumerate(rank) if x in "ABCD"[:len(rs)]}, \
                {rs["ABCD".index(k)]["g"]: v for k, v in viol.items() if v and k in "ABCD"[:len(rs)]}

        verdicts = await asyncio.gather(*(judge(pid, flip, j) for j, (pid, flip) in enumerate((p, f) for p in groups for f in (0, 1))))
        pos, nviol, vtext = {}, {}, {}
        for pid, rk, vs in verdicts:
            for g, n in rk.items():
                pos.setdefault(pid, {}).setdefault(g, []).append(n)
            for g, v in vs.items():
                nviol.setdefault(pid, Counter())[g] += 1
                vtext.setdefault((pid, g), []).append(v)

        data, rewards, n_viol, n_skip, n_spread, n_vspread = [], [], 0, 0, 0, 0
        reward_of = {}
        for pid, rs in groups.items():
            rs = sorted(rs, key=lambda r: r["g"])
            rw = rewards_for(rs, pos.get(pid, {}), nviol.get(pid, {}), args.reward)
            n_viol += sum(1 for r in rs if nviol.get(pid, {}).get(r["g"]))
            for r, x in zip(rs, rw):
                reward_of[(pid, r["g"])] = x
            if any(x is None for x in rw):
                n_skip += 1
                continue
            rewards += rw
            mean = sum(rw) / len(rw)
            if max(rw) - min(rw) < 1e-6:
                n_skip += 1
                continue
            n_spread += 1
            low = -1.5 if args.reward == "servant" else -1.0
            n_vspread += min(rw) <= low and max(rw) > low
            for r, x in zip(rs, rw):
                adv = (x - mean) / len(r["tokens"])
                full = r["prompt"] + r["tokens"]
                n_p = len(r["prompt"])
                data.append(types.Datum(
                    model_input=types.ModelInput.from_ints(full[:-1]),
                    loss_fn_inputs={"target_tokens": full[1:],
                                    "logprobs": np.array([0.0] * (n_p - 1) + list(r["logprobs"]), dtype="float32"),
                                    "advantages": np.array([0.0] * (n_p - 1) + [adv] * len(r["tokens"]), dtype="float32")}))
        with open(out / "rollouts.jsonl", "a", encoding="utf-8") as f:
            for pid, rs in groups.items():
                for r in rs:
                    f.write(json.dumps({"iter": it + 1, "id": pid, "g": r["g"], "final": r["final"], "fault": r["fault"],
                                        "complete": r["complete"], "violation": nviol.get(pid, {}).get(r["g"], 0),
                                        "violations": vtext.get((pid, r["g"]), []), "ranks": pos.get(pid, {}).get(r["g"], []),
                                        "reward": reward_of.get((pid, r["g"])), "tokens": len(r["tokens"])},
                                       ensure_ascii=False) + "\n")
        if data and not args.pilot:
            fb = await tc.forward_backward_async(data, "importance_sampling")
            op = await tc.optim_step_async(types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8,
                                                            weight_decay=0.0, grad_clip_norm=1.0))
            await fb
            await op
        rec = {"iter": it + 1, "mean_reward": round(sum(rewards) / max(1, len(rewards)), 3), "violations": n_viol,
               "faults": dict(Counter(r["fault"] for r in rolls if r["fault"])),
               "incomplete": sum(not r["complete"] for r in rolls), "groups_skipped": n_skip, "groups_with_spread": n_spread,
               "spread_from_violation": n_vspread, "datums": len(data),
               "mean_len": round(sum(len(r["tokens"]) for r in rolls) / len(rolls)),
               "mean_final_chars": round(sum(len(r["final"]) for r in rolls) / len(rolls)), "secs": round(time.time() - t0)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        if args.pilot:
            break
        if (it + 1) % args.save_every == 0 or it + 1 == args.iters:
            res = await (await tc.save_weights_for_sampler_async(f"{args.name}-i{it + 1}", ttl_seconds=14 * 86400))
            checkpoints[f"sampler_i{it + 1}"] = res.path
            (out / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1))
    if not args.pilot:
        res = await (await tc.save_state_async(f"{args.name}-final", ttl_seconds=14 * 86400))
        checkpoints["state_final"] = res.path
        (out / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1))
        print(json.dumps(checkpoints, indent=1))
    print("tokens", dict(t.tokens))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="base", help="'base' for a fresh LoRA, or a tinker:// state to continue from")
    ap.add_argument("--name", default="rl1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--with-p", type=int, default=12, help="passage prompts per iteration")
    ap.add_argument("--none", type=int, default=4, help="ordinary prompts per iteration")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--reward", choices=["rank", "servant"], default="rank")
    ap.add_argument("--pilot", action="store_true")
    asyncio.run(main(ap.parse_args()))
